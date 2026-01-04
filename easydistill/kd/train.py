# Copyright 2024 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import json
import argparse
import logging
import os
from jinja2 import Environment, BaseLoader, FileSystemLoader
from datasets import load_dataset,Dataset
from typing import Optional, Dict, Union, List
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase,AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer,SFTConfig
import torch
import jsonlines
import numpy as np
import torch.nn.functional as F


class DistillSFTTrainer(SFTTrainer):
    """
    Memory-efficient Knowledge Distillation Trainer.
    
    Instead of creating a full (batch, seq, vocab_size) tensor for teacher logits,
    this implementation only stores Top-K indices and probabilities, significantly
    reducing memory usage from ~600MB to ~40KB per sample (for vocab_size=152k, K=20).
    """

    def __init__(
        self,
        logits_dir: str = None,  
        teacher_vocab_size: int = None,  
        kd_ratio: float = 0.5,    
        max_seq_length: int = 1024,
        distillation_type: str = "forward_kld",
        top_logits_num: int = 10,  # Top-K number from config
        **kwargs
    ):
        super().__init__(**kwargs)
        self.logits_dir = logits_dir
        self.teacher_vocab_size = teacher_vocab_size
        self.kd_ratio = kd_ratio
        self.max_seq_length = max_seq_length
        self.distillation_type = distillation_type
        self.top_logits_num = top_logits_num
        
        # Load teacher logits from JSONL file
        self.teacher_logits = []
        if self.logits_dir:
            with jsonlines.open(self.logits_dir) as reader:
                for obj in reader:
                    self.teacher_logits.append(obj)
            logging.info(f"Loaded {len(self.teacher_logits)} teacher logits from {self.logits_dir}")


    def _load_teacher_logits(
        self, 
        batch_size: int, 
        it: int, 
        dp_rank: int, 
        device: torch.device, 
        no_model_batch: Dict
    ) -> tuple:
        """
        Load teacher logits as sparse representation (indices + probabilities).
        
        Returns:
            tuple: (indices_tensor, probs_tensor)
                - indices_tensor: (batch, seq, K) - Token IDs of Top-K
                - probs_tensor: (batch, seq, K) - Probabilities of Top-K
        """
        total_samples = len(self.teacher_logits)
        
        # Use modulo to handle multiple epochs
        # global_step increases across epochs, so we cycle through the data
        start_idx = (it * batch_size) % total_samples
        end_idx = start_idx + batch_size
        
        # Handle wrap-around at epoch boundary
        if end_idx <= total_samples:
            loaded_data = self.teacher_logits[start_idx:end_idx]
        else:
            # Wrap around: take remaining from end + beginning
            loaded_data = (
                self.teacher_logits[start_idx:total_samples] + 
                self.teacher_logits[0:end_idx - total_samples]
            )
        
        # Find max sequence length in this batch
        max_len_in_batch = max(len(sample) for sample in loaded_data) if loaded_data else 0
        actual_seq_len = min(max_len_in_batch, self.max_seq_length)
        
        # Initialize tensors with padding
        # Using 0 as padding token ID (safe for gather operation)
        batch_indices = torch.zeros(
            (len(loaded_data), actual_seq_len, self.top_logits_num), 
            dtype=torch.long, 
            device=device
        )
        batch_probs = torch.zeros(
            (len(loaded_data), actual_seq_len, self.top_logits_num), 
            dtype=torch.bfloat16, 
            device=device
        )
        
        for b_idx, sample in enumerate(loaded_data):
            seq_len = min(len(sample), actual_seq_len)
            for s_idx in range(seq_len):
                logit_dict = sample[s_idx]
                # Extract token IDs and probabilities from dict
                ids = list(map(int, logit_dict.keys()))
                probs = list(logit_dict.values())
                
                # Handle variable K (take min of actual and expected)
                k = min(len(ids), self.top_logits_num)
                batch_indices[b_idx, s_idx, :k] = torch.tensor(ids[:k], dtype=torch.long, device=device)
                batch_probs[b_idx, s_idx, :k] = torch.tensor(probs[:k], dtype=torch.bfloat16, device=device)
        
        # Apply sequence shift (align with labels)
        labels = no_model_batch.get('label')
        if labels is not None:
            indices_shifted = self._shift_tensor_right_sparse(batch_indices, labels, pad_value=0)
            probs_shifted = self._shift_tensor_right_sparse(batch_probs, labels, pad_value=0.0)
        else:
            indices_shifted = batch_indices
            probs_shifted = batch_probs
        
        return indices_shifted, probs_shifted


    @staticmethod
    def _shift_tensor_right_sparse(
        inputs: torch.Tensor, 
        labels: torch.Tensor, 
        pad_value: float = 0.0
    ) -> torch.Tensor:
        """
        Shift tensor right based on label positions.
        Works with sparse representation (batch, seq, K).
        
        Args:
            inputs: (batch, seq, K) tensor to shift
            labels: (batch, seq) labels tensor, -100 indicates padding/prompt
            pad_value: Value to use for padding after shift
            
        Returns:
            Shifted tensor of same shape
        """
        batch_size, seqlen, k_dim = inputs.shape
        device = inputs.device
        
        # Find first non-padding position in labels
        labels_ne = labels != -100
        shift_distances = torch.argmax(labels_ne.int(), dim=1)
        
        # Create index for gather
        idx = torch.arange(seqlen, device=device).unsqueeze(0).expand(batch_size, seqlen)
        shifted_idx = idx - shift_distances.unsqueeze(1)
        mask = shifted_idx >= 0
        shifted_idx = shifted_idx.clamp(min=0)
        
        # Expand for K dimension
        shifted_idx_expanded = shifted_idx.unsqueeze(2).expand(-1, -1, k_dim)
        
        # Gather and apply mask
        gathered = torch.gather(inputs, 1, shifted_idx_expanded)
        mask_expanded = mask.unsqueeze(2).expand(-1, -1, k_dim)
        
        return torch.where(mask_expanded, gathered, torch.full_like(gathered, pad_value))


    def _compute_white_box_distillation_loss(
        self, 
        student_logits: torch.Tensor, 
        teacher_indices: torch.Tensor, 
        teacher_probs: torch.Tensor, 
        labels: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute KL divergence loss using sparse teacher representation.
        
        This method avoids creating full (batch, seq, vocab) tensors for teacher,
        instead using torch.gather to extract only the necessary logits.
        
        Args:
            student_logits: (batch, seq, vocab) - Full student logits
            teacher_indices: (batch, seq, K) - Top-K token indices from teacher
            teacher_probs: (batch, seq, K) - Top-K probabilities from teacher
            labels: (batch, seq) - Labels for masking
            
        Returns:
            Scalar loss value
        """
        # Align sequence lengths
        min_len = min(student_logits.size(1), teacher_indices.size(1))
        student_logits = student_logits[:, :min_len, :]
        teacher_indices = teacher_indices[:, :min_len, :]
        teacher_probs = teacher_probs[:, :min_len, :]
        
        # Create mask from labels
        if labels is not None:
            mask = (labels[:, :min_len] != -100).float()
        else:
            mask = torch.ones(
                (student_logits.size(0), min_len), 
                device=student_logits.device
            )
        
        # Compute student log probabilities for full vocabulary
        student_log_probs_all = F.log_softmax(student_logits, dim=-1)
        
        # Extract student log probs only for teacher's Top-K tokens
        student_log_probs_selected = torch.gather(
            student_log_probs_all, 
            dim=-1, 
            index=teacher_indices
        )
        
        # Teacher log probs (with numerical stability)
        teacher_log_probs_selected = torch.log(teacher_probs.clamp(min=1e-10))
        
        if self.distillation_type == "forward_kld":
            # Forward KLD: D_KL(Teacher || Student)
            # = sum(P_teacher * log(P_teacher / P_student))
            # = sum(P_teacher * (log(P_teacher) - log(P_student)))
            pointwise_loss = teacher_probs * (teacher_log_probs_selected - student_log_probs_selected)
            
        elif self.distillation_type == "reverse_kld":
            # Reverse KLD: D_KL(Student || Teacher)
            # For sparse teacher (Top-K only), we compute on the Top-K support
            # = sum(P_student_topk * (log(P_student_topk) - log(P_teacher_topk)))
            # Note: This is an approximation since we only consider Top-K tokens
            student_probs_selected = torch.exp(student_log_probs_selected)
            pointwise_loss = student_probs_selected * (student_log_probs_selected - teacher_log_probs_selected)
            
        else:
            raise ValueError(
                f"Unsupported distillation type: {self.distillation_type}. "
                "Use 'forward_kld' or 'reverse_kld'"
            )
        
        # Sum over K dimension -> (batch, seq)
        loss_per_token = pointwise_loss.sum(dim=-1)
        
        # Apply mask and compute mean
        masked_loss = loss_per_token * mask
        loss = masked_loss.sum() / (mask.sum() + 1e-10)
        
        return loss


    def compute_loss(
        self, 
        model: PreTrainedModel, 
        inputs: Dict[str, torch.Tensor], 
        return_outputs: bool = False, 
        num_items_in_batch: int = None
    ):
        """
        Compute combined LM loss and distillation loss.
        """
        outputs = model(**inputs)
        lm_loss = outputs.loss
        
        if self.logits_dir:
            # Load sparse teacher logits
            teacher_indices, teacher_probs = self._load_teacher_logits(
                batch_size=inputs['input_ids'].size(0),
                it=self.state.global_step,
                dp_rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
                device=model.device,
                no_model_batch={'label': inputs.get('labels', None)}
            )
            
            # Compute distillation loss
            distil_loss = self._compute_white_box_distillation_loss(
                student_logits=outputs.logits,
                teacher_indices=teacher_indices,
                teacher_probs=teacher_probs,
                labels=inputs.get('labels', None)
            )
            
            # Combine losses
            total_loss = (1 - self.kd_ratio) * lm_loss + self.kd_ratio * distil_loss
            
            # Log losses for debugging (optional)
            if self.state.global_step % 100 == 0:
                logging.info(
                    f"Step {self.state.global_step}: "
                    f"LM Loss={lm_loss.item():.4f}, "
                    f"Distil Loss={distil_loss.item():.4f}, "
                    f"Total Loss={total_loss.item():.4f}"
                )
        else:
            total_loss = lm_loss
            
        return (total_loss, outputs) if return_outputs else total_loss


def formatting_func(examples):
    env = Environment(loader=BaseLoader())
    try:
        message = {"content": examples["instruction"],"output":examples["output"]}
        full_text = template.render(
            message=message,
            add_generation_prompt=False,
            add_output=True
        )
        return full_text
    except Exception as e:
        logging.warning(f"Error processing sample: {str(e)}")
        return ""


def train(config):
    dataset = load_dataset("json", data_files=config["dataset"]["labeled_path"])
    
    student_tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["student"], 
        trust_remote_code=True
    )
    student_model = AutoModelForCausalLM.from_pretrained(
        config["models"]["student"],
        trust_remote_code=True
    )

    global template
    full_path = config["dataset"]["template"]
    template_dir = os.path.dirname(full_path)
    template_file = os.path.basename(full_path)
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_file)
    training_arguments = SFTConfig(**config["training"])
    
    try:
        job_type = config["job_type"]
        if "kd_black_box" in job_type:
            dataset = dataset.shuffle(seed=config["dataset"]["seed"])
            trainer = SFTTrainer(
                model=student_model,
                processing_class=student_tokenizer,
                args=training_arguments,
                train_dataset=dataset["train"],
                formatting_func=formatting_func
            )
        elif "kd_white_box" in job_type:
            teacher_vocab_size = json.load(
                open(os.path.join(config["models"]["teacher"], 'config.json'))
            )['vocab_size']
            
            # Get top_logits_num from config (distillation section or inference section)
            top_logits_num = config["distillation"].get(
                "top_logits_num", 
                config.get("inference", {}).get("top_logits_num", 10)
            )
            
            trainer = DistillSFTTrainer(
                logits_dir=config["dataset"]["logits_path"],
                teacher_vocab_size=teacher_vocab_size,
                kd_ratio=config["distillation"]["kd_ratio"], 
                max_seq_length=config["distillation"]["max_seq_length"],
                distillation_type=config["distillation"].get("distillation_type", "forward_kld"),
                top_logits_num=top_logits_num,
                model=student_model,
                processing_class=student_tokenizer,
                args=training_arguments,
                train_dataset=dataset["train"],
                formatting_func=formatting_func
            )
        else:
            logging.error(f"Invalid job type: {job_type}")
            raise ValueError(f"Invalid job type: {job_type}")
    except ValueError as e:
        logging.error(f"Training job terminated: {e}")
        return
        
    trainer.train()
    trainer.save_model(config["training"]["output_dir"])
    student_tokenizer.save_pretrained(config["training"]["output_dir"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to the json config file')
    args = parser.parse_args()
    config = json.load(open(args.config))
    train(config)  


if __name__ == "__main__":
    main()


