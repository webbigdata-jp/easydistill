# Agent Training Framework

A framework for generating, converting, and training agent trajectory data.

## Quick Start

### 1. Generate Agent Trajectory Data
Use `infer/infer.py` to generate agent trajectory data:
```bash
python infer.py --config configs/data_gen_config.yaml
```
### 2. Convert to Training Format
Convert the generated trajectory data to training format using `train/utils/train_data_convertion.py`

### 3. Model Training
Train the agent model using `train/train.py`:
```bash
python train.py --config configs/config.json
```