import json

class PromptManager:
    """A singleton class to store and manage all prompts."""
    def __init__(self):
        self._prompts_data = {}
        self.is_loaded = False

    def load(self, config_data: dict):
        """
        Loads prompts from a dictionary. This method will be called by infer.py.
        """
        prompts_dict = config_data.get("prompts")
        if not prompts_dict:
            raise ValueError("The 'prompts' section was not found in the configuration file.")

        self._prompts_data = prompts_dict
        self.is_loaded = True
        print("Prompts have been successfully loaded.")

    def __getattr__(self, name: str):
        """
        Implements the `__getattr__` magic method, allowing access
        via dot notation, e.g., `PROMPTS.REACT_SYSTEM_PROMPT`.
        """
        if not self.is_loaded:
            raise AttributeError("Error: Prompts have not been loaded yet! Please call PROMPTS.load() in your main function first.")
        
        if name in self._prompts_data:
            return self._prompts_data[name]
        
        raise AttributeError(f"'{name}' is not a valid prompt name.")

PROMPTS = PromptManager()