import json
import os

class PretrainedConfig:
    """Base class for model configurations."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        config_file = os.path.join(save_directory, "config.json")
        with open(config_file, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):
        config_file = os.path.join(pretrained_model_name_or_path, "config.json")
        with open(config_file, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)
