# Config loader for AutoML system

import yaml


def load_config(path='config.yaml'):
    """Loads configuration from a YAML file."""
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config