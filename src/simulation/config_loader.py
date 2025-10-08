"""
Configuration loader for simulation.
"""

import yaml
from pathlib import Path
from typing import Dict


def load_config(config_path: str = None) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file (defaults to config/config.yaml)
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        root_dir = Path(__file__).parent.parent.parent
        config_path = root_dir / "config" / "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


if __name__ == "__main__":
    config = load_config()
    print("Loaded configuration:")
    for key in config.keys():
        print(f"  {key}: {type(config[key])}")
