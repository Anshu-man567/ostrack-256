import argparse
import yaml
import os
from pathlib import Path

default_config = {
    'show_dumps': 0,
    'print_stats': 0,
    'en_ece': 1,
    'save_outputs': 0,
    'pretrained_weights': '',
    'hidden_dim': 768,
    'search_size': 256,
    'template_size': 128
}

def get_project_root():
    """Get absolute path to project root directory"""
    return str(Path(__file__).parent.parent)


def load_yaml_config(config_path):
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        dict: Configuration parameters with defaults if file not found
    """
    try:
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config and 'pretrained_weights' in yaml_config and yaml_config['pretrained_weights'] != '':
                # Append project root to pretrained weights path
                yaml_config['pretrained_weights'] = os.path.join(
                    get_project_root(),
                    yaml_config['pretrained_weights']
                )
            if yaml_config:
                default_config.update(yaml_config)
    except FileNotFoundError:
        print(f"Warning: Config file {config_path} not found, using defaults")
    
    return default_config

def print_args(args):
    """
    Print all argument values in a formatted way.

    Args:
        args: Parsed command line arguments
    """
    print("\n" + "="*50)
    print("Configuration:")
    print("="*50)
    for arg, value in vars(args).items():
        print(f"{arg:20s}: {value}")
    print("="*50 + "\n")

def parse_args():
    """
    Parse YAML config first, then override with command line arguments.
    CLI arguments take precedence over YAML config.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    # First load YAML config
    script_dir = str(Path(__file__).parent.parent)
    config_path = script_dir + '/ostrack_appl/ostrack_config.yaml'

    parser = argparse.ArgumentParser(description='OSTrack Testing CLI')
    parser.add_argument('--config', 
                       type=str,
                       default=config_path,
                       help='Path to YAML config file')
    
    # Parse just the config path first
    config_arg, remaining_args = parser.parse_known_args()
    
    # Load config from YAML
    config = load_yaml_config(config_arg.config)
    
    # Add remaining arguments with YAML values as defaults
    parser.add_argument('--show-dumps', 
                       type=int,
                       choices=[0, 1],
                       default=config['show_dumps'],
                       help='Enable visualization dumps')
    
    parser.add_argument('--print-stats',
                       type=int, 
                       choices=[0, 1],
                       default=config['print_stats'],
                       help='Enable printing of debug statistics')
    
    parser.add_argument('--en-ece',
                       type=int,
                       choices=[0, 1],
                       default=config['en_ece'],
                       help='Enable early candidate elimination')
    
    parser.add_argument('--pretrained-weights',
                       type=str,
                       default=config['pretrained_weights'],
                       help='Path to pretrained weights')
    
    parser.add_argument('--hidden-dim',
                       type=int,
                       default=config['hidden_dim'],
                       help='Hidden dimension size for the model')
    
    parser.add_argument('--search-size',
                       type=int,
                       default=config['search_size'],
                       help='Size of search area image')
    
    parser.add_argument('--template-size',
                       type=int,
                       default=config['template_size'],
                       help='Size of template image')
    
    parser.add_argument('--save-outputs',
                       type=int,
                       choices=[0, 1],
                       default=config['save_outputs'],
                       help='Enable saving of output results')
    
    # Parse all arguments
    args = parser.parse_args()
    
    print_args(args)

    return args

if __name__ == "__main__":
    args = parse_args()
    print(f"Running with arguments: {args}")