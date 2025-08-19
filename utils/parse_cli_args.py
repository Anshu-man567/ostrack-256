import argparse
from enum import Enum

def parse_args():
    """
    Parse command line arguments for OSTrack testing.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='OSTrack Testing CLI')
    
    parser = argparse.ArgumentParser(description='OSTrack Testing CLI')
    
    parser.add_argument('--show-dumps', 
                       type=int,
                       choices=[0, 1],
                       default=0,
                       help='Enable visualization dumps')
    
    parser.add_argument('--print-stats',
                       type=int, 
                       choices=[0, 1],
                       default=0,
                       help='Enable printing of debug statistics')
    
    parser.add_argument('--en-ece',
                       type=int,
                       choices=[0, 1], 
                       default=0,
                       help='Enable early candidate elimination')

    # parser.add_argument('--ece-ratio',
    #                    type=float,
    #                    default=1.0,
    #                    help='Ratio for early candidate elimination (0.0-1.0)')
    
    # parser.add_argument('--pretrained-weights',
    #                    type=str,
    #                    default='',
    #                    help='Path to input data directory')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print(f"Running with arguments: {args}")