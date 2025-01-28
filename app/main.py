# app/main.py

import argparse
from app.training import main as training_main

def parse_arguments():
    parser = argparse.ArgumentParser(description="StabilAI Training and Evaluation")
    parser.add_argument('--config', type=str, default='app/config.py', help='Path to configuration file')
    return parser.parse_args()

def main():
    args = parse_arguments()
    training_main(config_path=args.config)

if __name__ == "__main__":
    main()
