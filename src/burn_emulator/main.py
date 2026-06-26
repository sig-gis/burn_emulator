import argparse
import yaml

from burn_emulator.train import train
from burn_emulator.test import test


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-m", "--method", default="train", choices=["train", "test"])
    parser.add_argument("-c", "--config", action="append", required=True)
    args = parser.parse_args()

    configs = {}
    for config_path in args.config:
        with open(config_path) as f:
            configs |= yaml.safe_load(f)
    
    match args.method:
        case "train":
            train(**configs)
        case "test":
            test(**configs)

if __name__ == "__main__":
    main()