import hydra
import sys

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg):
    print("Hello World")
    sys.stdout.flush()

if __name__ == "__main__":
    main()