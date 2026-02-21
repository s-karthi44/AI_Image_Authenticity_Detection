import argparse

def evaluate_system(data_path):
    # Main logic: python -m evaluation.evaluate_system
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset/test")
    args = parser.parse_args()
    evaluate_system(args.data)
