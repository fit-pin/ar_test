import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='모델 이름')
args = parser.parse_args()

def main():
    model = YOLO(f"./model/{args.model}") 

    model.train(data="dataset/data.yaml", epochs=100, name=f"{args.model}")

if __name__ == "__main__":
    main()
