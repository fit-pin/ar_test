from ultralytics import YOLO


def main():
    # Load a model
    model = YOLO("model/yolov8m.pt") 

    train = model.train(data="dataset/data.yaml", epochs=100)

if __name__ == "__main__":
    main()
