from ultralytics import YOLO


def main():
    # Load a model
    model = YOLO("./model/yolov8m-obb.pt") 

    train = model.train(data="dataset/data.yaml", epochs=100, name='yolov8m_card')

if __name__ == "__main__":
    main()
