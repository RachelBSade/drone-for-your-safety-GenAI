from ultralytics import YOLO

def main():
    # Load the base YOLO model
    model = YOLO('yolo11n.pt')

    # Start the training process
    model.train(
        data='data.yaml',
        epochs=10,
        imgsz=640,
        batch=16,
        name='fire_detection_demo',
        device='cpu'
    )

if __name__ == '__main__':
    main()