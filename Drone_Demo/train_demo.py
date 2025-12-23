from ultralytics import YOLO

if __name__ == '__main__':
    # 1. Load the base model (YOLOv8 nano - smallest and fastest)
    # This will download the pretrained weights from COCO if not present
    print("Loading model...")
    model = YOLO('yolov8n.pt')

    # 2. Start the training process
    print("Starting training...")
    results = model.train(
        data='dataset/data.yaml',
        epochs=20,                 # Number of training cycles (20 is enough for a quick demo)
        imgsz=640,                 # Standard image size for YOLO
        batch=2,                   # Process 2 images at a time (low memory usage)
        name='sar_demo_model'      # The name of the folder where results will be saved
    )

    print("Training finished successfully!")