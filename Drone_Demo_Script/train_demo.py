from ultralytics import YOLO

if __name__ == '__main__':

    print("Loading model...")
    model = YOLO('yolov8n.pt')

    # Start the training process
    print("Starting training...")
    results = model.train(
        data='dataset/data.yaml',
        epochs=20,                 # Number of training cycles (20 is enough for a quick demo)
        imgsz=640,
        batch=2,                   # Process 2 images at a time (low memory usage)
        name='sar_demo_model'
    )

    print("Training finished successfully!")