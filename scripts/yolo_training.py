from ultralytics import YOLO
import torch

def main():
    # Load YOLOv8n model with default pretrained weights
    model = YOLO("yolov8n.pt")
    
    # Auto-detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")


    
    # Train the model
    results = model.train(
        data="data_2class.yaml",
        epochs=10,  # increase later
        imgsz=640,
        batch=16,
        device=device,
        project="runs/detect",
        name="probe_yolov8n",
        patience=10,
        save=True,
        verbose=False
    )
    
    print("Training complete!")

if __name__ == "__main__":
    main()
