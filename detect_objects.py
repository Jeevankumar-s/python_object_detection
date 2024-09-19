import torch
from PIL import Image
import matplotlib.pyplot as plt

def detect_objects(image_path):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
    results = model(image_path)
    
    results.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python detect_objects.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    detect_objects(image_path)
