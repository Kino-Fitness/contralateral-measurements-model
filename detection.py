import torch
from PIL import Image

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', )  # or yolov5m, yolov5l, yolov5x, custom

from ultralytics import YOLO
import numpy as np
from PIL import Image
model = YOLO("yolov8n.pt")
# Images
img = 'saved/images/frontpic2.png'

# Inference
results = model(img, classes=0)  # or specify custom classes
results[0]
type(results[0])

results = model(img)  # results list
boxes = results[0].boxes
coords = boxes.xyxy.tolist()[0]
# Crop image using the coordinates
img = Image.open(img)
cropped_image = img.crop(coords)

# Display the cropped image
cropped_image.show()

