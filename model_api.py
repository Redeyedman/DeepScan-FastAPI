import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("resnet_sdnet_final.pth", map_location=DEVICE))
model.to(DEVICE).eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def analyze_image(img: Image.Image):
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(x)
        _, pred = torch.max(out, 1)

    result = "Cracked" if pred.item() == 1 else "Uncracked"
    severity, orientation = "N/A", "N/A"

    if result == "Cracked":
        gray = np.array(img.convert("L"))
        blur = cv2.GaussianBlur(gray,(5,5),0)
        _, thresh = cv2.threshold(blur,120,255,cv2.THRESH_BINARY_INV)

        ratio = (np.sum(thresh==255)/thresh.size)*100
        severity = "High" if ratio>5 else "Medium" if ratio>2 else "Low"

        contours,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            w,h = cv2.minAreaRect(max(contours,key=cv2.contourArea))[1]
            orientation = "Horizontal" if w>h else "Vertical"

    return result, severity, orientation