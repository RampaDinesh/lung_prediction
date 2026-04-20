from torchvision import transforms
from torchvision.models import resnet18,ResNet18_Weights
import torch
from torch import nn
import numpy as np
from PIL import Image
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
device = "cuda" if torch.cuda.is_available() else "cpu"
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
image1 = Image.open(r"E:\datasets\archive\chest_xray\val\PNEUMONIA\person1949_bacteria_4880.jpeg").convert("RGB")
image = transform(image1)
image = image.unsqueeze(0)
image = image.to(device)


model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

    
model.load_state_dict(torch.load("resent182.pth", map_location=device))
model.to(device=device)

model.eval()
image_np =cv2.resize(np.array(image1),(224,224))/255.0
input_tensor = image

with torch.no_grad():
    output = model(input_tensor)
    probs = torch.softmax(output,dim=1)
    predicted = (probs[:,1]>0.9).long()
    if predicted.item() ==1:
       
        print("PNEUMONIA detected")
    else:
        print("Normal ")



if predicted.item()==1:
        target_layer = model.layer4[-1]
        cam = GradCAM(model,target_layers=[target_layer])
        targets=[ClassifierOutputTarget(1)]
        grayscale_cam = cam(input_tensor,targets)
        cam_image = show_cam_on_image(image_np,grayscale_cam[0],use_rgb=True)
        plt.figure(figsize=(10,5))

        plt.subplot(1,2,1)
        plt.imshow(image_np)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1,2,2)
        plt.imshow(cam_image)
        plt.title("Grad-CAM (Model Focus)")
        plt.axis('off')

        plt.show()