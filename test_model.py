import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import os
from models import CNN

animals = ["Cat", "Dog", "Fish", "Fox", "Monkey", "Parrot", "Rabbit", "Snake"]

model = CNN()

model.load_state_dict(torch.load('zooverzum_model.pth'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def process_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def predict_image(image_path):
    image = process_image(image_path)
    with torch.no_grad():
        output = model(image)
    probabilities = F.softmax(output, dim=1)
    print(probabilities)
    predicted_class = torch.argmax(probabilities).item()
    return animals[predicted_class]

image_path = os.path.join('data', 'test') + "\\rabbit1.jpg"
prediction = predict_image(image_path)
print("Prediction:", prediction)
