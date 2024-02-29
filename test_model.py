import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import os
from .models import CNN

model = CNN()

model.load_state_dict(torch.load('zooverzum_model.pth'))
model.eval()

# Step 3: Preprocess the image
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def process_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Step 4: Make predictions
def predict_image(image_path):
    image = process_image(image_path)
    with torch.no_grad():
        output = model(image)
    probabilities = F.softmax(output, dim=1)
    print(probabilities)
    predicted_class = torch.argmax(probabilities).item()
    if predicted_class == 0:
        return "Cat"
    else:
        return "Dog"

# Example usage:
image_path = os.path.join('data', 'test') + "\\8.jpg"
prediction = predict_image(image_path)
print("Prediction:", prediction)
