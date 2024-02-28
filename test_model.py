import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import os

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # 2 output neurons for cat and dog

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = CNN()

model.load_state_dict(torch.load('dawg_vs_cat_recognizer.pth'))
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
