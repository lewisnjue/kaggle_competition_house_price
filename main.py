
from functools import lru_cache
import pathlib
import os
from PIL import Image
import torchvision.transforms as T
import torch
import torch.nn as nn
"""  """
class CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer_1 = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.layer_2 = nn.Sequential(
        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.layer_3 = nn.Sequential(
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.classifier = nn.Linear(in_features=32*8*8, out_features=3)
  def forward(self, x):
    x = self.layer_1(x)
    x = self.layer_2(x)
    x = self.layer_3(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x
"""  """
# Define image transformation pipeline
image_transform = T.Compose([
    T.Resize((64, 64)),  # Resize to match model input size
    T.ToTensor(),        # Convert to tensor
])

@lru_cache()
def Create_model():
    # Load the model with trained weights
    PATH = os.path.join('model.pth')
    model = CNN()
    model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    model.eval()  # Set model to evaluation mode
    return model

# Initialize the model
model = Create_model()

# Load the image
image_path = 'dog_image_1.jpeg'
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file '{image_path}' not found.")

try:
    # Open and preprocess the image
    image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
    transformed_image = image_transform(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    outputs = model(transformed_image)
    _, predicted_class = torch.max(outputs, 1)  # Get the class with the highest probability

    # Define labels
    labels = ['cat', 'dog', 'fox']

    # Output the prediction
    predicted_label = labels[predicted_class.item()]
    print(f"The model predicts the image is a '{predicted_label}'.")
except Exception as e:
    print(f"An error occurred: {e}")
