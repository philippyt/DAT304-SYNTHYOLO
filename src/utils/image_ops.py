from torchvision import transforms
from PIL import Image
import torch

def preprocess_image(img_path, image_size=640):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img = Image.open(img_path).convert("L")
    return transform(img).unsqueeze(0)