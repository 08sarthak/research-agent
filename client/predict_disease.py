import torch
import timm
from PIL import Image
import torchvision.transforms as transforms
import os

# =========================
# Classes
# =========================
CLASS_NAMES = [
    'Melanocytic nevi',
    'Melanoma',
    'Benign keratosis-like lesions',
    'Basal cell carcinoma',
    'Actinic keratoses',
    'Vascular lesions',
    'Dermatofibroma'
]

# =========================
# Model path
# =========================
MODEL_PATH = r"D:\work(tryzent)\research-agent\client\best_model_s3.pth"

# =========================
# Load model (EfficientNet-B4)
# =========================
print("🔍 Loading model from:", MODEL_PATH)

checkpoint = torch.load(MODEL_PATH, map_location='cpu')

# 👉 Create SAME model used in training
model = timm.create_model(
    'efficientnet_b4',
    pretrained=False,
    num_classes=len(CLASS_NAMES)
)

# 👉 Load weights (handle both formats)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()

print("✅ Model loaded successfully")


# =========================
# Preprocessing
# =========================
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# =========================
# Prediction
# =========================
def predict_disease(image_path: str | Image.Image):
    if isinstance(image_path, Image.Image):
        image = image_path.convert('RGB')
    else:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path).convert('RGB')

    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)

        confidence, pred = torch.max(probs, 0)

    disease = CLASS_NAMES[pred.item()]
    confidence = int(confidence.item() * 100)

    return disease, confidence


# =========================
# Test
# =========================
# if __name__ == "__main__":
#     img = "D:\work(tryzent)\research-agent\client\acne-cystic-122.jpg"

#     if os.path.exists(img):
#         result = predict_disease(img)
#         print("Prediction:", result)
#     else:
#         print("⚠️ Test image not found")