import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image

# Define disease names for 15 classes
class_names = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia", "No Finding"
]

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load the model architecture and weights
@st.cache_resource
def load_model():
    model = models.densenet121(pretrained=False)
    num_classes = 15
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)

    import os
    model_path = os.path.join(os.path.dirname(__file__), "densenet121_chestxray_2.pth")
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # Fix key mismatch if saved with Sequential classifier
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("classifier.0."):
            new_key = k.replace("classifier.0.", "classifier.")
        else:
            new_key = k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    return model

model = load_model()

# Prediction function
def predict_image(image):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Streamlit app
st.title("Chest X-ray Disease Classification")
st.write("Upload a Chest X-ray image and get the predicted disease!")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded X-ray.', use_column_width=True)

    predicted_idx = predict_image(image)
    predicted_label = class_names[predicted_idx]

    st.success(f"**Predicted Disease:** {predicted_label}")
