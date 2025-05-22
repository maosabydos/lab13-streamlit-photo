import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from model import CNN  # Make sure model.py is in the same directory

device = torch.device("cpu")

model = CNN().to(device)
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.eval()

class_names = ["Cat", "Dog"]

transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])

st.title("🐱🐶 Cat vs Dog Classifier")
st.write("Upload a cat or dog image and see what the model predicts.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        label = class_names[predicted.item()]
        st.markdown(f"### 🧠 Prediction: **{label}**")

