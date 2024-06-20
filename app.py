import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

# Load your model
def load_model(model_path):
    model = models.vgg16(pretrained=False)
    n_inputs = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Sequential(
        torch.nn.Linear(n_inputs, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(256, 2),  # assuming 2 classes
        torch.nn.LogSoftmax(dim=1)
    )
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Load the model state_dict
    model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()
    return model

# Preprocessing function
def preprocess_image(image):
    # Convert the image to RGB if it's not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image

# Prediction function
def predict(model, image):
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        return preds

# Streamlit app interface
def main():
    st.title("Image Classification with VGG16")
    
    model_path = "vgg16-chest-4.pth"  # Replace with your model path
    model = load_model(model_path)
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        st.write("")
        st.write("Classifying...")
        
        image = preprocess_image(image)
        
        predictions = predict(model, image)
        
        if predictions.item() == 0:
            st.write("Prediction: Image is Normal")
        else:
            st.write("Prediction: Pneumonia")

if __name__ == "__main__":
    main()
