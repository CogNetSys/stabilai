import streamlit as st
import torch
import torch.nn.functional as F
from models.surge_collapse_net import SurgeCollapseNet
from transformers import RobertaTokenizer, RobertaModel

# Load StableMax and OrthogonalGrad if necessary

@st.cache(allow_output_mutation=True)
def load_model(model_path, device):
    model = SurgeCollapseNet(input_size=128, hidden_size=256, output_size=128)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def main():
    st.title("StabilAI - AI Model Optimization Dashboard")
    st.write("Upload your model and data to optimize and monitor training dynamics.")

    model_path = st.text_input("Path to Trained Model (.pth)", "models/best_model.pth")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if st.button("Load Model"):
        try:
            model = load_model(model_path, device)
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}")

    st.write("### Enter Input Data for Prediction")
    input_data = st.text_area("Input Data (comma-separated numbers, e.g., 0.1, 0.2, ...)", "")

    if st.button("Predict"):
        try:
            input_tensor = torch.tensor([float(x.strip()) for x in input_data.split(',')]).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                prob = probs[0][pred].item()
            label = "Anomaly" if pred == 1 else "Normal"
            st.write(f"**Prediction:** {label}")
            st.write(f"**Probability:** {prob:.4f}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

if __name__ == "__main__":
    main()
