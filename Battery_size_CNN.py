from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import joblib
import os
import requests
import openai


app = FastAPI()

# ✅ Load API keys securely
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # Secure API Key

# ✅ Configure OpenAI Client (DeepSeek API)
client = openai.OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")



# ✅ Fix CORS Issue
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all domains (change this for security in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Get the current directory of the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model
MODEL_PATH = os.path.join(BASE_DIR, "Battery_size_CNN.h5")  
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# Load the saved scaler for input features (X)
SCALER_X_PATH = os.path.join(BASE_DIR, "scaler_X.pkl")
if os.path.exists(SCALER_X_PATH):
    scaler_X = joblib.load(SCALER_X_PATH)
else:
    raise FileNotFoundError(f"Scaler file not found: {SCALER_X_PATH}")

# Load the saved scaler for output labels (Y)
SCALER_Y_PATH = os.path.join(BASE_DIR, "scaler_Y.pkl")
if os.path.exists(SCALER_Y_PATH):
    scaler_Y = joblib.load(SCALER_Y_PATH)
else:
    raise FileNotFoundError(f"Scaler file not found: {SCALER_Y_PATH}")




# Define the input data structure
class BatteryInput(BaseModel):
    Length_pack: float
    Width_pack: float
    Height_pack: float
    Energy: float
    Total_Voltage: float

@app.get("/")
def home():
    return {"message": "Battery CNN Prediction API is running"}


def analyze_with_deepseek(predictions, input_data):
    if not DEEPSEEK_API_KEY:
        return {"error": "DeepSeek API key is missing!"}

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI expert in battery optimization. "
                        "Explain concepts clearly and use markdown format. "
                        "Avoid LaTeX formatting (no \\[ \\], \\text{}, \\frac{}). "
                        "Do NOT use markdown headings (#). "
                        "Instead, use bold labels like **Battery Pack Specs**, and format formulas like `E = P × t`."
                    )
                },
                {
                    "role": "user",
                    "content": f"""
**🔋 Battery Pack Specifications**
- **Dimensions**: {input_data.Length_pack:.0f} mm × {input_data.Width_pack:.0f} mm × {input_data.Height_pack:.0f} mm  
- **Volume**: {input_data.Length_pack * input_data.Width_pack * input_data.Height_pack / 1e9:.2f} m³  
- **Energy**: {input_data.Energy:.2f} kWh  
- **Voltage**: {input_data.Total_Voltage:.2f} V  

**📦 Predicted Cell Specifications**
- **Cell Dimensions**: {predictions['Length_cell']:.0f} mm × {predictions['Width_cell']:.0f} mm × {predictions['Height_cell']:.0f} mm  
- **Cell Volume**: {predictions['Length_cell'] * predictions['Width_cell'] * predictions['Height_cell'] / 1e6:.3f} L  
- **Power Density**: {predictions['Power_density']:.2f} Wh/kg  

Please analyze:
- Physical fit between pack and cells
- Energy & power density consistency
- Suggestions to improve configuration

Use **clear markdown**, plain equations like `Energy Density = Energy / Volume`, and do not use large heading styles.
"""
                }
            ],
            max_tokens=500
        )

        return response.choices[0].message.content

    except openai.OpenAIError as e:
        return {"error": "DeepSeek AI failed", "message": str(e)}


@app.post("/predict/")
def predict(input_data: BatteryInput):
    try:
        # Convert input data to NumPy array
        input_array = np.array([[input_data.Length_pack, input_data.Width_pack, 
                                 input_data.Height_pack, input_data.Energy, 
                                 input_data.Total_Voltage]])
        
        # Normalize input using the saved scaler
        input_scaled = scaler_X.transform(input_array)
        
        # Reshape for CNN input (batch_size, features, channels)
        input_scaled = input_scaled.reshape((input_scaled.shape[0], input_scaled.shape[1], 1))
        
        # Make prediction
        prediction_scaled = model.predict(input_scaled)
        
        # Convert prediction back to original scale
        prediction_original = scaler_Y.inverse_transform(prediction_scaled)
        prediction_original = np.where(prediction_original < 0, -prediction_original, prediction_original)
        # Prepare response
        response = {
            "Length_cell": float(prediction_original[0][0]),
            "Width_cell": float(prediction_original[0][1]),
            "Height_cell": float(prediction_original[0][2]),
            "Power_density": float(prediction_original[0][3])
        }
        
# ✅ Send predictions to ChatGPT for optimization suggestions
        chatgpt_response = analyze_with_deepseek(response, input_data)

        return {"predictions": response, "deepseek_analysis": chatgpt_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
