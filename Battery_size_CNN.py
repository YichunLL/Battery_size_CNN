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
openai.api_key = DEEPSEEK_API_KEY
openai.base_url = "https://api.deepseek.com"


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


# ✅ Function to analyze predictions using ChatGPT API
def analyze_with_deepseek(predictions, input_data):
    if not DEEPSEEK_API_KEY:
        return {"error": "DeepSeek API key is missing!"}

    try:
        response = openai.ChatCompletion.create(
            model="deepseek-chat",  # ✅ Ensure this is the correct model name
            messages=[
                {"role": "system", "content": "You are an AI assistant helping with battery optimization."},
                {
                    "role": "user",
                    "content": f"""
                    Analyze and optimize this battery prediction:

                    **Battery Pack Specs**:
                    - Length: {input_data.Length_pack} mm
                    - Width: {input_data.Width_pack} mm
                    - Height: {input_data.Height_pack} mm
                    - Energy: {input_data.Energy} kWh
                    - Voltage: {input_data.Total_Voltage} V

                    **Predicted Cell Size**:
                    - Length: {predictions['Length_cell']} mm
                    - Width: {predictions['Width_cell']} mm
                    - Height: {predictions['Height_cell']} mm
                    - Power Density: {predictions['Power_density']} Wh/kg

                    Suggest improvements for better performance.
                    """
                }
            ],
            max_tokens=200
        )
        
        return response["choices"][0]["message"]["content"]

    except Exception as e:
        return {"error": "ChatGPT API failed", "message": str(e)}
  



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
