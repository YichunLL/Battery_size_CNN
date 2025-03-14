from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import joblib
import os
import requests

# Initialize FastAPI app
app = FastAPI()
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


# ✅ Function to send predictions to DeepSeek API for insights
def analyze_with_deepseek(predictions, input_data):
    deepseek_api_url = "https://api.deepseek.com/analyze"  # Replace with actual DeepSeek API
    headers = {
        "Authorization": "Bearer YOUR_DEEPSEEK_API_KEY",
        "Content-Type": "application/json"
    }
    
    payload = {
        "query": "Analyze and optimize this battery prediction.",
        "input_data": {
            "pack_dimensions": {
                "Length_pack": input_data.Length_pack,
                "Width_pack": input_data.Width_pack,
                "Height_pack": input_data.Height_pack
            },
            "electrical_properties": {
                "Energy": input_data.Energy,
                "Total_Voltage": input_data.Total_Voltage
            },
            "predicted_cell_size": predictions
        }
    }
    
    response = requests.post(deepseek_api_url, json=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json()  # Return DeepSeek's response
    else:
        return {"error": "DeepSeek API failed", "status_code": response.status_code}




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
        
# ✅ Send predictions and input data to DeepSeek for deeper analysis
        deepseek_response = analyze_with_deepseek(response, input_data)

        return {"predictions": response, "deepseek_analysis": deepseek_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
