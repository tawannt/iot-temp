import os
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List
import pyrebase
import pandas as pd
from joblib import load

# Load biến môi trường
load_dotenv()

# === Firebase Client Initialization ===
firebase_config = {
    "apiKey": "AIzaSyCxSSc0-cHA2JS5gTXMLXIYx5Lxvxg-rOI",
    "authDomain": "catcare-iot.firebaseapp.com",
    "databaseURL": "https://catcare-iot-default-rtdb.firebaseio.com",
    "projectId": "catcare-iot",
    "storageBucket": "catcare-iot.firebasestorage.app",
    "messagingSenderId": "93953920260",
    "appId": "1:93953920260:web:77f2181d5621e7fcff5451",
    "measurementId": "G-8J6N6BELLM"
}

firebase = pyrebase.initialize_app(firebase_config)
db = firebase.database()

# === Load ML model ===
model = load('model.joblib')

# === FastAPI App ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    response: str

# HuggingFace API
API_URL = "https://router.huggingface.co/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
}

# === Fetch data from Firebase ===
def fetch_sensor_data():
    sensors = db.child("sensors").get()
    if not sensors.val():
        return {}

    data = sensors.val()
    return {
        "temperature": data.get("temperature"),
        "humidity": data.get("humidity"),
        "food_percentage": data.get("food_percentage"),
        "water_percentage": data.get("water_percentage"),
        "air_quality_voltage": data.get("air_quality_voltage"),
        "last_updated": data.get("last_updated"),
        "esp32_connected": data.get("esp32_connected"),
        "esp32_last_seen": data.get("esp32_last_seen")
    }

def is_feeding_prediction_query(text: str) -> bool:
    text = text.lower()
    # Các từ khóa liên quan
    keywords = [
        "dự đoán số lần cho ăn",
        "dự đoán cho ăn",
        "nên cho ăn",
        "mấy lần cho ăn",
        "bao nhiêu lần cho ăn",
        "số lần cho ăn",
        "nên cho mèo ăn",
        "bao nhiêu lần trong ngày"
    ]
    return any(kw in text for kw in keywords)

# === Predict feeding times ===
def predict_feeding_times(temp, hum, airq) -> int:
    X_new = pd.DataFrame([[temp, hum, airq]], columns=['Temperature', 'Humidity', 'Air_Quality'])
    result = model.predict(X_new)
    return int(result[0])

# === Build prompt ===
def build_prompt(data, user_query):
    if not data:
        context = "No sensor data available."
    else:
        context = (
            "You are a helpful IoT cat care assistant.\n"
            f"Temperature: {data.get('temperature')}°C, "
            f"Humidity: {data.get('humidity')}%, "
            f"Food Level: {data.get('food_percentage')}%, "
            f"Water Level: {data.get('water_percentage')}%, "
            f"Air Quality Voltage: {data.get('air_quality_voltage')}V, "
            f"Last Updated: {data.get('last_updated')}, "
            f"ESP32 Connected: {data.get('esp32_connected')}, "
            f"Last Seen: {data.get('esp32_last_seen')}\n"
        )
    return context + f"\nUser: {user_query}\nAssistant:"

# === Query HuggingFace ===
def query_huggingface(message):
    payload = {
        "messages": [{"role": "user", "content": message}],
        "model": "openai/gpt-oss-120b:cerebras"
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error communicating with AI: {str(e)}"

# === Chat endpoint ===
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    user_message = request.messages[-1].content
    data = fetch_sensor_data()

    # Nếu người dùng hỏi về "dự đoán số lần cho ăn"
    if is_feeding_prediction_query(user_message):
        if not data:
            return {"response": "Không có dữ liệu cảm biến để dự đoán."}
        temp = data.get("temperature")
        hum = data.get("humidity")
        airq = data.get("air_quality_voltage")
        if None in (temp, hum, airq):
            return {"response": "Thiếu dữ liệu cảm biến để dự đoán."}
        predicted_times = predict_feeding_times(temp, hum, airq)
        return {"response": f"Với nhiệt độ: {temp} độ C, độ ẩm: {hum}%, chất lượng không khí:{airq} Volt. \
                \nDự đoán số lần cho ăn hôm nay là {predicted_times} lần. Theo mô hình Linear Regression của Nhóm phát triển xây dựng."}

    # Nếu không phải câu hỏi dự đoán => hỏi AI
    prompt = build_prompt(data, user_message)
    response_text = query_huggingface(prompt)
    return {"response": response_text}