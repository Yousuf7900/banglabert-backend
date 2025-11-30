# Block 1: Imports
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Block 2: App Initialization
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows React frontend to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Block 3: Load Model

MODEL_NAME = "Yousuf-Islam/banglabert-shirk-classifier"

try:
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# Block 4: Request Schema
class TextRequest(BaseModel):
    text: str

# Block 5: Prediction Endpoint
@app.post("/predict")
def predict(request: TextRequest):
    text = request.text
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Calculate Confidence
    probs = F.softmax(logits, dim=1)
    predicted_class_id = logits.argmax().item()
    predicted_label = model.config.id2label[predicted_class_id]
    confidence = probs[0][predicted_class_id].item()
    
    return {
        "text": text,
        "label": predicted_label,
        "confidence": round(confidence, 4)
    }

# Block 6: Health Check
@app.get("/")
def home():
    return {"message": "BanglaBERT API is Running"}