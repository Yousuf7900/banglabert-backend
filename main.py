import os
import requests
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 1. Initialize the App
app = FastAPI()

# 2. CORS Setup (Required for Frontend later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Hugging Face Configuration
# Your exact model URL
HF_MODEL_ID = "Yousuf-Islam/banglabert-shirk-classifier"
API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

# Optional: Add your HF Token if the model is private or if you hit rate limits
# HF_TOKEN = "hf_xxxxxxxxxxxxxxxx" 
# headers = {"Authorization": f"Bearer {HF_TOKEN}"}
headers = {} # Leave empty for public models

# 4. Input Schema
class TextRequest(BaseModel):
    text: str

# 5. Prediction Logic
def query_huggingface(payload, retries=3):
    """
    Sends request to HF API. Handles 'Model Loading' errors automatically.
    """
    for i in range(retries):
        response = requests.post(API_URL, headers=headers, json=payload)
        output = response.json()

        # Case 1: Model is loading (common on free tier)
        if isinstance(output, dict) and "estimated_time" in output:
            wait_time = output["estimated_time"]
            print(f"Model is loading... waiting {wait_time:.2f}s")
            time.sleep(wait_time + 1)
            continue # Try again
        
        # Case 2: Actual Error
        if isinstance(output, dict) and "error" in output:
            raise Exception(f"Hugging Face Error: {output['error']}")
            
        return output
    
    raise Exception("Model took too long to load.")

# 6. API Endpoint
@app.post("/predict")
def predict(request: TextRequest):
    try:
        # Send to Hugging Face
        output = query_huggingface({"inputs": request.text})
        
        # The API returns a list of lists: [[{'label': 'shirk', 'score': 0.9}, ...]]
        # We flatten this to get the inner list
        if isinstance(output, list) and len(output) > 0 and isinstance(output[0], list):
            predictions = output[0]
        else:
            # Sometimes format varies slightly based on pipeline
            predictions = output

        # Sort by score (highest confidence first)
        # Note: HF might return 'LABEL_0', 'LABEL_1' if config isn't perfect, 
        # but since you uploaded config.json, it should return 'shirk', 'neutral' etc.
        best_prediction = max(predictions, key=lambda x: x["score"])
        
        return {
            "text": request.text,
            "label": best_prediction["label"],
            "confidence": round(best_prediction["score"], 4),
            "all_scores": predictions
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 7. Health Check
@app.get("/")
def home():
    return {"status": "online", "model": HF_MODEL_ID}