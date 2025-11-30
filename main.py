import os
import requests
import time
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_MODEL_ID = "Yousuf-Islam/banglabert-shirk-classifier"
API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

# ✅ SECURE FIX: Read from Environment Variable
# We will set this in Render Dashboard later.
HF_TOKEN = os.getenv("HF_TOKEN")

headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

class TextRequest(BaseModel):
    text: str

def query_huggingface(payload, retries=5):
    for i in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            
            if response.status_code != 200:
                print(f"Attempt {i+1}: Status {response.status_code}")
            
            try:
                output = response.json()
            except json.JSONDecodeError:
                print("Received HTML instead of JSON. Retrying...")
                time.sleep(2)
                continue

            if isinstance(output, dict) and "estimated_time" in output:
                wait_time = output["estimated_time"]
                print(f"Model is loading... waiting {wait_time:.2f}s")
                time.sleep(wait_time + 1)
                continue 
            
            if isinstance(output, dict) and "error" in output:
                raise Exception(f"Hugging Face Error: {output['error']}")
                
            return output

        except requests.exceptions.ConnectionError:
            time.sleep(1)
            continue
            
    raise Exception("Model is busy. Please try again in a moment.")

@app.post("/predict")
def predict(request: TextRequest):
    try:
        output = query_huggingface({"inputs": request.text})
        
        if isinstance(output, list) and len(output) > 0 and isinstance(output[0], list):
            predictions = output[0]
        elif isinstance(output, dict) and "error" in output:
            raise Exception(output["error"])
        else:
            predictions = output

        if not isinstance(predictions, list):
             raise Exception("Unexpected response format.")

        best_prediction = max(predictions, key=lambda x: x["score"])
        
        return {
            "text": request.text,
            "label": best_prediction["label"],
            "confidence": round(best_prediction["score"], 4),
            "all_scores": predictions
        }

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"status": "online", "model": HF_MODEL_ID}