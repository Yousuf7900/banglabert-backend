import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient, InferenceTimeoutError

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
HF_MODEL_ID = "Yousuf-Islam/banglabert-shirk-classifier"
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize Client
client = InferenceClient(model=HF_MODEL_ID, token=HF_TOKEN)

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: TextRequest):
    retries = 3
    for i in range(retries):
        try:
            # ✅ FIX: Use the specific text_classification method
            # This automatically handles URLs and parsing
            result = client.text_classification(request.text)
            
            # The library returns a clean list of objects: [{'label': 'A', 'score': 0.9}, ...]
            # We convert it to a standard list of dicts for JSON serialization
            predictions = [{"label": item.label, "score": item.score} for item in result]

            # Find best prediction
            best_prediction = max(predictions, key=lambda x: x["score"])

            return {
                "text": request.text,
                "label": best_prediction["label"],
                "confidence": round(best_prediction["score"], 4),
                "all_scores": predictions
            }

        except InferenceTimeoutError:
            print("Request timed out. Retrying...")
            time.sleep(2)
            continue
            
        except Exception as e:
            # Handle "Model Loading" errors (503) specifically
            if "503" in str(e) or "loading" in str(e).lower():
                print(f"Model is loading... (Attempt {i+1})")
                time.sleep(20) # Wait longer for loading
                continue
                
            print(f"Attempt {i+1} failed: {e}")
            if i == retries - 1:
                raise HTTPException(status_code=500, detail=str(e))
            time.sleep(1)

@app.get("/")
def home():
    return {"status": "online", "model": HF_MODEL_ID, "method": "InferenceClient.text_classification"}