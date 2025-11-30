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

# ✅ FIXED: Reverted to the CORRECT URL for custom models
API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

# Read Token from Environment
HF_TOKEN = os.getenv("HF_TOKEN")
headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

class TextRequest(BaseModel):
    text: str

def query_huggingface(payload, retries=5):
    """
    Sends request to HF API with robust error handling for Loading (503) states.
    """
    for i in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            
            # DEBUG: Print status if it's not 200 OK
            if response.status_code != 200:
                print(f"Attempt {i+1}: Status {response.status_code} | {response.text[:100]}")
            
            # Case 1: Model is Loading (503)
            if response.status_code == 503:
                try:
                    est_time = response.json().get("estimated_time", 20.0)
                    print(f"Model is loading... waiting {est_time:.2f}s")
                    time.sleep(est_time)
                except:
                    print("Model is loading (no time estimate). Waiting 10s...")
                    time.sleep(10)
                continue # Retry loop

            # Case 2: 404 Not Found (Wrong Model Name)
            if response.status_code == 404:
                raise Exception(f"Model not found at {API_URL}. Check model name or privacy settings.")

            # Attempt to parse JSON response
            try:
                output = response.json()
            except json.JSONDecodeError:
                print("Received non-JSON response. Retrying...")
                time.sleep(2)
                continue
            
            # Case 3: Explicit API Error Message
            if isinstance(output, dict) and "error" in output:
                # Ignore "warnings" (sometimes included), catch real errors
                if isinstance(output["error"], list): 
                     # Sometimes errors are lists
                     raise Exception(f"HF Error: {output['error'][0]}")
                if "no longer supported" in str(output["error"]):
                     # Rare edge case protection
                     print("Warning: API deprecation message received. Retrying...")
                     time.sleep(1)
                     continue
                raise Exception(f"Hugging Face Error: {output['error']}")
                
            return output

        except requests.exceptions.ConnectionError:
            print("Connection error... retrying")
            time.sleep(1)
            continue
            
    raise Exception("Model is busy or unreachable after multiple attempts.")

@app.post("/predict")
def predict(request: TextRequest):
    try:
        output = query_huggingface({"inputs": request.text})
        
        # Handle format: [[{label, score}, ...]] vs [{label, score}, ...]
        if isinstance(output, list) and len(output) > 0 and isinstance(output[0], list):
            predictions = output[0]
        elif isinstance(output, dict) and "error" in output:
            raise Exception(output["error"])
        else:
            predictions = output

        if not isinstance(predictions, list):
             raise Exception(f"Unexpected format: {str(output)[:100]}")

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