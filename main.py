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

# ✅ FIXED: The EXACT new URL structure required by Hugging Face
# Previous error 404/410 was because we missed the '/hf-inference' part
API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL_ID}"

# Token from Environment
HF_TOKEN = os.getenv("HF_TOKEN")
headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

class TextRequest(BaseModel):
    text: str

def query_huggingface(payload, retries=5):
    """
    Sends request to HF API with the NEW URL structure.
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

            # Attempt to parse JSON response
            try:
                output = response.json()
            except json.JSONDecodeError:
                print("Received non-JSON response. Retrying...")
                time.sleep(2)
                continue
            
            # Case 2: Explicit API Error Message
            if isinstance(output, dict) and "error" in output:
                # If error is a list (rare but happens)
                if isinstance(output["error"], list): 
                     raise Exception(f"HF Error: {output['error'][0]}")
                
                # Check for the specific "no longer supported" error again to be safe
                if "no longer supported" in str(output["error"]):
                     print("CRITICAL: URL mismatch. Please check API_URL config.")
                
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
    return {"status": "online", "model": HF_MODEL_ID, "url_version": "router/hf-inference"}