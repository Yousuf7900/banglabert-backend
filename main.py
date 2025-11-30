import os
import json
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# We use the Official Client to handle the URL chaos automatically
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
# This handles the connection, token, and correct URL automatically
client = InferenceClient(model=HF_MODEL_ID, token=HF_TOKEN)

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: TextRequest):
    retries = 3
    for i in range(retries):
        try:
            # Send request using the official library
            output = client.post(json={"inputs": request.text})
            
            # Parse output (sometimes it returns bytes)
            if isinstance(output, bytes):
                output = json.loads(output.decode("utf-8"))
            
            # 1. Handle "List of Lists" format (common for classification)
            if isinstance(output, list) and len(output) > 0 and isinstance(output[0], list):
                predictions = output[0]
            else:
                predictions = output

            # 2. Check for Errors in response
            if isinstance(predictions, dict) and "error" in predictions:
                error_msg = predictions["error"]
                # If model is loading, wait and retry
                if "loading" in str(error_msg).lower():
                    est_time = predictions.get("estimated_time", 5.0)
                    print(f"Model loading... waiting {est_time}s")
                    time.sleep(est_time)
                    continue
                raise Exception(f"HF API Error: {error_msg}")

            # 3. Validate and Sort
            if not isinstance(predictions, list):
                raise Exception(f"Unexpected response format: {predictions}")

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
            print(f"Attempt {i+1} failed: {e}")
            if i == retries - 1:
                # This ensures the Frontend sees the real error
                raise HTTPException(status_code=500, detail=str(e))
            time.sleep(2)

@app.get("/")
def home():
    return {"status": "online", "model": HF_MODEL_ID, "method": "Official Client"}