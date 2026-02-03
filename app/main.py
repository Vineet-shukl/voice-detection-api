
from fastapi import FastAPI, HTTPException, Header, UploadFile, File
from pydantic import BaseModel, Field
from app.core.audio import decode_base64_audio, preprocess_audio
from app.core.model import voice_detector
import uvicorn
import logging

# Setup Logger
logger = logging.getLogger("api")

app = FastAPI(
    title="AI Voice Detection API",
    description="API to detect whether a voice sample is AI-generated or Human.",
    version="1.0.0"
)

# Request Model
class AudioRequest(BaseModel):
    audio: str = Field(..., description="Base64 encoded audio string")

# Response Model
class DetectionResponse(BaseModel):
    result: str
    confidence: float

@app.get("/")
def read_root():
    return {"message": "AI Voice Detection API is running. use POST /detect to analyze audio."}

@app.post("/detect", response_model=DetectionResponse)
async def detect_voice(request: AudioRequest):
    """
    Analyzes the provided Base64 audio and determines if it is AI-generated or Human.
    """
    try:
        # 1. Decode Base64
        audio_file = decode_base64_audio(request.audio)
        
        # 2. Preprocess Audio
        audio_tensor = preprocess_audio(audio_file)
        
        # 3. Predict
        label, confidence = voice_detector.predict(audio_tensor)
        
        return DetectionResponse(result=label, confidence=confidence)
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Internal Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error processing the request.")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
