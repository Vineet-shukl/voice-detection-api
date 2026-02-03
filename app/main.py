
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field, validator
from typing import Optional
from app.core.audio import decode_base64_audio, preprocess_audio
from app.core.model import voice_detector
from app.config import settings
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
    audio: Optional[str] = Field(None, description="Base64 encoded audio file (MP3, WAV, OGG, FLAC)")
    audioBase64: Optional[str] = Field(None, description="Alternative field name for base64 audio")
    language: Optional[str] = Field(None, description="Language of the audio (optional)")
    audioFormat: Optional[str] = Field(None, description="Format of the audio file (optional)")
    
    @validator('audio', always=True)
    def get_audio_data(cls, v, values):
        # If 'audio' is not provided, use 'audioBase64'
        if v is None and 'audioBase64' in values and values['audioBase64']:
            return values['audioBase64']
        if v is None:
            raise ValueError("Either 'audio' or 'audioBase64' field is required")
        return v

# Response Model
class DetectionResponse(BaseModel):
    result: str
    confidence: float

def verify_api_key(x_api_key: str = Header(None, alias="X-API-Key")):
    """
    Verify the API key from the request header.
    If API_KEY is not configured, this check is skipped (development mode).
    """
    # If no API key is configured, allow all requests (development mode)
    if not settings.API_KEY:
        return True
    
    # If API key is configured, verify it
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API key is required. Please provide X-API-Key header."
        )
    
    if x_api_key != settings.API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    
    return True

@app.get("/")
def read_root():
    auth_status = "enabled" if settings.API_KEY else "disabled (development mode)"
    return {
        "status": "healthy",
        "message": "AI Voice Detection API is running. Use POST /detect to analyze audio.",
        "authentication": auth_status
    }

@app.post("/detect", response_model=DetectionResponse)
async def detect_voice(
    request: AudioRequest,
    authenticated: bool = Depends(verify_api_key)
):
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
