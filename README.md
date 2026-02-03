---
title: AI Voice Detection API
emoji: 🎤
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
---

# AI Voice Detection API

This Space provides an API to detect AI-generated voice samples across multiple languages (Tamil, English, Hindi, Malayalam, Telugu).

## API Endpoint

**POST** `/detect`

### Request
```json
{
  "audio": "BASE64_ENCODED_AUDIO"
}
```

### Headers
```
X-API-Key: uNaRqJimOAQUK4uL-YRN_DjvHwpGiV8igbhJUUVm3NkY
```

### Response
```json
{
  "result": "AI_GENERATED",
  "confidence": 0.8542,
  "message": "Audio classified successfully"
}
```

## Usage

```python
import requests
import base64

# Read audio file
with open('audio.mp3', 'rb') as f:
    audio_base64 = base64.b64encode(f.read()).decode('utf-8')

# Make request
headers = {"X-API-Key": "uNaRqJimOAQUK4uL-YRN_DjvHwpGiV8igbhJUUVm3NkY"}
response = requests.post(
    "https://YOUR-USERNAME-voice-detection.hf.space/detect",
    json={"audio": audio_base64},
    headers=headers
)

print(response.json())
```

## Model

Uses `mo-thecreator/Deepfake-audio-detection` (Wav2Vec2-based) for voice classification.
