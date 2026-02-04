"""
Debug test - check what error we're getting
"""
import requests
import json
import base64

API_URL = "https://Pandaisop-voice-detection-api.hf.space/detect"
API_KEY = "uNaRqJimOAQUK4uL-YRN_DjvHwpGiV8igbhJUUVm3NkY"

# Create a small test audio
with open("tests/audio_samples/human_voice_test.wav", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode('utf-8')

payload = {
    "language": "English",
    "audioFormat": "mp3",
    "audioBase64": audio_base64[:1000]  # Use just first 1000 chars for quick test
}

headers = {
    "Content-Type": "application/json",
    "x-api-key": API_KEY
}

print("Testing with small payload...")
response = requests.post(API_URL, json=payload, headers=headers, timeout=30)

print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")
