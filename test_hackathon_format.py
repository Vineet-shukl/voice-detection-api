"""
Test script for hackathon-compliant API format
"""
import requests
import base64
import json

# API Configuration
API_URL = "http://localhost:8001/detect"
API_KEY = "uNaRqJimOAQUK4uL-YRN_DjvHwpGiV8igbhJUUVm3NkY"

# Read and encode test audio
with open("tests/audio_samples/human_voice_test.wav", "rb") as f:
    audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

# Prepare request (Hackathon Format)
payload = {
    "language": "English",
    "audioFormat": "mp3",
    "audioBase64": audio_base64
}

headers = {
    "Content-Type": "application/json",
    "x-api-key": API_KEY
}

print("Testing Hackathon-Compliant API...")
print(f"Request payload keys: {list(payload.keys())}")
print(f"Audio length: {len(audio_base64)} characters\n")

# Make request
try:
    response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response:")
    print(json.dumps(response.json(), indent=2))
    
    # Validate response format
    if response.status_code == 200:
        data = response.json()
        required_fields = ["status", "language", "classification", "confidenceScore", "explanation"]
        
        print("\n✅ Response Validation:")
        for field in required_fields:
            if field in data:
                print(f"  ✓ {field}: {data[field]}")
            else:
                print(f"  ✗ Missing field: {field}")
                
except Exception as e:
    print(f"❌ Error: {str(e)}")
