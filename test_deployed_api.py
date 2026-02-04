"""
Quick test script for the deployed Hugging Face API
"""
import requests
import base64

# Read test audio
with open('tests/audio_samples/human_voice_test.wav', 'rb') as f:
    audio_base64 = base64.b64encode(f.read()).decode('utf-8')

# API details
API_URL = "https://Pandaisop-voice-detection-api.hf.space/detect"
API_KEY = "uNaRqJimOAQUK4uL-YRN_DjvHwpGiV8igbhJUUVm3NkY"

# Make request
headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

payload = {
    "audio": audio_base64
}

print("Testing deployed API...")
print(f"URL: {API_URL}")
print(f"Audio length: {len(audio_base64)} characters")
print("\nSending request...")

try:
    response = requests.post(API_URL, json=payload, headers=headers)
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    if response.status_code == 200:
        print("\n✅ SUCCESS! Your API is working perfectly!")
    else:
        print(f"\n❌ Error: {response.status_code}")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
