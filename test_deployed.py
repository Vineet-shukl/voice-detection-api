"""
Quick test script for the deployed hackathon-compliant API
"""
import requests
import json

# API Configuration
API_URL = "https://Pandaisop-voice-detection-api.hf.space/detect"
API_KEY = "uNaRqJimOAQUK4uL-YRN_DjvHwpGiV8igbhJUUVm3NkY"

# Get base64 audio from the file
try:
    with open("base64_output.txt", "r") as f:
        audio_base64 = f.read().strip()
except FileNotFoundError:
    print("❌ base64_output.txt not found!")
    print("Run: python get_base64.py first")
    exit(1)

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

print("🧪 Testing Deployed API...")
print(f"📍 URL: {API_URL}")
print(f"🔑 Using API Key: {API_KEY[:20]}...")
print(f"📊 Audio length: {len(audio_base64)} characters\n")

# Make request
try:
    print("⏳ Sending request...")
    response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
    
    print(f"\n✅ Status Code: {response.status_code}\n")
    
    # Pretty print response
    response_data = response.json()
    print("📦 Response:")
    print(json.dumps(response_data, indent=2))
    
    # Validate response format
    if response.status_code == 200:
        required_fields = ["status", "language", "classification", "confidenceScore", "explanation"]
        
        print("\n✅ Response Validation:")
        all_present = True
        for field in required_fields:
            if field in response_data:
                value = response_data[field]
                if isinstance(value, float):
                    print(f"  ✓ {field}: {value}")
                else:
                    print(f"  ✓ {field}: '{value}'")
            else:
                print(f"  ✗ Missing field: {field}")
                all_present = False
        
        if all_present:
            print("\n🎉 SUCCESS! API is fully compliant with hackathon specification!")
        else:
            print("\n⚠️  Some required fields are missing")
    else:
        print(f"\n⚠️  API returned error status: {response.status_code}")
        
except requests.exceptions.ConnectionError:
    print("\n❌ Connection Error!")
    print("Possible reasons:")
    print("  1. Hugging Face Space is still building (wait 2-3 minutes)")
    print("  2. Space URL is incorrect")
    print("  3. Network connectivity issue")
    print("\n💡 Check Space status at: https://huggingface.co/spaces/Pandaisop/voice-detection-api")
    
except requests.exceptions.Timeout:
    print("\n❌ Request Timeout!")
    print("The API took too long to respond. Try again in a moment.")
    
except Exception as e:
    print(f"\n❌ Error: {str(e)}")
