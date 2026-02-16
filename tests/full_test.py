
import requests
import base64
import json
import os

print("=" * 60)
print("AI Voice Detection API - Full Test")
print("=" * 60)

# Test 1: Server health check
print("\n[1/3] Checking server health...")
try:
    response = requests.get("http://localhost:8000/", timeout=5)
    print(f"✅ Server is running!")
    print(f"    Response: {response.json()}")
except Exception as e:
    print(f"❌ Server health check failed: {e}")
    exit(1)

# Test 2: Test with real audio file
print("\n[2/3] Testing /detect endpoint with real audio...")
audio_path = "tests/sample_silence.wav"

if not os.path.exists(audio_path):
    print(f"❌ Audio file not found: {audio_path}")
    exit(1)

try:
    # Read and encode audio
    with open(audio_path, "rb") as f:
        audio_data = f.read()
    
    b64_audio = base64.b64encode(audio_data).decode("utf-8")
    print(f"    Audio file size: {len(audio_data)} bytes")
    print(f"    Base64 size: {len(b64_audio)} characters")
    
    # Send request
    payload = {"audio": b64_audio}
    print("    Sending request to API...")
    response = requests.post("http://localhost:8000/detect", json=payload, timeout=60)
    
    print(f"    Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Detection successful!")
        print(f"    Result: {result['result']}")
        print(f"    Confidence: {result['confidence']:.4f}")
    else:
        print(f"❌ Request failed with status {response.status_code}")
        print(f"    Error: {response.text}")
        
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)
