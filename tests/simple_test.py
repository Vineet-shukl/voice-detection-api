
import requests
import json

# Test 1: Check if server is running
print("Test 1: Checking if server is running...")
try:
    response = requests.get("http://localhost:8000/", timeout=5)
    print(f"✅ Server is running! Status: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"❌ Server check failed: {e}")
    exit(1)

# Test 2: Test the /detect endpoint with a simple payload
print("\nTest 2: Testing /detect endpoint...")
try:
    # Create a minimal base64 audio (just a few bytes for testing structure)
    import base64
    test_audio = base64.b64encode(b"test").decode()
    
    payload = {"audio": test_audio}
    response = requests.post("http://localhost:8000/detect", json=payload, timeout=30)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        print("✅ API endpoint is responding!")
    else:
        print(f"⚠️  Got response but with status {response.status_code}")
        
except Exception as e:
    print(f"❌ Endpoint test failed: {e}")
