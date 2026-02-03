"""
Test the API with authentication.
"""
import requests
import base64
import os

def test_without_api_key():
    """Test without API key - should fail if auth is enabled"""
    print("Test 1: Request WITHOUT API key")
    print("-" * 50)
    
    # Simple test payload
    test_audio = base64.b64encode(b"test").decode('utf-8')
    payload = {"audio": test_audio}
    
    try:
        response = requests.post("http://localhost:8001/detect", json=payload, timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    print()

def test_with_wrong_api_key():
    """Test with wrong API key - should fail"""
    print("Test 2: Request with WRONG API key")
    print("-" * 50)
    
    test_audio = base64.b64encode(b"test").decode('utf-8')
    payload = {"audio": test_audio}
    headers = {"X-API-Key": "wrong-key-12345"}
    
    try:
        response = requests.post("http://localhost:8001/detect", json=payload, headers=headers, timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    print()

def test_with_correct_api_key():
    """Test with correct API key - should succeed"""
    print("Test 3: Request with CORRECT API key")
    print("-" * 50)
    
    # Read API key from .env file
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("API_KEY")
    
    if not api_key:
        print("⚠️  No API_KEY found in .env file")
        return
    
    print(f"Using API Key: {api_key[:10]}...")
    
    test_audio = base64.b64encode(b"test").decode('utf-8')
    payload = {"audio": test_audio}
    headers = {"X-API-Key": api_key}
    
    try:
        response = requests.post("http://localhost:8001/detect", json=payload, headers=headers, timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    print()

def test_health_check():
    """Test health check endpoint"""
    print("Test 4: Health Check")
    print("-" * 50)
    
    try:
        response = requests.get("http://localhost:8001/", timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    print()

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("API AUTHENTICATION TESTS")
    print("=" * 50 + "\n")
    
    test_health_check()
    test_without_api_key()
    test_with_wrong_api_key()
    test_with_correct_api_key()
    
    print("=" * 50)
    print("Tests Complete!")
    print("=" * 50)
