"""
Test your deployed Hugging Face API
"""
import requests
import base64
import sys

def test_hf_api(username, audio_file="tests/audio_samples/human_voice_test.wav"):
    """Test the Hugging Face deployed API"""
    
    # Construct API URL
    api_url = f"https://{username}-voice-detection.hf.space/detect"
    api_key = "uNaRqJimOAQUK4uL-YRN_DjvHwpGiV8igbhJUUVm3NkY"
    
    print("=" * 60)
    print("TESTING HUGGING FACE DEPLOYMENT")
    print("=" * 60)
    print(f"API URL: {api_url}")
    print()
    
    # Test 1: Health Check
    print("Test 1: Health Check")
    print("-" * 60)
    try:
        health_url = f"https://{username}-voice-detection.hf.space/"
        response = requests.get(health_url, timeout=30)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        print("✅ Health check passed!")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    print()
    
    # Test 2: Voice Detection
    print("Test 2: Voice Detection with Audio")
    print("-" * 60)
    
    try:
        # Read and encode audio
        print(f"Reading audio file: {audio_file}")
        with open(audio_file, 'rb') as f:
            audio_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        print(f"Audio encoded: {len(audio_base64)} characters")
        
        # Make request
        headers = {"X-API-Key": api_key}
        payload = {"audio": audio_base64}
        
        print("Sending request...")
        response = requests.post(api_url, json=payload, headers=headers, timeout=60)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ SUCCESS!")
            print(f"Result: {result['result']}")
            print(f"Confidence: {result['confidence']:.4f}")
        else:
            print(f"\n❌ ERROR")
            print(f"Response: {response.text}")
            
    except FileNotFoundError:
        print(f"❌ Audio file not found: {audio_file}")
        print("Run: python tests/create_test_audio.py")
    except Exception as e:
        print(f"❌ Request failed: {e}")
    
    print()
    print("=" * 60)
    print("Testing complete!")
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_hf_deployment.py YOUR_HF_USERNAME")
        print("\nExample:")
        print("  python test_hf_deployment.py john-doe")
        sys.exit(1)
    
    username = sys.argv[1]
    test_hf_api(username)
