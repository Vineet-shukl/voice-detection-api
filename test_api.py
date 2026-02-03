"""
Test the Voice Detection API with audio files.
Supports both local files and URLs.
"""
import requests
import base64
import sys
import os

def encode_audio_file(file_path):
    """Encode audio file to base64."""
    with open(file_path, 'rb') as f:
        audio_bytes = f.read()
    return base64.b64encode(audio_bytes).decode('utf-8')

def test_api(audio_path, api_url="http://localhost:8001/detect"):
    """Test the API with an audio file."""
    print(f"Testing API at {api_url}")
    print(f"Audio file: {audio_path}\n")
    
    # Encode audio
    print("Encoding audio to base64...")
    base64_audio = encode_audio_file(audio_path)
    print(f"Base64 length: {len(base64_audio)} characters\n")
    
    # Send request
    print("Sending request to API...")
    payload = {"audio": base64_audio}
    
    try:
        response = requests.post(api_url, json=payload, timeout=60)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n" + "="*50)
            print("✅ SUCCESS!")
            print("="*50)
            print(f"Result: {result['result']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Message: {result.get('message', 'N/A')}")
            print("="*50)
        else:
            print("\n" + "="*50)
            print("❌ ERROR")
            print("="*50)
            print(f"Response: {response.text}")
            print("="*50)
            
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed: {e}")
        return False
    
    return response.status_code == 200

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <audio_file_path> [api_url]")
        print("\nExamples:")
        print("  python test_api.py tests/audio_samples/human_voice_test.wav")
        print("  python test_api.py tests/audio_samples/ai_voice_test.wav")
        print("  python test_api.py my_audio.mp3 http://your-deployed-api.com/detect")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    api_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8001/detect"
    
    if not os.path.exists(audio_path):
        print(f"❌ Error: File not found: {audio_path}")
        sys.exit(1)
    
    success = test_api(audio_path, api_url)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
