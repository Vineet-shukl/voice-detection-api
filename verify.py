
import requests
import base64
import sys
import os

def test_api(audio_path_or_url, api_url="http://localhost:8000/detect"):
    print(f"Testing API at {api_url} with audio: {audio_path_or_url}")
    
    # Get audio data
    if audio_path_or_url.startswith("http"):
        print("Downloading audio...")
        response = requests.get(audio_path_or_url)
        audio_data = response.content
    else:
        if not os.path.exists(audio_path_or_url):
            print(f"Error: File {audio_path_or_url} not found.")
            return
        with open(audio_path_or_url, "rb") as f:
            audio_data = f.read()
            
    # Encode to Base64
    b64_audio = base64.b64encode(audio_data).decode("utf-8")
    
    # Payload
    payload = {"audio": b64_audio}
    
    # Send Request
    try:
        print("Sending request to API...")
        response = requests.post(api_url, json=payload)
        
        if response.status_code == 200:
            print("\n✅ Success!")
            print("Response:", response.json())
        else:
            print(f"\n❌ Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"\n❌ Request Failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify.py <path_to_audio_file_or_url> [api_url]")
        print("Example: python verify.py sample.mp3")
    else:
        audio_src = sys.argv[1]
        url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000/detect"
        test_api(audio_src, url)
