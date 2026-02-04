"""
Quick checker to see if Hugging Face deployment is ready
"""
import requests
import time

SPACE_URL = "https://Pandaisop-voice-detection-api.hf.space/"

print("🔍 Checking if Hugging Face Space is ready...\n")

try:
    response = requests.get(SPACE_URL, timeout=10)
    
    if response.status_code == 200:
        print("✅ SUCCESS! Your Space is RUNNING!")
        print(f"\n📦 Response from API:")
        print(response.json())
        print("\n🎉 You can now test your API!")
        print(f"🔗 Space URL: https://huggingface.co/spaces/Pandaisop/voice-detection-api")
    else:
        print(f"⚠️  Space responded with status code: {response.status_code}")
        print("It might still be building...")
        
except requests.exceptions.ConnectionError:
    print("❌ Connection failed!")
    print("The Space is likely still building or starting up.")
    print("\n💡 Wait 1-2 more minutes and try again.")
    
except requests.exceptions.Timeout:
    print("⏱️  Request timed out!")
    print("The Space might be starting up. Try again in a moment.")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")

print(f"\n🌐 Check manually at: https://huggingface.co/spaces/Pandaisop/voice-detection-api")
