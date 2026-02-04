"""
Generate base64 audio string for endpoint tester
"""
import base64
import os

# Check if test audio exists
audio_path = 'tests/audio_samples/human_voice_test.wav'

if not os.path.exists(audio_path):
    print(f"❌ Audio file not found: {audio_path}")
    print("\nRun this first:")
    print("  python tests/create_test_audio.py")
    exit(1)

# Read and encode audio
with open(audio_path, 'rb') as f:
    audio_base64 = base64.b64encode(f.read()).decode('utf-8')

print("=" * 60)
print("BASE64 AUDIO STRING (Copy everything below)")
print("=" * 60)
print()
print(audio_base64)
print()
print("=" * 60)
print(f"Length: {len(audio_base64)} characters")
print("=" * 60)
print()
print("📋 Copy the string above and paste it in the endpoint tester")
print("   Field: Audio Base64 Format")
