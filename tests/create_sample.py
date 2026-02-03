
import numpy as np
import soundfile as sf
import os

# Create a 1-second silence file (16kHz mono)
sr = 16000
duration = 1.0
audio = np.zeros(int(sr * duration))

output_dir = "tests"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

file_path = os.path.join(output_dir, "sample_silence.wav")
sf.write(file_path, audio, sr)
print(f"Created dummy audio at {file_path}")
