# 🎙️ VoiceGuard — AI Voice Detection API

**State-of-the-Art Deepfake Audio Detection System**

VoiceGuard is a high-performance API designed to detect AI-generated speech with exceptional accuracy. It combines advanced neural network inference with traditional forensic audio analysis to provide a robust defense against deepfake audio.

## 🚀 Key Features

*   **Multi-Stage Detection Pipeline**: Fuses deep learning with signal processing forensics.
*   **Explainable AI**: Provides detailed, human-readable explanations for every detection.
*   **Dual Analysis Engine**:
    *   **Neural Model**: Wav2Vec2-based classifier with attentive pooling.
    *   **Forensic Analyzers**: Spectral, Temporal, Formant, and Artifact detection.
*   **Real-time Base64 Processing**: Optimized for low-latency API integration.
*   **Audio Quality Profiling**: Automatically assesses SNR, clipping, and silence ratios.

## 🌐 Live Demo
Experience the API instantly on Hugging Face Spaces:
**[👉 Try VoiceGuard Demo](https://huggingface.co/spaces/Pandaisop/voice-detection-api)**

---

## 🛠️ Installation & Setup

### Prerequisites
*   Python 3.9+
*   RAM: 4GB+ (8GB recommended for optimal performance)

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd voice-detection-api
```

### 2. Install Dependencies
```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Configure Model (Optional)
By default, the API downloads a pre-trained model. To use your local trained model:
1.  Ensure your model files are in the `model/` directory.
2.  Update `.env` file:
    ```bash
    MODEL_NAME=./model
    ```

### 4. Run the API
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
The API will be available at `http://localhost:8000`.

---

## 📖 Usage

### API Endpoint: `/detect`
**Method**: `POST`
**Content-Type**: `application/json`

#### Request Body
```json
{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "<base64_encoded_audio_string>"
}
```

#### Response Example
```json
{
  "status": "success",
  "language": "English",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.98,
  "explanation": "Strong indicators of AI-generated speech detected. Evidence: unnaturally uniform spectral texture, and metronomic pause timing. Neural model and forensic analyzers are in agreement.",
  "analyzersAgree": true,
  "inferenceTimeMs": 450.2
}
```

### Web UI
Navigate to `http://localhost:8000` in your browser to access the built-in testing console. You can upload audio files directly to test the detection engine.

---

## 🧠 Model Architecture & Approach

VoiceGuard uses a **Hybrid Detection Architecture** to maximize robustness.

### 1. Neural Analysis Engine
*   **Backbone**: Fine-tuned **Wav2Vec 2.0** (XLSR-53) for extracting high-level speech representations.
*   **Classification Head**: **Attentive Statistics Pooling** layer that learns to weigh important frames, followed by a dense MLP classifier.
*   **Strategy**: analyzing multiple overlapping segments of the audio to catch partial deepfakes.

### 2. Forensic Analysis Engine
A suite of signal processing algorithms detects artifacts that neural models might miss:
*   **Spectral Analysis**: Detects unnatural smoothness in the frequency domain (typical of vocoders).
*   **Temporal Analysis**: Identifies robotic cadence and lack of natural micro-jitter in energy.
*   **Formant Analysis**: Checks for realistic formant transitions and vocal tract consistency.
*   **Artifact Detection**: Scans for phase discontinuities, digital silence, and synthesis clicks.

### 3. Decision Fusion
The **Fusion Engine** combines the probabilistic output of the Neural Model with the weighted findings of the Forensic Analyzers.
*   **Agreement Check**: If both engines agree, confidence is boosted.
*   **Disagreement Handling**: If engines disagree, the system lowers confidence and flags the result for manual review in the explanation.

---

## 🧪 Development

### Running Tests
```bash
pytest
```

### Project Structure
*   `app/main.py`: FastAPI entry point and route definitions.
*   `app/core/model.py`: Neural model inference logic.
*   `app/core/forensics.py`: Signal processing and forensic analyzers.
*   `app/core/explanation.py`: Logic for generating human-readable explanations.
*   `trainer/`: Scripts used for training and evaluating the model.

---

## 📄 License
MIT License. See `LICENSE` for more information.
