# Your Hugging Face Space URLs

## Space Management Page
This is where you see logs, settings, and status:
```
https://huggingface.co/spaces/Pandaisop/voice-detection-api
```

## Your API Endpoint (The actual API)
This is what you use for API calls:
```
https://Pandaisop-voice-detection.hf.space
```

## Important Note
If the Space shows "Running" but the API URL returns 404, try these:

### 1. Check the "App" tab
On your Space page, click the "App" tab. This should show your running API.

### 2. Look for the iframe URL
The "App" tab might show your API in an iframe. The actual URL might be slightly different.

### 3. Wait a few minutes
Sometimes there's a delay between "Running" status and the domain becoming active.

### 4. Check the logs
In the "Logs" tab, look for:
```
INFO: Uvicorn running on http://0.0.0.0:8000
✅ API Key authentication enabled
```

## Test Commands

### Health Check
```bash
curl https://Pandaisop-voice-detection.hf.space/
```

### Or in browser
```
https://Pandaisop-voice-detection.hf.space/
```

### API Documentation
```
https://Pandaisop-voice-detection.hf.space/docs
```

## If Still Getting 404

The Space page URL and API URL are different:
- **Space page**: `https://huggingface.co/spaces/Pandaisop/voice-detection-api`
- **API endpoint**: `https://Pandaisop-voice-detection.hf.space`

Make sure you're using the API endpoint URL, not the Space page URL!
