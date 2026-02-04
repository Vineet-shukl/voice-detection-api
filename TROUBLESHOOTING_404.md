# Troubleshooting 404 Error

## Possible Causes

### 1. Space URL Format
The correct URL format for Hugging Face Spaces is:
```
https://huggingface.co/spaces/USERNAME/SPACE-NAME
```

For your Space:
```
https://huggingface.co/spaces/Pandaisop/voice-detection-api
```

### 2. API Endpoint URL
The actual API endpoint (once deployed) is:
```
https://USERNAME-SPACE-NAME.hf.space
```

For your API:
```
https://Pandaisop-voice-detection.hf.space
```

**Note**: Sometimes it takes a few minutes after the build completes for the domain to become active.

---

## Check Your Space Status

1. **Go to your Space page**:
   ```
   https://huggingface.co/spaces/Pandaisop/voice-detection-api
   ```

2. **Look for**:
   - Build status (Building/Running/Failed)
   - Build logs at the bottom
   - "App" tab should show your running API

3. **Check the logs** for:
   ```
   INFO: Uvicorn running on http://0.0.0.0:7860
   ```

---

## If Build is Still Running

Wait for the build to complete. You'll see:
- Progress bar
- Real-time logs
- Status will change from "Building" to "Running"

---

## If Build Completed

### Test the Space page first:
```
https://huggingface.co/spaces/Pandaisop/voice-detection-api
```

### Then try the API endpoint:
```
https://Pandaisop-voice-detection.hf.space/
```

---

## Alternative: Check with curl

```bash
curl https://Pandaisop-voice-detection.hf.space/
```

If you get a connection error, the Space might still be building.

---

## Quick Checklist

- [ ] Space build completed successfully
- [ ] Logs show "Uvicorn running"
- [ ] "App" tab is accessible
- [ ] API endpoint responds

**If all else fails**: Check the Space page for error messages in the build logs.
