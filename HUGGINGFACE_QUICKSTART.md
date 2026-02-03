# Quick Hugging Face Setup Guide

## Step 1: Install Hugging Face CLI

Run this command:
```bash
pip install huggingface_hub
```

## Step 2: Get Your Hugging Face Token

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **"New token"**
3. Give it a name (e.g., "voice-detection-api")
4. Select **"Write"** access
5. Click **"Generate token"**
6. **Copy the token** (you'll need it in the next step)

## Step 3: Login to Hugging Face

Run this command:
```bash
huggingface-cli login
```

When prompted:
1. Paste your token
2. Press Enter
3. Choose "Y" to add token as git credential

## Step 4: Create Your Space

### Option A: Via Web (Easier)
1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill in:
   - **Space name**: `voice-detection-api`
   - **SDK**: Docker
   - **Hardware**: CPU basic (free)
3. Click **"Create Space"**

### Option B: Via CLI
```bash
huggingface-cli repo create voice-detection-api --type space --space_sdk docker
```

## Step 5: Prepare Files

```bash
# Rename README files
mv README.md README_LOCAL.md
mv README_HF.md README.md
```

## Step 6: Add Git Remote

Replace `YOUR_USERNAME` with your Hugging Face username:
```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/voice-detection-api
```

## Step 7: Set API Key Secret

### Via Web:
1. Go to your Space settings
2. Click **"Repository secrets"**
3. Add secret:
   - Name: `API_KEY`
   - Value: `uNaRqJimOAQUK4uL-YRN_DjvHwpGiV8igbhJUUVm3NkY`

### Via CLI:
```bash
huggingface-cli secrets add API_KEY uNaRqJimOAQUK4uL-YRN_DjvHwpGiV8igbhJUUVm3NkY --space YOUR_USERNAME/voice-detection-api
```

## Step 8: Deploy!

```bash
git add .
git commit -m "Deploy to Hugging Face"
git push hf main
```

## That's It! 🎉

Your API will be live at:
```
https://YOUR_USERNAME-voice-detection.hf.space
```

Build time: ~10-15 minutes
