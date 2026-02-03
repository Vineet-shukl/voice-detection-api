# Hugging Face CLI - Fixed Commands

## The Issue
PowerShell treats `venv\Scripts\huggingface-cli` as a module path, not a command.

## ✅ Solution: Use These Commands Instead

### 1. Install Hugging Face CLI (if needed)
```bash
.\venv\Scripts\pip install huggingface_hub[cli]
```

### 2. Login to Hugging Face
**Option A: Using Python module** (Recommended)
```bash
.\venv\Scripts\python -m huggingface_hub.commands.huggingface_cli login
```

**Option B: Direct executable**
```bash
.\venv\Scripts\huggingface-cli.exe login
```

**Option C: Using full path**
```bash
& "X:\voice-detection-api\venv\Scripts\huggingface-cli.exe" login
```

### 3. Create Space
```bash
.\venv\Scripts\python -m huggingface_hub.commands.huggingface_cli repo create voice-detection-api --type space --space_sdk docker
```

### 4. Add Secret
```bash
.\venv\Scripts\python -m huggingface_hub.commands.huggingface_cli secrets add API_KEY uNaRqJimOAQUK4uL-YRN_DjvHwpGiV8igbhJUUVm3NkY --space YOUR_USERNAME/voice-detection-api
```

---

## Quick Reference

| Task | Command |
|------|---------|
| **Login** | `.\venv\Scripts\python -m huggingface_hub.commands.huggingface_cli login` |
| **Create Space** | Use web interface (easier) |
| **Add Secret** | Use web interface (easier) |
| **Deploy** | `git push hf main` |

---

## Recommended Approach

**Use the Web Interface for most tasks** - it's easier!

1. **Login**: Use CLI (see above)
2. **Create Space**: https://huggingface.co/new-space
3. **Add Secret**: In Space settings
4. **Deploy**: `git push hf main`

This avoids CLI issues and is more user-friendly! 🎯
