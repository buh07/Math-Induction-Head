# Model Download Guide

## Quick Start (Without HuggingFace Token)

For testing with a small, publicly available model:

```bash
source venv/bin/activate
python download_model.py --model gpt2
```

This downloads GPT-2 (~350M parameters) instantly and requires no authentication.

---

## Download Llama3-8B (With Authentication)

Since you have a working HuggingFace account, follow these steps:

### Step 1: Get Your HuggingFace Token

1. Go to https://huggingface.co/settings/tokens
2. Create a new token (if you don't have one)
3. Copy the token (looks like: `hf_xxxxxxxxxxxxxxxxxxxx`)

### Step 2: Authenticate in Terminal

**Option A: Interactive Login (Recommended)**

```bash
source venv/bin/activate
huggingface-cli login
```

Then paste your token when prompted. It will be saved for future use.

**Option B: Environment Variable (For this session only)**

```bash
source venv/bin/activate
export HF_TOKEN=hf_YOUR_TOKEN_HERE
```

Replace `hf_YOUR_TOKEN_HERE` with your actual token.

### Step 3: Download the Model

```bash
source venv/bin/activate

# Llama3-8B (recommended for this experiment)
python download_model.py --model meta-llama/Llama-3-8b-instruct

# OR Llama2-7B (also works, slightly smaller)
python download_model.py --model meta-llama/Llama-2-7b-hf

# With explicit token (if not using login)
python download_model.py --model meta-llama/Llama-3-8b-instruct --token hf_YOUR_TOKEN_HERE
```

The first run will download ~16GB (Llama3-8B) and cache it locally.

---

## Verify Download

After download completes, verify the model was cached:

```bash
python -c "from transformers import AutoModelForCausalLM; m = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3-8b-instruct'); print('✓ Model loaded successfully')"
```

---

## Common Issues & Solutions

### Issue: "Access Denied" or "gated model"

**Solution:**
1. Visit https://huggingface.co/meta-llama/Llama-3-8b-instruct
2. Click "Access this model" and accept the license
3. Wait a few minutes for it to propagate
4. Try download again

### Issue: "Invalid token"

**Solution:**
1. Double-check your token (copy from https://huggingface.co/settings/tokens)
2. Ensure no extra spaces before/after
3. Try with explicit token: `python download_model.py --model ... --token hf_YOUR_TOKEN`

### Issue: "Out of memory" during download

**Solution:**
Use GPT-2 for testing instead:
```bash
python download_model.py --model gpt2
```

Or download to external storage:
```bash
export HF_HOME=/path/to/larger/disk
python download_model.py --model meta-llama/Llama-3-8b-instruct
```

---

## Next Steps (Once Model is Downloaded)

```bash
source venv/bin/activate
python main.py --phase 0 --model meta-llama/Llama-3-8b-instruct
```

This starts Phase 0 of the experiment (quick validation, ~1-2 hours).

---

## Model Cache Location

Downloads are cached at: `~/.cache/huggingface/hub/`

To see what's cached:
```bash
ls ~/.cache/huggingface/hub/
```

To use a different cache location:
```bash
export HF_HOME=/custom/path
python download_model.py --model ...
```
