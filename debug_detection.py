"""Debug script to understand induction head detection failure."""

import torch
import logging
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

import src.utils as utils
import src.validation_suite as phase0

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Setup model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModelForCausalLM.from_pretrained(
    'gpt2-medium',
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map='auto' if torch.cuda.is_available() else None,
    attn_implementation='eager',
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained('gpt2-medium', trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()

# Test with DIFFERENT problem types
print("=" * 60)
print("Testing Different Problem Types")
print("=" * 60)

# Type 1: Simple arithmetic (current)
arithmetic_problems = [
    {"problem": "45 + 23 ="},
    {"problem": "100 + 50 ="},
    {"problem": "22 + 18 ="},
]

# Type 2: Sequence copying (better for induction heads)
sequence_problems = [
    {"problem": "The pattern is: 1 2 3 1 2 3 1 2 3 what comes next? Answer: 1"},
    {"problem": "A B C A B C A B C, the next is: A"},
    {"problem": "Red green blue red green blue red green blue, the next is: red"},
]

# Type 3: Repeated tokens
repeated_problems = [
    {"problem": "apple basket apple basket apple basket apple, the next is: basket"},
    {"problem": "x y x y x y x, the next is: y"},
    {"problem": "1 2 1 2 1 2 1, the next should be: 2"},
]

print("\n1. Testing Arithmetic Problems:")
for prob in arithmetic_problems:
    text = prob['problem']
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt')
        tokens = inputs['input_ids'][0].tolist()
        token_texts = [tokenizer.decode([t]) for t in tokens]
        print(f"  Text: {text}")
        print(f"  Tokens: {tokens}")
        print(f"  Token texts: {token_texts}")

print("\n2. Testing Sequence Problems:")
for prob in sequence_problems:
    text = prob['problem']
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt')
        tokens = inputs['input_ids'][0].tolist()
        token_texts = [tokenizer.decode([t]) for t in tokens]
        print(f"  Text: {text[:50]}...")
        print(f"  Num tokens: {len(tokens)}")

print("\n3. Testing Repeated Token Problems:")
for prob in repeated_problems:
    text = prob['problem']
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt')
        tokens = inputs['input_ids'][0].tolist()
        token_texts = [tokenizer.decode([t]) for t in tokens]
        print(f"  Text: {text}")
        print(f"  Tokens: {tokens}")
        
        # Count repeated tokens
        repeated_pairs = 0
        for i in range(len(tokens)):
            for j in range(i):
                if tokens[i] == tokens[j]:
                    repeated_pairs += 1
        print(f"  Repeated token pairs: {repeated_pairs}")

# Now test detection on repeated problems
print("\n" + "=" * 60)
print("Testing Induction Head Detection")
print("=" * 60)

detector = phase0.InductionHeadDetector(model, tokenizer)

print("\nDetecting on repeated problems:")
candidates = detector.detect_heads_quick(repeated_problems, threshold=-1.0)  # Lower threshold to see all scores
print(f"Found {len(candidates)} candidate heads")
if candidates:
    for head in candidates[:5]:
        print(f"  Layer {head['layer']}, Head {head['head']}: entropy={head['entropy_score']:.4f}, "
              f"repeated_focus={head['repeated_focus_score']:.4f}, combined={head['combined_score']:.4f}")
