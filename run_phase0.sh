#!/bin/bash
# Run Phase 0 with existing cached models

cd /scratch2/f004ndc/Math\ Induction\ Head

# Set HuggingFace cache to existing models
export HF_HOME="/scratch2/f004ndc/LLM Second-Order Effects/models"

source venv/bin/activate

echo "=========================================="
echo "PHASE 0: QUICK VALIDATION"
echo "=========================================="
echo ""
echo "Available models:"
ls -1 $HF_HOME/models--* | sed 's/.*models--/  - /' | sed 's/--/\//g'
echo ""
echo "Using model: meta-llama/Meta-Llama-3-8B"
echo ""

# Run Phase 0
python main.py --phase 0 --model meta-llama/Meta-Llama-3-8B

echo ""
echo "Phase 0 complete!"
