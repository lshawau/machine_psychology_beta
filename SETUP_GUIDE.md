# Machine Psychology Lab - Setup Guide

## ⚠️ Technical Prerequisites

**This tool is designed for computational psychology researchers.** Before proceeding with installation, ensure you have:

- **Python Environment Management**: Experience with conda/pip, virtual environments, and dependency resolution
- **Command-Line Proficiency**: Comfortable navigating terminals, running scripts, and interpreting error messages
- **AI/ML Model Understanding**: Familiarity with transformer models, embeddings, and NLP pipeline concepts
- **Research Methodology**: Knowledge of psychological assessment principles and statistical validation

**If you cannot complete this installation independently, we recommend developing these technical skills before attempting to evaluate AI psychological analysis.** This ensures proper interpretation of results and appropriate use of the research methodology.

---

## Quick Start

### 1. Environment Setup
```bash
# Update your conda environment with new dependencies
conda env update -f environment.yml

# Activate the environment
conda activate machine-psychology
```

### 2. Model Download
```bash
# Download all required models (internet required)
python setup_models.py

# This will download:
# - spaCy English model (~15MB)
# - RoBERTa sentiment model (~500MB) 
# - Sentence-BERT embeddings (~90MB)
```

### 3. Verification
```bash
# Verify everything is working
python setup_models.py --verify-only

# Should show all engines as ✅ working
```

### 4. Run Application
```bash
python machine_psychology_lab_1.py
```

## Analysis Engine Options

Once running, go to **Settings → Analysis Engine Configuration**:

- **✓ Basic (VADER + TextBlob + spaCy)** - Fast, reliable, minimal dependencies
- **✓ Advanced (Transformers + Sentence-BERT)** - State-of-the-art accuracy
- **✓ Hybrid (Basic + Advanced)** - Comprehensive analysis with consensus scoring

## Troubleshooting

### Models Not Downloading?
```bash
# Check internet connection
python setup_models.py --offline  # Should show what's cached

# Try with custom cache directory
python setup_models.py --cache-dir ./models
```

### Advanced Engine Not Available?
```bash
# Install missing dependencies
pip install transformers sentence-transformers torch

# Re-run setup
python setup_models.py
```

### spaCy Model Issues?
```bash
# Manual spaCy model download
python -m spacy download en_core_web_sm

# Verify
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('✅ spaCy working')"
```

## Offline Usage

After initial setup, the application works offline:

```bash
# All models cached locally
python machine_psychology_lab_1.py

# Verify offline capability
python setup_models.py --offline
```

## Storage Requirements

- **Basic Engine**: ~15MB (spaCy model only)
- **Advanced Engine**: ~605MB (RoBERTa + Sentence-BERT + spaCy)
- **Full Setup**: ~620MB total

Models are cached in:
- **Windows**: `%USERPROFILE%\.cache\huggingface`
- **macOS/Linux**: `~/.cache/huggingface`

## Research Usage

### Engine Selection Strategy

**For Speed**: Use Basic Engine
- VADER sentiment analysis
- Rule-based confidence detection
- Fast processing for large datasets

**For Accuracy**: Use Advanced Engine  
- RoBERTa transformer sentiment
- Semantic similarity reasoning
- Better handling of nuanced language

**For Research**: Use Hybrid Engine
- Comparative analysis between methods
- Consensus confidence scoring
- Method validation studies

### Citation Information

When publishing research using these engines:

**Basic Engine**:
- VADER: Hutto, C.J. & Gilbert, E.E. (2014)
- TextBlob: Loria, S. (2018)
- spaCy: Honnibal, M. & Montani, I. (2017)

**Advanced Engine**:
- RoBERTa: Barbieri, F. et al. (2022) - cardiffnlp/twitter-roberta-base-sentiment-latest
- Sentence-BERT: Reimers, N. & Gurevych, I. (2019) - all-MiniLM-L6-v2

## Support

For issues:
1. Check `python setup_models.py --verify-only` output
2. Review console output for specific error messages
3. Ensure all dependencies are installed via `environment.yml`

The application gracefully degrades - if Advanced Engine fails, Basic Engine will still work for core functionality.
