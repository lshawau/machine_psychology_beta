# Machine Psychology Lab

A comprehensive research tool for testing AI models on psychological and cognitive tasks. Evaluate theory of mind, cognitive biases, reasoning patterns, and more with standardized psychological assessments.

## 🧠 What This Is

The Machine Psychology Lab provides a systematic framework for psychological evaluation of AI models, featuring:

- **Standardized Test Battery**: Built-in psychological tests (Sally-Anne, cognitive bias tasks, reasoning assessments)
- **Dual Evaluation System**: Machine analysis + human expert evaluation
- **Advanced NLP Analysis**: Multiple analysis engines from basic (VADER/TextBlob) to state-of-the-art (RoBERTa/Sentence-BERT)
- **Research-Grade Methodology**: Proper controls, validation, and reproducible results

## ⚠️ Technical Prerequisites

**This tool is designed for computational psychology researchers.** The installation and proper use require:

- Familiarity with Python environments and package management
- Command-line interface experience
- Understanding of AI/ML model deployment and troubleshooting
- Knowledge of psychological assessment methodology

**If you cannot complete the installation independently, we recommend developing these technical skills before attempting to evaluate AI psychological analysis.** This ensures proper interpretation of results and appropriate use of the research methodology.

## 🚀 Quick Start

### Option 1: Automated Installation (Recommended)
```bash
# Clone the repository
git clone https://github.com/lshawau/machine_psychology_beta.git
cd machine-psychology-lab

# Run automated installation script
python install.py

# Activate environment and download models
conda activate machine-psychology
python setup_models.py

# Run application
python machine_psychology_lab_1.py
```

### Option 2: Manual Installation
```bash
# Clone the repository
git clone https://github.com/lshawau/machine_psychology_beta.git
cd machine-psychology-lab

# Create conda environment with Python 3.11
conda create -n machine-psychology python=3.11
conda activate machine-psychology

# Install all packages from conda-forge (more reliable than pip)
conda install -c conda-forge spacy nltk pandas numpy scipy matplotlib scikit-learn
conda install -c conda-forge transformers sentence-transformers torch torchvision torchaudio
conda install -c conda-forge textblob vadersentiment requests pyyaml tqdm

# Download spaCy English model
python -m spacy download en_core_web_sm

# Download required models (~620MB total)
python setup_models.py

# Verify installation
python setup_models.py --verify-only

# Run application
python machine_psychology_lab_1.py
```

## 🔧 Analysis Engines

Choose your analysis approach in **Settings → Analysis Engine Configuration**:

- **Basic Engine**: VADER + TextBlob + spaCy (fast, reliable, minimal dependencies)
- **Advanced Engine**: RoBERTa + Sentence-BERT (state-of-the-art accuracy)
- **Hybrid Engine**: Combined analysis with consensus scoring (recommended for research)

## 📊 Research Applications

### Designed For:
- Computational psychology laboratories
- AI safety and alignment research
- Cognitive science studies comparing human vs. machine reasoning
- Academic research on AI psychological capabilities and limitations

### Key Features:
- **Methodological Rigor**: Controlled testing conditions with validated psychological instruments
- **Comparative Analysis**: Human expert evaluation as ground truth vs. machine estimation
- **Reproducible Results**: Standardized prompts, scoring criteria, and analysis methods
- **Publication Ready**: Detailed methodology tracking and citation information

## 📚 Documentation

- **[Setup Guide](SETUP_GUIDE.md)** - Detailed installation and troubleshooting
- **[Engine Upgrade Summary](ENGINE_UPGRADE_SUMMARY.md)** - Technical architecture overview
- **[Citations](CITATIONS.md)** - Academic references and model attributions

## 🎯 Research Philosophy

This tool expects NLP models to perform poorly compared to human psychological analysis - and that's the point. We're building a systematic way to measure and understand the current limitations of AI in psychological assessment, not to replace human expertise.

**Machine analysis provides interpretive insights only. Human evaluation is required for all clinical or research conclusions.**

## 🔬 Academic Use

When publishing research using this tool, please cite:
- The specific analysis engines used (see [CITATIONS.md](CITATIONS.md))
- This tool's methodology and approach
- The foundational research that inspired this work

## 📋 System Requirements

- **Python 3.11+**
- **Storage**: ~620MB for full model setup (Basic engine: ~15MB)
- **Memory**: 4GB+ RAM recommended for Advanced engine
- **Network**: Internet connection required for initial model download

## 🤝 Contributing

This is research software built for the computational psychology community. Contributions welcome from researchers who understand both psychological methodology and AI/ML systems.

## 📄 License

GPL-3.0 License - see LICENSE file for details.

## ⚡ Support

For technical issues:
1. Check `python setup_models.py --verify-only` output
2. Review detailed setup guide and troubleshooting section
3. Ensure all dependencies are properly installed

**Note**: This tool gracefully degrades - if Advanced engines fail, Basic engine will still provide core functionality.

---

*Built for researchers who understand that measuring AI limitations is as important as measuring AI capabilities.*
