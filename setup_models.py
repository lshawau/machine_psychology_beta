#!/usr/bin/env python3
"""
Machine Psychology Lab - Model Setup Script
==========================================

Downloads and verifies all required models for the analysis engines.
Handles spaCy models, transformer models, and sentence transformers.

Usage:
    python setup_models.py [--offline] [--cache-dir PATH] [--verify-only]

Author: Built with Claude Code
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

# Import with fallbacks for graceful error handling
try:
    import spacy
    from spacy.cli.download import download as spacy_download
    SPACY_AVAILABLE = True
except ImportError:
    spacy = None
    spacy_download = None
    SPACY_AVAILABLE = False
    print("⚠️  spaCy not available - will skip spaCy model verification")

try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    from sentence_transformers import SentenceTransformer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pipeline = None
    AutoTokenizer = None
    AutoModel = None
    SentenceTransformer = None
    torch = None
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers/PyTorch not available - will skip transformer model downloads")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    REQUESTS_AVAILABLE = False
    print("⚠️  Requests not available - network checks disabled")

class ModelSetup:
    """Handles downloading and verification of all required models."""
    
    def __init__(self, cache_dir: Optional[str] = None, offline: bool = False):
        self.cache_dir = cache_dir
        self.offline = offline
        self.models_status = {}
        
        # Model definitions
        self.spacy_models = {
            'en_core_web_sm': {
                'name': 'English Core Web (Small)',
                'size': '~15MB',
                'required_for': 'BasicAnalysisEngine, AdvancedAnalysisEngine'
            }
        }
        
        self.transformer_models = {
            'cardiffnlp/twitter-roberta-base-sentiment-latest': {
                'name': 'RoBERTa Sentiment Analysis',
                'size': '~500MB',
                'required_for': 'AdvancedAnalysisEngine',
                'type': 'sentiment-analysis'
            }
        }
        
        self.sentence_transformer_models = {
            'all-MiniLM-L6-v2': {
                'name': 'Sentence-BERT Embeddings',
                'size': '~90MB', 
                'required_for': 'AdvancedAnalysisEngine',
                'type': 'sentence-transformer'
            }
        }
    
    def print_header(self):
        """Print setup script header."""
        print("=" * 60)
        print("🧠 MACHINE PSYCHOLOGY LAB - MODEL SETUP")
        print("=" * 60)
        print()
        print("This script will download and verify models for:")
        print("• BasicAnalysisEngine (spaCy + VADER + TextBlob)")
        print("• AdvancedAnalysisEngine (RoBERTa + Sentence-BERT)")
        print("• HybridAnalysisEngine (combines both)")
        print()
    
    def check_internet_connection(self) -> bool:
        """Check if internet connection is available."""
        if not REQUESTS_AVAILABLE or self.offline or requests is None:
            return False
            
        try:
            response = requests.get("https://huggingface.co", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def setup_spacy_models(self) -> bool:
        """Download and verify spaCy models."""
        print("📚 Setting up spaCy models...")
        
        if not SPACY_AVAILABLE or spacy is None:
            print("❌ spaCy not available - please install with: pip install spacy")
            return False
        
        success = True
        for model_name, info in self.spacy_models.items():
            print(f"\n🔍 Checking {info['name']} ({model_name})...")
            print(f"   Size: {info['size']}")
            print(f"   Required for: {info['required_for']}")
            
            try:
                # Try to load the model
                nlp = spacy.load(model_name)
                print(f"✅ {model_name} already installed and working")
                self.models_status[model_name] = "available"
                
                # Quick test
                doc = nlp("This is a test sentence.")
                if len(doc) > 0:
                    print(f"   ✓ Model test passed ({len(doc)} tokens processed)")
                
            except OSError:
                if self.offline:
                    print(f"❌ {model_name} not found (offline mode)")
                    success = False
                    self.models_status[model_name] = "missing"
                    continue
                
                print(f"📥 Downloading {model_name}...")
                try:
                    if spacy_download is not None:
                        spacy_download(model_name)
                        print(f"✅ {model_name} downloaded successfully")
                        self.models_status[model_name] = "downloaded"
                    else:
                        print(f"❌ spaCy download function not available")
                        success = False
                        self.models_status[model_name] = "failed"
                except Exception as e:
                    print(f"❌ Failed to download {model_name}: {e}")
                    success = False
                    self.models_status[model_name] = "failed"
            
            except Exception as e:
                print(f"❌ Error with {model_name}: {e}")
                success = False
                self.models_status[model_name] = "error"
        
        return success
    
    def setup_transformer_models(self) -> bool:
        """Download and verify transformer models."""
        print("\n🤖 Setting up Transformer models...")
        
        if not TRANSFORMERS_AVAILABLE or AutoTokenizer is None or AutoModel is None or pipeline is None:
            print("❌ Transformers not available - please install with: pip install transformers torch")
            return False
        
        success = True
        for model_name, info in self.transformer_models.items():
            print(f"\n🔍 Checking {info['name']} ({model_name})...")
            print(f"   Size: {info['size']}")
            print(f"   Required for: {info['required_for']}")
            
            try:
                if self.offline:
                    # Try to load from cache only
                    print("   🔄 Checking local cache (offline mode)...")
                    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                    model = AutoModel.from_pretrained(model_name, local_files_only=True)
                    print(f"✅ {model_name} found in cache")
                else:
                    print("   📥 Downloading/verifying model...")
                    # This will download if not cached, or load from cache if available
                    sentiment_pipeline = pipeline(
                        "sentiment-analysis",
                        model=model_name,
                        return_all_scores=True
                    )
                    
                    # Quick test
                    test_result = sentiment_pipeline("This is a test sentence.")
                    if test_result and isinstance(test_result, list) and len(test_result) > 0:
                        first_result = test_result[0]
                        if isinstance(first_result, list) and len(first_result) > 0:
                            print(f"✅ {model_name} working correctly")
                            print(f"   ✓ Test result: {len(first_result)} sentiment scores")
                    
                self.models_status[model_name] = "available"
                
            except Exception as e:
                print(f"❌ Error with {model_name}: {e}")
                if "offline" in str(e).lower() or "local_files_only" in str(e):
                    print("   💡 Model not in cache - run without --offline to download")
                success = False
                self.models_status[model_name] = "failed"
        
        return success
    
    def setup_sentence_transformer_models(self) -> bool:
        """Download and verify sentence transformer models."""
        print("\n🔤 Setting up Sentence Transformer models...")
        
        if not TRANSFORMERS_AVAILABLE or SentenceTransformer is None:
            print("❌ Sentence Transformers not available - please install with: pip install sentence-transformers")
            return False
        
        success = True
        for model_name, info in self.sentence_transformer_models.items():
            print(f"\n🔍 Checking {info['name']} ({model_name})...")
            print(f"   Size: {info['size']}")
            print(f"   Required for: {info['required_for']}")
            
            try:
                if self.offline:
                    print("   🔄 Checking local cache (offline mode)...")
                    # Try to load from cache only
                    model = SentenceTransformer(model_name, cache_folder=self.cache_dir)
                    if hasattr(model, '_modules'):
                        print(f"✅ {model_name} found in cache")
                    else:
                        raise Exception("Model not properly loaded from cache")
                else:
                    print("   📥 Downloading/verifying model...")
                    model = SentenceTransformer(model_name, cache_folder=self.cache_dir)
                    
                    # Quick test
                    test_embedding = model.encode(["This is a test sentence."])
                    if test_embedding is not None and len(test_embedding) > 0:
                        print(f"✅ {model_name} working correctly")
                        print(f"   ✓ Test embedding shape: {test_embedding.shape}")
                
                self.models_status[model_name] = "available"
                
            except Exception as e:
                print(f"❌ Error with {model_name}: {e}")
                if "offline" in str(e).lower() or "cache" in str(e).lower():
                    print("   💡 Model not in cache - run without --offline to download")
                success = False
                self.models_status[model_name] = "failed"
        
        return success
    
    def verify_analysis_engines(self) -> bool:
        """Test that analysis engines can be initialized."""
        print("\n🔧 Verifying Analysis Engines...")
        
        # Import the analysis engines from the main application
        try:
            sys.path.append('.')
            from machine_psychology_lab_1 import BasicAnalysisEngine, AdvancedAnalysisEngine, HybridAnalysisEngine
            
            engines_status = {}
            
            # Test BasicAnalysisEngine
            print("\n🔍 Testing BasicAnalysisEngine...")
            try:
                basic_engine = BasicAnalysisEngine()
                if basic_engine.available:
                    # Quick test
                    result = basic_engine.analyze_sentiment("This is a positive test sentence.")
                    if result and 'confidence_level' in result:
                        print("✅ BasicAnalysisEngine working correctly")
                        engines_status['basic'] = "working"
                    else:
                        print("⚠️  BasicAnalysisEngine loaded but test failed")
                        engines_status['basic'] = "partial"
                else:
                    print("❌ BasicAnalysisEngine not available")
                    engines_status['basic'] = "unavailable"
            except Exception as e:
                print(f"❌ BasicAnalysisEngine error: {e}")
                engines_status['basic'] = "error"
            
            # Test AdvancedAnalysisEngine
            print("\n🔍 Testing AdvancedAnalysisEngine...")
            try:
                advanced_engine = AdvancedAnalysisEngine()
                if advanced_engine.available:
                    # Quick test
                    result = advanced_engine.analyze_sentiment("This is a positive test sentence.")
                    if result and 'confidence_level' in result:
                        print("✅ AdvancedAnalysisEngine working correctly")
                        engines_status['advanced'] = "working"
                    else:
                        print("⚠️  AdvancedAnalysisEngine loaded but test failed")
                        engines_status['advanced'] = "partial"
                else:
                    print("❌ AdvancedAnalysisEngine not available")
                    engines_status['advanced'] = "unavailable"
            except Exception as e:
                print(f"❌ AdvancedAnalysisEngine error: {e}")
                engines_status['advanced'] = "error"
            
            # Test HybridAnalysisEngine
            print("\n🔍 Testing HybridAnalysisEngine...")
            try:
                hybrid_engine = HybridAnalysisEngine()
                if hybrid_engine.available:
                    # Quick test
                    result = hybrid_engine.analyze_sentiment("This is a positive test sentence.")
                    if result and 'engine' in result:
                        print("✅ HybridAnalysisEngine working correctly")
                        engines_status['hybrid'] = "working"
                    else:
                        print("⚠️  HybridAnalysisEngine loaded but test failed")
                        engines_status['hybrid'] = "partial"
                else:
                    print("❌ HybridAnalysisEngine not available")
                    engines_status['hybrid'] = "unavailable"
            except Exception as e:
                print(f"❌ HybridAnalysisEngine error: {e}")
                engines_status['hybrid'] = "error"
            
            # Summary
            working_engines = sum(1 for status in engines_status.values() if status == "working")
            total_engines = len(engines_status)
            
            print(f"\n📊 Engine Status: {working_engines}/{total_engines} engines working")
            return working_engines > 0
            
        except ImportError as e:
            print(f"❌ Could not import analysis engines: {e}")
            print("   💡 Make sure machine_psychology_lab_1.py is in the current directory")
            return False
    
    def print_summary(self):
        """Print setup summary."""
        print("\n" + "=" * 60)
        print("📋 SETUP SUMMARY")
        print("=" * 60)
        
        # Count statuses
        available = sum(1 for status in self.models_status.values() if status in ["available", "downloaded"])
        total = len(self.models_status)
        
        print(f"\n📊 Models Status: {available}/{total} models available")
        
        for model_name, status in self.models_status.items():
            status_icon = {
                "available": "✅",
                "downloaded": "📥",
                "missing": "❌",
                "failed": "❌",
                "error": "⚠️"
            }.get(status, "❓")
            print(f"   {status_icon} {model_name}: {status}")
        
        print("\n🎯 Next Steps:")
        if available == total:
            print("✅ All models ready! You can now run Machine Psychology Lab.")
            print("   Run: python machine_psychology_lab_1.py")
        else:
            print("⚠️  Some models are missing. The application will use fallback engines.")
            print("   • BasicAnalysisEngine should work with spaCy models")
            print("   • AdvancedAnalysisEngine requires transformer models")
            print("   • HybridAnalysisEngine works if either engine is available")
            
            if self.offline:
                print("\n💡 You're in offline mode. To download missing models:")
                print("   python setup_models.py  # (without --offline)")
        
        print(f"\n📁 Cache location: {self.cache_dir or 'Default HuggingFace cache'}")
        print("🔧 For troubleshooting, see the Machine Psychology Lab documentation.")

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup models for Machine Psychology Lab")
    parser.add_argument("--offline", action="store_true", 
                       help="Only check cached models, don't download")
    parser.add_argument("--cache-dir", type=str,
                       help="Custom cache directory for models")
    parser.add_argument("--verify-only", action="store_true",
                       help="Only verify existing models, don't download new ones")
    
    args = parser.parse_args()
    
    # Set up cache directory
    if args.cache_dir:
        os.environ['TRANSFORMERS_CACHE'] = args.cache_dir
        os.environ['SENTENCE_TRANSFORMERS_HOME'] = args.cache_dir
    
    setup = ModelSetup(cache_dir=args.cache_dir, offline=args.offline or args.verify_only)
    setup.print_header()
    
    # Check internet connection
    if not args.offline and not args.verify_only:
        print("🌐 Checking internet connection...")
        if setup.check_internet_connection():
            print("✅ Internet connection available")
        else:
            print("❌ No internet connection - switching to offline mode")
            setup.offline = True
    
    # Setup models
    success = True
    
    if not args.verify_only:
        success &= setup.setup_spacy_models()
        success &= setup.setup_transformer_models()
        success &= setup.setup_sentence_transformer_models()
    
    # Verify engines
    success &= setup.verify_analysis_engines()
    
    # Print summary
    setup.print_summary()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
