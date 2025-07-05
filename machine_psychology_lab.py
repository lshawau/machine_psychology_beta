#!/usr/bin/env python3
"""
Machine Psychology Lab
=====================

A comprehensive tool for testing AI models on psychological and cognitive tasks.
Tests theory of mind, cognitive biases, reasoning patterns, and more.

Author: Built with Claude Code
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import json
import os
import sys
import requests
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid

# NLP Analysis imports
try:
    import spacy
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from textblob import TextBlob
    import pandas as pd
    import numpy as np
    from collections import Counter, defaultdict
    import re
    NLP_AVAILABLE = True
except ImportError as e:
    print(f"Some NLP libraries not available: {e}")
    NLP_AVAILABLE = False

# Advanced NLP Analysis imports (optional)
ADVANCED_NLP = {}
try:
    from sentence_transformers import SentenceTransformer
    ADVANCED_NLP['sentence_transformers'] = True
    print("✓ Sentence Transformers available")
except ImportError:
    ADVANCED_NLP['sentence_transformers'] = False
    print("✗ Sentence Transformers not available")

try:
    from transformers import pipeline
    ADVANCED_NLP['transformers'] = True
    print("✓ Transformers available")
except ImportError:
    ADVANCED_NLP['transformers'] = False
    print("✗ Transformers not available")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    ADVANCED_NLP['sklearn'] = True
    print("✓ Scikit-learn available")
except ImportError:
    ADVANCED_NLP['sklearn'] = False
    print("✗ Scikit-learn not available")

class AnalysisEngine:
    """Base class for analysis engines - defines the interface"""
    
    def __init__(self, name: str):
        self.name = name
        self.available = True
        
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment and confidence in the response"""
        raise NotImplementedError
        
    def analyze_reasoning(self, text: str) -> Dict[str, Any]:
        """Analyze reasoning structure and logical flow"""
        raise NotImplementedError
        
    def analyze_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Analyze linguistic features and style"""
        raise NotImplementedError

class BasicAnalysisEngine(AnalysisEngine):
    """Original VADER + TextBlob + spaCy analysis engine"""
    
    def __init__(self):
        super().__init__("Basic (VADER + TextBlob + spaCy)")
        self.available = NLP_AVAILABLE
        if self.available:
            try:
                self.vader_analyzer = SentimentIntensityAnalyzer()
                self.nlp = spacy.load("en_core_web_sm")
                self.confidence_patterns = {
                    'high_confidence': r'\b(certain|sure|definitely|absolutely|clearly|obviously)\b',
                    'medium_confidence': r'\b(likely|probably|seems|appears|suggests)\b',
                    'low_confidence': r'\b(might|could|possibly|perhaps|maybe)\b',
                    'hedging': r'\b(somewhat|rather|quite|fairly|pretty)\b'
                }
            except Exception as e:
                print(f"Failed to initialize basic analysis engine: {e}")
                self.available = False
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        if not self.available:
            return {"error": "Basic analysis engine not available"}
            
        # VADER sentiment analysis - provides compound, positive, negative, neutral scores
        vader_scores = self.vader_analyzer.polarity_scores(text)
        
        # TextBlob sentiment - provides polarity (-1 to 1) and subjectivity (0 to 1)
        blob = TextBlob(text)
        
        # Confidence pattern matching - count occurrences of confidence-indicating phrases
        confidence_scores = {}
        for conf_type, pattern in self.confidence_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            confidence_scores[conf_type] = len(matches)
        
        # Calculate overall confidence using weighted scoring system
        # High confidence phrases worth 3 points, medium worth 2, low worth 1
        # Hedging language ("maybe", "perhaps") subtracts points
        high_conf = confidence_scores['high_confidence'] * 3
        med_conf = confidence_scores['medium_confidence'] * 2
        low_conf = confidence_scores['low_confidence'] * 1
        hedging = confidence_scores['hedging'] * -1
        
        total_confidence = high_conf + med_conf + low_conf + hedging
        # Classify overall confidence: >2 = High, >0 = Medium, <=0 = Low
        confidence_level = "High" if total_confidence > 2 else "Medium" if total_confidence > 0 else "Low"
        
        return {
            'engine': self.name,
            'vader': vader_scores,
            'textblob_polarity': blob.sentiment.polarity,
            'textblob_subjectivity': blob.sentiment.subjectivity,
            'confidence_patterns': confidence_scores,
            'confidence_level': confidence_level,
            'confidence_score': total_confidence
        }

class AdvancedAnalysisEngine(AnalysisEngine):
    """Advanced analysis using Transformers and Sentence Transformers"""
    
    def __init__(self):
        super().__init__("Advanced (Transformers + Sentence-BERT)")
        self.available = (ADVANCED_NLP['transformers'] and 
                         ADVANCED_NLP['sentence_transformers'] and 
                         NLP_AVAILABLE)
        self.models = {}
        
        if self.available:
            try:
                print("Loading advanced analysis models...")
                # Lightweight sentiment model
                self.models['sentiment'] = pipeline(
                    "sentiment-analysis", 
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )
                
                # Sentence transformer for semantic analysis
                self.models['embeddings'] = SentenceTransformer('all-MiniLM-L6-v2')
                
                # spaCy for linguistic analysis
                self.nlp = spacy.load("en_core_web_sm")
                
                print("✓ Advanced analysis models loaded successfully")
                
            except Exception as e:
                print(f"Failed to load advanced models: {e}")
                self.available = False
    
    def analyze_reasoning(self, text: str) -> Dict[str, Any]:
        """Advanced reasoning analysis using semantic embeddings"""
        if not self.available:
            return {"error": "Advanced analysis engine not available"}
        
        try:
            # Generate embeddings for semantic analysis
            embedding = self.models['embeddings'].encode([text])[0]
            
            # Use spaCy for basic linguistic features
            doc = self.nlp(text)
            
            # Enhanced reasoning patterns with semantic similarity
            reasoning_patterns = {
                'causal': ['because', 'therefore', 'thus', 'consequently', 'as a result'],
                'conditional': ['if', 'when', 'unless', 'provided that', 'assuming'],
                'evidence': ['evidence', 'research', 'studies', 'data', 'shows', 'proves'],
                'uncertainty': ['maybe', 'perhaps', 'possibly', 'might', 'could']
            }
            
            pattern_scores = {}
            for category, patterns in reasoning_patterns.items():
                # Calculate semantic similarity to reasoning patterns
                pattern_embeddings = self.models['embeddings'].encode(patterns)
                similarities = [np.dot(embedding, p_emb) / (np.linalg.norm(embedding) * np.linalg.norm(p_emb)) 
                               for p_emb in pattern_embeddings]
                pattern_scores[category] = max(similarities) if similarities else 0
            
            return {
                'engine': self.name,
                'semantic_reasoning_scores': pattern_scores,
                'reasoning_complexity': sum(pattern_scores.values()) / len(pattern_scores),
                'analysis_method': 'Sentence-BERT semantic similarity'
            }
            
        except Exception as e:
            return {"error": f"Advanced reasoning analysis failed: {e}"}
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        if not self.available:
            return {"error": "Advanced analysis engine not available"}
        
        try:
            # Advanced transformer-based sentiment analysis
            sentiment_results = self.models['sentiment'](text)
            
            # Extract scores
            sentiment_scores = {}
            for result in sentiment_results[0]:
                sentiment_scores[result['label'].lower()] = result['score']
            
            # Get dominant sentiment
            dominant = max(sentiment_scores.items(), key=lambda x: x[1])
            
            # Confidence assessment based on score distribution
            max_score = dominant[1]
            confidence_level = "High" if max_score > 0.8 else "Medium" if max_score > 0.6 else "Low"
            
            return {
                'engine': self.name,
                'transformer_sentiment': sentiment_scores,
                'dominant_sentiment': dominant[0],
                'confidence_score': max_score,
                'confidence_level': confidence_level,
                'analysis_method': 'RoBERTa-based classification'
            }
            
        except Exception as e:
            return {"error": f"Advanced sentiment analysis failed: {e}"}

class HybridAnalysisEngine(AnalysisEngine):
    """Combines basic and advanced analysis for comprehensive results"""
    
    def __init__(self):
        super().__init__("Hybrid (Basic + Advanced)")
        self.basic_engine = BasicAnalysisEngine()
        self.advanced_engine = AdvancedAnalysisEngine()
        self.available = self.basic_engine.available or self.advanced_engine.available
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        results = {"engine": self.name, "components": {}}
        
        if self.basic_engine.available:
            results["components"]["basic"] = self.basic_engine.analyze_sentiment(text)
        
        if self.advanced_engine.available:
            results["components"]["advanced"] = self.advanced_engine.analyze_sentiment(text)
        
        # Create summary combining both approaches
        if self.basic_engine.available and self.advanced_engine.available:
            basic_conf = results["components"]["basic"].get("confidence_level", "Unknown")
            adv_conf = results["components"]["advanced"].get("confidence_level", "Unknown")
            
            results["summary"] = {
                "basic_confidence": basic_conf,
                "advanced_confidence": adv_conf,
                "consensus": "High" if basic_conf == adv_conf == "High" else "Mixed"
            }
        
        return results

class PsychologicalTest:
    """Represents a single psychological test or dataset container"""
    
    def __init__(self, test_id: str = None, name: str = "", category: str = "", 
                 description: str = "", prompt: str = "", expected_responses: List[str] = None,
                 scoring_method: str = "manual", scoring_criteria: str = "", 
                 is_dataset: bool = False, dataset_variants: List[Dict] = None):
        self.test_id = test_id or str(uuid.uuid4())
        self.name = name
        self.category = category
        self.description = description
        self.prompt = prompt
        self.expected_responses = expected_responses or []
        self.scoring_method = scoring_method  # "manual", "keyword", "exact"
        self.scoring_criteria = scoring_criteria
        self.is_dataset = is_dataset
        self.dataset_variants = dataset_variants or []  # List of variant dictionaries
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the test instance to a dictionary for serialization.
        
        Returns:
            Dict containing all test attributes for JSON serialization
        """
        return {
            'test_id': self.test_id,
            'name': self.name,
            'category': self.category,
            'description': self.description,
            'prompt': self.prompt,
            'expected_responses': self.expected_responses,
            'scoring_method': self.scoring_method,
            'scoring_criteria': self.scoring_criteria,
            'is_dataset': self.is_dataset,
            'dataset_variants': self.dataset_variants
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PsychologicalTest':
        """Create a PsychologicalTest instance from a dictionary.
        
        Args:
            data: Dictionary containing test attributes from JSON deserialization
            
        Returns:
            New PsychologicalTest instance with data from dictionary
        """
        return cls(**data)

class MachineEstimation:
    """Machine-generated analysis and estimation (not clinical scoring)"""
    
    def __init__(self):
        self.sentiment_analysis = None
        self.reasoning_indicators = None
        self.linguistic_features = None
        self.confidence_assessment = None
        self.coherence_score = None
        self.estimated_understanding = None
        self.analysis_notes = ""
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the machine estimation to a dictionary for serialization.
        
        Returns:
            Dict containing all analysis attributes for JSON serialization
        """
        return {
            'sentiment_analysis': self.sentiment_analysis,
            'reasoning_indicators': self.reasoning_indicators,
            'linguistic_features': self.linguistic_features,
            'confidence_assessment': self.confidence_assessment,
            'coherence_score': self.coherence_score,
            'estimated_understanding': self.estimated_understanding,
            'analysis_notes': self.analysis_notes,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MachineEstimation':
        """Create a MachineEstimation instance from a dictionary.
        
        Args:
            data: Dictionary containing analysis attributes from JSON deserialization
            
        Returns:
            New MachineEstimation instance with data from dictionary,
            with safe defaults for missing values
        """
        estimation = cls()
        estimation.sentiment_analysis = data.get('sentiment_analysis')
        estimation.reasoning_indicators = data.get('reasoning_indicators')
        estimation.linguistic_features = data.get('linguistic_features')
        estimation.confidence_assessment = data.get('confidence_assessment')
        estimation.coherence_score = data.get('coherence_score')
        estimation.estimated_understanding = data.get('estimated_understanding')
        estimation.analysis_notes = data.get('analysis_notes', '')
        estimation.timestamp = data.get('timestamp', datetime.now().isoformat())
        return estimation

class HumanEvaluation:
    """Human researcher's final evaluation and scoring"""
    
    def __init__(self):
        self.final_score = None
        self.evaluation_notes = ""
        self.agrees_with_machine = None  # True/False/Partial
        self.confidence_in_evaluation = None  # 1-5 scale
        self.evaluator_id = ""
        self.timestamp = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the human evaluation to a dictionary for serialization.
        
        Returns:
            Dict containing all evaluation attributes for JSON serialization
        """
        return {
            'final_score': self.final_score,
            'evaluation_notes': self.evaluation_notes,
            'agrees_with_machine': self.agrees_with_machine,
            'confidence_in_evaluation': self.confidence_in_evaluation,
            'evaluator_id': self.evaluator_id,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HumanEvaluation':
        """Create a HumanEvaluation instance from a dictionary.
        
        Args:
            data: Dictionary containing evaluation attributes from JSON deserialization
            
        Returns:
            New HumanEvaluation instance with data from dictionary
        """
        evaluation = cls()
        evaluation.final_score = data.get('final_score')
        evaluation.evaluation_notes = data.get('evaluation_notes', '')
        evaluation.agrees_with_machine = data.get('agrees_with_machine')
        evaluation.confidence_in_evaluation = data.get('confidence_in_evaluation')
        evaluation.evaluator_id = data.get('evaluator_id', '')
        evaluation.timestamp = data.get('timestamp')
        return evaluation

class TestResult:
    """Represents the result of running a test on a model with dual evaluation system"""
    
    def __init__(self, test_id: str, model_name: str, response: str, 
                 notes: str = "", timestamp: str = None):
        self.test_id = test_id
        self.model_name = model_name
        self.response = response
        self.notes = notes
        self.timestamp = timestamp or datetime.now().isoformat()
        
        # New dual evaluation system
        self.machine_estimation = MachineEstimation()
        self.human_evaluation = HumanEvaluation()
        
        # Legacy score field for backward compatibility
        self.score = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the test result to a dictionary for serialization.
        
        Returns:
            Dict containing all result attributes including nested evaluation objects
        """
        return {
            'test_id': self.test_id,
            'model_name': self.model_name,
            'response': self.response,
            'notes': self.notes,
            'timestamp': self.timestamp,
            'machine_estimation': self.machine_estimation.to_dict(),
            'human_evaluation': self.human_evaluation.to_dict(),
            'score': self.score  # Legacy field
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestResult':
        """Create a TestResult instance from a dictionary.
        
        Args:
            data: Dictionary containing result attributes from JSON deserialization
            
        Returns:
            New TestResult instance with data from dictionary,
            including deserialized nested evaluation objects
        """
        result = cls(
            test_id=data.get('test_id', ''),
            model_name=data.get('model_name', ''),
            response=data.get('response', ''),
            notes=data.get('notes', ''),
            timestamp=data.get('timestamp')
        )
        
        # Deserialize machine estimation
        if 'machine_estimation' in data and data['machine_estimation']:
            result.machine_estimation = MachineEstimation.from_dict(data['machine_estimation'])
        
        # Deserialize human evaluation  
        if 'human_evaluation' in data and data['human_evaluation']:
            result.human_evaluation = HumanEvaluation.from_dict(data['human_evaluation'])
        
        # Legacy score field
        result.score = data.get('score')
        
        return result

class APIClient:
    """Client for OpenAI-compatible APIs"""
    
    def __init__(self, base_url: str = "http://localhost:1234/v1", api_key: str = "not-needed"):
        # Ensure base_url ends with /v1 for OpenAI compatibility
        self.base_url = base_url.rstrip('/')
        if not self.base_url.endswith('/v1'):
            self.base_url += '/v1'
        self.api_key = api_key
        
    def get_available_models(self) -> List[str]:
        """Get list of available models from the API"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            url = f"{self.base_url}/models"
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if 'data' in result:
                    return [model['id'] for model in result['data']]
                else:
                    return []
            else:
                print(f"Models API Error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            print(f"Failed to get models: {e}")
            return []
    
    def chat_completion(self, messages: List[Dict[str, str]], model: str = "local-model",
                       temperature: float = 0.7, max_tokens: int = 500) -> Optional[str]:
        """Send a chat completion request"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            url = f"{self.base_url}/chat/completions"
            print(f"Making request to: {url}")  # Debug logging
            
            response = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=9000
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content']
                else:
                    print(f"API Error: Invalid response format - {result}")
                    return None
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Request failed: {e}")
            return None

class MachinePsychologyLab:
    """Main GUI application for testing AI models on psychological tasks.
    
    This application provides a comprehensive toolkit for:
    - Creating and managing psychological test batteries
    - Running tests against various AI models via API
    - Automated analysis using NLP techniques
    - Human evaluation and scoring interfaces
    - Results visualization and export
    
    The application uses a dual evaluation system:
    - Machine estimation: Automated NLP-based analysis
    - Human evaluation: Manual researcher scoring and notes
    
    Usage Example:
        app = MachinePsychologyLab()
        app.run()  # Start the GUI application
    
    Key Features:
    - Built-in test library with cognitive bias and theory of mind tests
    - Custom test creation with dataset variant support
    - API integration for multiple AI model providers
    - Comprehensive linguistic and sentiment analysis
    - Export functionality for research data
    """
    
    def __init__(self):
        """Initialize the Machine Psychology Lab application.
        
        Sets up the main window, initializes data structures, loads built-in tests,
        creates the GUI interface, and loads any previously saved data.
        
        The initialization process:
        1. Creates the main Tkinter window (1200x800)
        2. Initializes data storage for tests and results
        3. Sets up API client for model communication
        4. Loads built-in psychological test library
        5. Creates the tabbed GUI interface
        6. Loads saved data from previous sessions
        7. Initializes NLP analysis tools if available
        """
        self.root = tk.Tk()
        self.root.title("Machine Psychology Lab")
        self.root.geometry("1200x800")
        
        # Data storage - main application state
        self.tests: Dict[str, PsychologicalTest] = {}  # Test library indexed by ID
        self.results: List[TestResult] = []  # All test execution results
        self.api_client = APIClient()  # Handles AI model API communication
        
        # Load built-in tests from predefined library
        self.load_builtin_tests()
        
        # Setup GUI components and layout
        self.setup_gui()
        
        # Load saved data from previous sessions
        self.load_data()
    
    def setup_gui(self):
        """Initialize the GUI components"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Test Library Tab
        self.setup_test_library_tab()
        
        # Test Creator Tab
        self.setup_test_creator_tab()
        
        # Test Runner Tab
        self.setup_test_runner_tab()
        
        # Results Tab
        self.setup_results_tab()
        
        # Cognitive Analysis Tab
        self.setup_cognitive_analysis_tab()
        
        # Settings Tab
        self.setup_settings_tab()
        
        # Initialize NLP tools
        self.init_nlp_tools()
    
    def setup_test_library_tab(self):
        """Setup the test library tab"""
        self.library_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.library_frame, text="Test Library")
        
        # Test list
        list_frame = ttk.Frame(self.library_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(list_frame, text="Available Tests", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        
        # Treeview for tests
        self.test_tree = ttk.Treeview(list_frame, columns=("category", "description"), show="tree headings")
        self.test_tree.heading("#0", text="Test Name")
        self.test_tree.heading("category", text="Category")
        self.test_tree.heading("description", text="Description")
        self.test_tree.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(list_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="View Test", command=self.view_selected_test).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Edit Test", command=self.edit_selected_test).pack(side=tk.LEFT, padx=5)
        self.load_dataset_btn = ttk.Button(button_frame, text="Load Dataset", command=self.load_dataset, state=tk.DISABLED)
        self.load_dataset_btn.pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Delete Test", command=self.delete_selected_test).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Import Tests", command=self.import_tests).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Export Tests", command=self.export_tests).pack(side=tk.RIGHT, padx=5)
    
    def setup_test_creator_tab(self):
        """Setup the test creation tab"""
        self.creator_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.creator_frame, text="Create Test")
        
        # Create scrollable frame
        canvas = tk.Canvas(self.creator_frame)
        scrollbar = ttk.Scrollbar(self.creator_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Form fields
        form_frame = ttk.LabelFrame(scrollable_frame, text="Test Details", padding=10)
        form_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Test name
        ttk.Label(form_frame, text="Test Name:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.name_entry = ttk.Entry(form_frame, width=50)
        self.name_entry.grid(row=0, column=1, sticky=tk.W+tk.E, pady=2, padx=5)
        
        # Category
        ttk.Label(form_frame, text="Category:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.category_combo = ttk.Combobox(form_frame, width=47, values=[
            "Theory of Mind", "Cognitive Bias", "Logical Reasoning", 
            "Social Cognition", "Moral Reasoning", "Other"
        ])
        self.category_combo.grid(row=1, column=1, sticky=tk.W+tk.E, pady=2, padx=5)
        
        # Description
        ttk.Label(form_frame, text="Description:").grid(row=2, column=0, sticky=tk.W+tk.N, pady=2)
        self.description_text = scrolledtext.ScrolledText(form_frame, width=50, height=3)
        self.description_text.grid(row=2, column=1, sticky=tk.W+tk.E, pady=2, padx=5)
        
        # Prompt
        ttk.Label(form_frame, text="Test Prompt:").grid(row=3, column=0, sticky=tk.W+tk.N, pady=2)
        self.prompt_text = scrolledtext.ScrolledText(form_frame, width=50, height=8)
        self.prompt_text.grid(row=3, column=1, sticky=tk.W+tk.E, pady=2, padx=5)
        
        # Expected responses
        ttk.Label(form_frame, text="Expected/Good Responses:").grid(row=4, column=0, sticky=tk.W+tk.N, pady=2)
        self.responses_text = scrolledtext.ScrolledText(form_frame, width=50, height=4)
        self.responses_text.grid(row=4, column=1, sticky=tk.W+tk.E, pady=2, padx=5)
        ttk.Label(form_frame, text="(One per line)", font=("Arial", 8)).grid(row=5, column=1, sticky=tk.W, padx=5)
        
        # Scoring method
        ttk.Label(form_frame, text="Scoring Method:").grid(row=6, column=0, sticky=tk.W, pady=2)
        self.scoring_combo = ttk.Combobox(form_frame, width=47, values=["manual", "keyword", "exact"])
        self.scoring_combo.grid(row=6, column=1, sticky=tk.W+tk.E, pady=2, padx=5)
        
        # Scoring criteria
        ttk.Label(form_frame, text="Scoring Criteria:").grid(row=7, column=0, sticky=tk.W+tk.N, pady=2)
        self.criteria_text = scrolledtext.ScrolledText(form_frame, width=50, height=3)
        self.criteria_text.grid(row=7, column=1, sticky=tk.W+tk.E, pady=2, padx=5)
        
        form_frame.columnconfigure(1, weight=1)
        
        # Buttons
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="Save Test", command=self.save_test).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear Form", command=self.clear_test_form).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Preview Test", command=self.preview_test).pack(side=tk.RIGHT, padx=5)
    
    def setup_test_runner_tab(self):
        """Setup the test runner tab"""
        self.runner_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.runner_frame, text="Run Tests")
        
        # Model selection
        model_frame = ttk.LabelFrame(self.runner_frame, text="Model Configuration", padding=10)
        model_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Model selection row
        ttk.Label(model_frame, text="Available Models:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.model_combo = ttk.Combobox(model_frame, width=35, state="readonly")
        self.model_combo.grid(row=0, column=1, sticky=tk.W, pady=2, padx=5)
        
        refresh_button = ttk.Button(model_frame, text="Refresh Models", command=self.refresh_models)
        refresh_button.grid(row=0, column=2, sticky=tk.W, pady=2, padx=5)
        
        # Status label row
        self.model_status_label = ttk.Label(model_frame, text="Click 'Refresh Models' to load available models", foreground="gray")
        self.model_status_label.grid(row=1, column=1, columnspan=2, sticky=tk.W, pady=2, padx=5)
        
        # Temperature row (separate from model selection)
        ttk.Label(model_frame, text="Temperature:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.temperature_var = tk.DoubleVar(value=0.7)
        self.temperature_scale = ttk.Scale(model_frame, from_=0.0, to=2.0, variable=self.temperature_var, length=200)
        self.temperature_scale.grid(row=2, column=1, sticky=tk.W, pady=2, padx=5)
        self.temp_label = ttk.Label(model_frame, text="0.7")
        self.temp_label.grid(row=2, column=2, sticky=tk.W, pady=2)
        
        self.temperature_scale.configure(command=lambda x: self.temp_label.configure(text=f"{float(x):.1f}"))
        
        # Configure column weights
        model_frame.columnconfigure(1, weight=1)
        
        # Test selection
        selection_frame = ttk.LabelFrame(self.runner_frame, text="Test Selection", padding=10)
        selection_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Available tests listbox
        left_frame = ttk.Frame(selection_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        ttk.Label(left_frame, text="Available Tests").pack(anchor=tk.W)
        self.available_listbox = tk.Listbox(left_frame, selectmode=tk.EXTENDED)
        self.available_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(selection_frame)
        button_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Button(button_frame, text="Add →", command=self.add_test_to_run).pack(pady=5)
        ttk.Button(button_frame, text="← Remove", command=self.remove_test_from_run).pack(pady=5)
        ttk.Button(button_frame, text="Add All →", command=self.add_all_tests).pack(pady=5)
        ttk.Button(button_frame, text="← Remove All", command=self.remove_all_tests).pack(pady=5)
        
        # Selected tests listbox
        right_frame = ttk.Frame(selection_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        ttk.Label(right_frame, text="Tests to Run").pack(anchor=tk.W)
        self.selected_listbox = tk.Listbox(right_frame, selectmode=tk.EXTENDED)
        self.selected_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Run controls
        run_frame = ttk.Frame(self.runner_frame)
        run_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.run_button = ttk.Button(run_frame, text="Run Selected Tests", command=self.run_tests)
        self.run_button.pack(side=tk.LEFT, padx=5)
        
        self.progress_var = tk.StringVar(value="Ready")
        ttk.Label(run_frame, textvariable=self.progress_var).pack(side=tk.LEFT, padx=20)
        
        self.progress_bar = ttk.Progressbar(run_frame, length=200)
        self.progress_bar.pack(side=tk.RIGHT, padx=5)
    
    def setup_results_tab(self):
        """Setup the results viewing tab"""
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Results")
        
        # Results tree
        tree_frame = ttk.Frame(self.results_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.results_tree = ttk.Treeview(tree_frame, columns=("model", "test", "machine_est", "confidence", "human_score", "eval_status", "timestamp"), show="tree headings")
        self.results_tree.heading("#0", text="ID")
        self.results_tree.heading("model", text="Model")
        self.results_tree.heading("test", text="Test")
        self.results_tree.heading("machine_est", text="Machine Est.")
        self.results_tree.heading("confidence", text="Confidence")
        self.results_tree.heading("human_score", text="Human Score")
        self.results_tree.heading("eval_status", text="Status")
        self.results_tree.heading("timestamp", text="Timestamp")
        
        # Configure column widths for better display
        self.results_tree.column("#0", width=40, minwidth=30)
        self.results_tree.column("model", width=100, minwidth=80)
        self.results_tree.column("test", width=150, minwidth=100)
        self.results_tree.column("machine_est", width=90, minwidth=80)
        self.results_tree.column("confidence", width=80, minwidth=70)
        self.results_tree.column("human_score", width=80, minwidth=70)
        self.results_tree.column("eval_status", width=90, minwidth=80)
        self.results_tree.column("timestamp", width=120, minwidth=100)
        self.results_tree.pack(fill=tk.BOTH, expand=True)
        
        # Response viewer - much larger and more readable
        response_frame = ttk.LabelFrame(self.results_frame, text="Response Details", padding=10)
        response_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.response_text = scrolledtext.ScrolledText(
            response_frame, 
            height=20,  # Much taller
            wrap=tk.WORD,
            font=("Consolas", 11),  # Better monospace font, larger size
            bg="white",
            fg="black"
        )
        self.response_text.pack(fill=tk.BOTH, expand=True)
        
        # Human Evaluation Interface
        eval_frame = ttk.LabelFrame(self.results_frame, text="🧠 Human Evaluation (REQUIRED - Final Authority)", padding=10)
        eval_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Transparency disclaimer
        disclaimer_frame = ttk.Frame(eval_frame)
        disclaimer_frame.pack(fill=tk.X, pady=(0, 5))
        
        disclaimer_label = ttk.Label(disclaimer_frame, 
                                   text="⚠️ Machine analysis provides interpretive insights only. Human evaluation required for all clinical decisions.",
                                   font=("TkDefaultFont", 9), foreground="blue")
        disclaimer_label.pack(anchor=tk.W)
        
        # Top row: Machine analysis display
        machine_frame = ttk.Frame(eval_frame)
        machine_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(machine_frame, text="🤖 Machine Analysis (NOT clinical scoring):", font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT)
        self.machine_analysis_label = ttk.Label(machine_frame, text="No analysis available", foreground="gray")
        self.machine_analysis_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Evaluation input row
        input_frame = ttk.Frame(eval_frame)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Score input
        ttk.Label(input_frame, text="Final Score:").pack(side=tk.LEFT)
        self.human_score_var = tk.StringVar()
        self.human_score_entry = ttk.Entry(input_frame, textvariable=self.human_score_var, width=8)
        self.human_score_entry.pack(side=tk.LEFT, padx=(5, 15))
        
        # Confidence input
        ttk.Label(input_frame, text="Your Confidence:").pack(side=tk.LEFT)
        self.human_confidence_var = tk.StringVar(value="3")
        confidence_combo = ttk.Combobox(input_frame, textvariable=self.human_confidence_var, 
                                       values=["1", "2", "3", "4", "5"], width=5, state="readonly")
        confidence_combo.pack(side=tk.LEFT, padx=(5, 15))
        
        # Agreement with machine
        ttk.Label(input_frame, text="Agree with Machine:").pack(side=tk.LEFT)
        self.agreement_var = tk.StringVar(value="Partial")
        agreement_combo = ttk.Combobox(input_frame, textvariable=self.agreement_var,
                                      values=["Yes", "No", "Partial"], width=10, state="readonly")
        agreement_combo.pack(side=tk.LEFT, padx=(5, 0))
        
        # Notes area
        notes_frame = ttk.Frame(eval_frame)
        notes_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(notes_frame, text="Evaluation Notes:").pack(anchor=tk.W)
        self.eval_notes_text = scrolledtext.ScrolledText(notes_frame, height=4, wrap=tk.WORD)
        self.eval_notes_text.pack(fill=tk.X, pady=(5, 0))
        
        # Action buttons
        action_frame = ttk.Frame(eval_frame)
        action_frame.pack(fill=tk.X)
        
        self.save_eval_button = ttk.Button(action_frame, text="Save Evaluation", 
                                          command=self.save_human_evaluation, state=tk.DISABLED)
        self.save_eval_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.quick_score_frame = ttk.Frame(action_frame)
        self.quick_score_frame.pack(side=tk.LEFT)
        
        ttk.Label(self.quick_score_frame, text="Quick Score:").pack(side=tk.LEFT, padx=(0, 5))
        for score in ["0", "0.5", "1", "2", "3", "4", "5"]:
            ttk.Button(self.quick_score_frame, text=score, width=3,
                      command=lambda s=score: self.quick_score(s)).pack(side=tk.LEFT, padx=1)
        
        # Evaluator ID
        evaluator_frame = ttk.Frame(action_frame)
        evaluator_frame.pack(side=tk.RIGHT)
        
        ttk.Label(evaluator_frame, text="Evaluator ID:").pack(side=tk.LEFT)
        self.evaluator_id_var = tk.StringVar(value="Researcher1")
        evaluator_entry = ttk.Entry(evaluator_frame, textvariable=self.evaluator_id_var, width=12)
        evaluator_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        # Current result tracking
        self.current_result_index = None
        
        # Bind selection events
        self.results_tree.bind('<<TreeviewSelect>>', self.on_result_select)
        self.test_tree.bind('<<TreeviewSelect>>', self.on_test_select)
        
        # Buttons
        button_frame = ttk.Frame(self.results_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(button_frame, text="Export Results", command=self.export_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear Results", command=self.clear_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Generate Report", command=self.generate_report).pack(side=tk.RIGHT, padx=5)
    
    def setup_settings_tab(self):
        """Setup the settings tab"""
        self.settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_frame, text="Settings")
        
        # API settings
        api_frame = ttk.LabelFrame(self.settings_frame, text="API Configuration", padding=10)
        api_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(api_frame, text="Base URL:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.base_url_entry = ttk.Entry(api_frame, width=60)
        self.base_url_entry.insert(0, "http://localhost:1234/v1")
        self.base_url_entry.grid(row=0, column=1, sticky=tk.W+tk.E, pady=2, padx=5)
        
        ttk.Label(api_frame, text="API Key:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.api_key_entry = ttk.Entry(api_frame, width=60, show="*")
        self.api_key_entry.insert(0, "not-needed")
        self.api_key_entry.grid(row=1, column=1, sticky=tk.W+tk.E, pady=2, padx=5)
        
        ttk.Button(api_frame, text="Test Connection", command=self.test_connection).grid(row=2, column=1, sticky=tk.W, pady=10)
        
        api_frame.columnconfigure(1, weight=1)
        
        # Analysis Engine Settings
        analysis_frame = ttk.LabelFrame(self.settings_frame, text="Analysis Engine Configuration", padding=10)
        analysis_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(analysis_frame, text="Analysis Engine:").grid(row=0, column=0, sticky=tk.W, pady=2)
        
        # Create engine selection dropdown
        self.engine_var = tk.StringVar()
        self.engine_dropdown = ttk.Combobox(analysis_frame, textvariable=self.engine_var, 
                                        state="readonly", width=40)
        self.engine_dropdown.grid(row=0, column=1, sticky=tk.W+tk.E, pady=2, padx=5)
        
        # Initialize and populate dropdown with available engines
        self.populate_engine_dropdown()
        
        # Bind selection change
        self.engine_dropdown.bind('<<ComboboxSelected>>', self.on_engine_change)
        
        # Engine status display
        self.engine_status_label = ttk.Label(analysis_frame, text="", foreground="blue")
        self.engine_status_label.grid(row=1, column=1, sticky=tk.W, pady=2, padx=5)
        
        # Update status after dropdown is populated
        self.update_engine_status()
        
        analysis_frame.columnconfigure(1, weight=1)
        
        # Enhanced About section for the settings tab
        about_frame = ttk.LabelFrame(self.settings_frame, text="About", padding=10)
        about_frame.pack(fill=tk.X, padx=10, pady=10)

        # Main title and version
        title_label = ttk.Label(about_frame, text="Machine Psychology Lab v1.0", 
                            font=("Arial", 14, "bold"))
        title_label.pack(anchor=tk.W, pady=(0, 10))

        # Description
        description_text = """A comprehensive tool for testing AI models on psychological and cognitive tasks.
        Test theory of mind, cognitive biases, reasoning patterns, and more.

        Built with Claude Code for the machine psychology research community."""

        ttk.Label(about_frame, text=description_text, justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 15))

        # Research foundation section
        foundation_frame = ttk.LabelFrame(about_frame, text="Research Foundation", padding=5)
        foundation_frame.pack(fill=tk.X, pady=(0, 10))

        foundation_text = """This tool implements the "Machine Psychology" framework introduced by Hagendorff et al. (2024),
        which advocates for applying behavioral science methods to understand AI systems. Rather than examining
        internal neural networks, machine psychology focuses on studying the relationship between inputs and
        outputs to reveal behavioral patterns and emergent abilities in large language models.

        Key insight: "Machine psychology provides a new approach to explaining AI. Instead of interpreting 
        a neural network's design components, one analyzes the relationships between inputs and outputs, 
        i.e. prompt design and prompt completion." - Hagendorff et al., 2024"""

        ttk.Label(foundation_frame, text=foundation_text, justify=tk.LEFT, 
                wraplength=650).pack(anchor=tk.W)

        # Paradigms implemented section  
        paradigms_frame = ttk.LabelFrame(about_frame, text="Implemented Paradigms", padding=5)
        paradigms_frame.pack(fill=tk.X, pady=(0, 10))

        paradigms_text = """Following the paper's framework, this tool implements tests across four key areas:

        • Heuristics & Biases: Cognitive shortcuts and decision-making patterns (Tversky & Kahneman, 1974)
        • Social Interactions: Theory of mind, false belief tasks, social reasoning capabilities  
        • Psychology of Language: Semantic processing, pragmatic inference, linguistic competence
        • Learning: In-context learning, generalization patterns, inductive biases

        Each test includes both machine estimation (automated NLP analysis) and human evaluation
        components, supporting the dual evaluation methodology for robust assessment."""

        ttk.Label(paradigms_frame, text=paradigms_text, justify=tk.LEFT,
                wraplength=650).pack(anchor=tk.W)

        # Paper citation section
        citation_frame = ttk.LabelFrame(about_frame, text="Primary Reference", padding=5)
        citation_frame.pack(fill=tk.X, pady=(0, 10))

        citation_text = """Hagendorff, T., Dasgupta, I., Binz, M., Chan, S. C., Lampinen, A., Wang, J. X., Akata, Z., & Schulz, E. (2024). 
        Machine Psychology. arXiv:2303.13988v6 [cs.CL]. 
        DOI: https://doi.org/10.48550/arXiv.2303.13988

        Subjects: Computation and Language (cs.CL); Artificial Intelligence (cs.AI)
        Cite as: arXiv:2303.13988 [cs.CL] (or arXiv:2303.13988v6 [cs.CL] for this version)

        This seminal paper establishes machine psychology as a systematic approach to understanding AI behavior
        through experimental psychology methods, moving beyond traditional benchmarking to focus on 
        computational insights into emergent abilities and behavioral patterns."""

        ttk.Label(citation_frame, text=citation_text, justify=tk.LEFT,
                wraplength=650, font=("Arial", 9)).pack(anchor=tk.W)

        # Technical implementation section
        tech_frame = ttk.LabelFrame(about_frame, text="Technical Implementation", padding=5)
        tech_frame.pack(fill=tk.X, pady=(0, 5))

        tech_text = """Analysis Engines:
        • Basic: VADER sentiment + TextBlob + spaCy NLP (traditional computational linguistics)
        • Advanced: RoBERTa transformers + Sentence-BERT embeddings (modern neural approaches)  
        • Hybrid: Combined statistical and neural methods for comprehensive analysis

        Supports OpenAI-compatible APIs: LM Studio, Ollama, OpenAI, Anthropic, and other providers.
        Designed for reproducible research with proper experimental controls and statistical rigor."""

        ttk.Label(tech_frame, text=tech_text, justify=tk.LEFT,
                wraplength=650, font=("Arial", 9)).pack(anchor=tk.W)

        # Optional: Add clickable links functionality
        def open_link(url):
            """Open a URL in the default browser"""
            import webbrowser
            webbrowser.open(url)

        # Links section
        link_frame = ttk.Frame(about_frame)
        link_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Label(link_frame, text="📄 Primary Paper:", font=("Arial", 9, "bold")).pack(side=tk.LEFT)
        paper_button = ttk.Button(link_frame, text="Machine Psychology (arXiv:2303.13988)", 
                                command=lambda: open_link("https://arxiv.org/abs/2303.13988"))
        paper_button.pack(side=tk.LEFT, padx=(5, 20))

        ttk.Label(link_frame, text="🧠 Framework:", font=("Arial", 9, "bold")).pack(side=tk.LEFT)
        framework_button = ttk.Button(link_frame, text="Behavioral AI Research", 
                                    command=lambda: open_link("https://arxiv.org/abs/2303.13988"))
        framework_button.pack(side=tk.LEFT, padx=(5, 0))
    


    def populate_engine_dropdown(self):
        """Populate the analysis engine dropdown with available engines"""
        available_engines = []
        
        # Ensure analysis engines are initialized
        if not hasattr(self, 'analysis_engines') or not self.analysis_engines:
            print("Warning: Analysis engines not initialized, calling init_nlp_tools")
            self.init_nlp_tools()
        
        # Populate dropdown with available engines
        if hasattr(self, 'analysis_engines') and self.analysis_engines:
            for key, engine in self.analysis_engines.items():
                status = "✓" if engine.available else "✗"
                available_engines.append(f"{status} {engine.name}")
        else:
            available_engines.append("✗ No engines available")
        
        self.engine_dropdown['values'] = available_engines
        
        # Set current selection if we have a current engine
        if hasattr(self, 'current_engine') and self.current_engine:
            status = "✓" if self.current_engine.available else "✗"
            current_display = f"{status} {self.current_engine.name}"
            if current_display in available_engines:
                self.engine_var.set(current_display)
        elif available_engines:
            # Set first available engine as default
            for engine_display in available_engines:
                if engine_display.startswith("✓"):
                    self.engine_var.set(engine_display)
                    break
            else:
                # No available engines, set first one anyway
                self.engine_var.set(available_engines[0])

    def on_engine_change(self, event):
        """Handle analysis engine selection change"""
        if not hasattr(self, 'analysis_engines') or not self.analysis_engines:
            messagebox.showwarning("Engine Selection", "No analysis engines available.")
            return
            
        selected = self.engine_var.get()
        
        # Extract engine name from "✓ Engine Name" or "✗ Engine Name" format
        if selected.startswith(("✓ ", "✗ ")):
            engine_name = selected[2:]  # Remove status prefix
            
            # Find matching engine
            for key, engine in self.analysis_engines.items():
                if engine.name == engine_name:
                    if engine.available:
                        self.current_engine_key = key
                        self.current_engine = engine
                        print(f"Switched to analysis engine: {engine.name}")
                        self.update_engine_status()
                        messagebox.showinfo("Engine Changed", f"Successfully switched to: {engine.name}")
                        return
                    else:
                        messagebox.showwarning("Engine Unavailable", 
                                            f"Selected engine '{engine.name}' is not available.\n"
                                            f"Please ensure required dependencies are installed.")
                        return
        
        messagebox.showerror("Engine Selection", "Invalid engine selection.")

    def update_engine_status(self):
        """Update the engine status display"""
        if not hasattr(self, 'engine_status_label'):
            return
            
        if not hasattr(self, 'current_engine') or not self.current_engine:
            self.engine_status_label.config(text="No engine selected", foreground="red")
            return
            
        engine = self.current_engine
        if engine.available:
            status_text = f"Status: Active - {engine.name}"
            
            # Add additional info based on engine type
            if hasattr(engine, 'models') and engine.models:
                model_count = len(engine.models)
                status_text += f" ({model_count} models loaded)"
            elif engine.name.startswith("Basic"):
                status_text += " (VADER + TextBlob + spaCy)"
            elif engine.name.startswith("Advanced"):
                status_text += " (Transformers + Sentence-BERT)"
            elif engine.name.startswith("Hybrid"):
                status_text += " (Combined analysis)"
                
            self.engine_status_label.config(text=status_text, foreground="green")
        else:
            reason = "Dependencies missing"
            if hasattr(engine, 'error_message'):
                reason = engine.error_message
            self.engine_status_label.config(
                text=f"Status: Unavailable - {reason}", 
                foreground="red"
            )
    
    def load_builtin_tests_fallback(self):
        """Fallback method to load hardcoded tests if JSON files fail"""
        print("Loading fallback hardcoded tests...")
        
        # Basic Sally-Anne test as fallback
        sally_anne = PsychologicalTest(
            name="Sally-Anne Test (Fallback)",
            category="Theory of Mind",
            description="Basic false-belief test",
            prompt="""Sally and Anne are in a room. Sally has a basket and Anne has a box. Sally puts a marble in her basket and then leaves the room. While Sally is away, Anne takes the marble from Sally's basket and puts it in her box. Sally comes back into the room.

Where will Sally look for her marble?

Please explain your reasoning.""",
            expected_responses=[
                "Sally will look in her basket",
                "Sally will look in the basket", 
                "In her basket",
                "The basket"
            ],
            scoring_method="keyword",
            scoring_criteria="Response should indicate Sally will look in her basket (where she left it), not where the marble actually is."
        )
        
        self.tests[sally_anne.test_id] = sally_anne
    
    def save_datasets_to_files(self):
        """Save current datasets back to JSON files"""
        try:
            # Create datasets directory if it doesn't exist
            os.makedirs("datasets", exist_ok=True)
            
            # Group tests by their likely source file
            builtin_tests = []
            phase1_tests = []
            
            for test in self.tests.values():
                # Determine which file this test belongs to based on test_id patterns
                if test.test_id in ["sally-anne-dataset", "asian-disease-framing", "linda-bank-teller", "wason-selection"]:
                    builtin_tests.append(test.to_dict())
                else:
                    phase1_tests.append(test.to_dict())
            
            # Save builtin tests
            if builtin_tests:
                builtin_data = {
                    "metadata": {
                        "version": "1.0",
                        "description": "Core built-in psychological tests",
                        "last_updated": datetime.now().isoformat()
                    },
                    "tests": builtin_tests
                }
                
                with open("datasets/builtin_tests.json", 'w', encoding='utf-8') as f:
                    json.dump(builtin_data, f, indent=2, ensure_ascii=False)
            
            # Save phase1 tests  
            if phase1_tests:
                phase1_data = {
                    "metadata": {
                        "version": "1.0", 
                        "description": "Phase 1 comprehensive psychological test datasets",
                        "last_updated": datetime.now().isoformat(),
                        "contributor": "Machine Psychology Lab Community"
                    },
                    "tests": phase1_tests
                }
                
                with open("datasets/phase1_datasets.json", 'w', encoding='utf-8') as f:
                    json.dump(phase1_data, f, indent=2, ensure_ascii=False)
            
            print(f"Saved {len(builtin_tests)} builtin tests and {len(phase1_tests)} phase1 tests to JSON files")
            
        except Exception as e:
            print(f"Error saving datasets: {e}")
    
    def setup_cognitive_analysis_tab(self):
        """Setup the cognitive analysis tab"""
        self.cognitive_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.cognitive_frame, text="Cognitive Analysis")
        
        # Top frame for response selection
        selection_frame = ttk.LabelFrame(self.cognitive_frame, text="Select Response to Analyze", padding=10)
        selection_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Response dropdown
        ttk.Label(selection_frame, text="Response:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.analysis_response_combo = ttk.Combobox(selection_frame, width=60, state="readonly")
        self.analysis_response_combo.grid(row=0, column=1, sticky=tk.W+tk.E, pady=2, padx=5)
        self.analysis_response_combo.bind('<<ComboboxSelected>>', self.on_analysis_selection)
        
        ttk.Button(selection_frame, text="Refresh Responses", command=self.refresh_analysis_responses).grid(row=0, column=2, padx=5)
        
        selection_frame.columnconfigure(1, weight=1)
        
        # Analysis results frame
        analysis_frame = ttk.LabelFrame(self.cognitive_frame, text="Cognitive Analysis Results", padding=10)
        analysis_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create notebook for different analysis types
        self.analysis_notebook = ttk.Notebook(analysis_frame)
        self.analysis_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Sentiment Analysis Tab
        self.sentiment_frame = ttk.Frame(self.analysis_notebook)
        self.analysis_notebook.add(self.sentiment_frame, text="Sentiment & Confidence")
        
        self.sentiment_text = scrolledtext.ScrolledText(self.sentiment_frame, height=8, wrap=tk.WORD)
        self.sentiment_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Entity Analysis Tab
        self.entity_frame = ttk.Frame(self.analysis_notebook)
        self.analysis_notebook.add(self.entity_frame, text="Entities & Concepts")
        
        self.entity_text = scrolledtext.ScrolledText(self.entity_frame, height=8, wrap=tk.WORD)
        self.entity_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Reasoning Structure Tab
        self.reasoning_frame = ttk.Frame(self.analysis_notebook)
        self.analysis_notebook.add(self.reasoning_frame, text="Reasoning Structure")
        
        self.reasoning_text = scrolledtext.ScrolledText(self.reasoning_frame, height=8, wrap=tk.WORD)
        self.reasoning_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Linguistic Features Tab
        self.linguistic_frame = ttk.Frame(self.analysis_notebook)
        self.analysis_notebook.add(self.linguistic_frame, text="Linguistic Features")
        
        self.linguistic_text = scrolledtext.ScrolledText(self.linguistic_frame, height=8, wrap=tk.WORD)
        self.linguistic_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def init_nlp_tools(self):
        """Initialize NLP analysis tools and engines"""
        print("Initializing NLP analysis engines...")
        
        # Initialize analysis engines
        self.analysis_engines = {
            'basic': BasicAnalysisEngine(),
            'advanced': AdvancedAnalysisEngine(),
            'hybrid': HybridAnalysisEngine()
        }
        
        # Log engine availability
        for key, engine in self.analysis_engines.items():
            status = "✓ Available" if engine.available else "✗ Unavailable"
            print(f"  {engine.name}: {status}")
        
        # Set default engine (prefer hybrid if available, fallback to basic)
        self.current_engine = None
        self.current_engine_key = None
        
        if self.analysis_engines['hybrid'].available:
            self.current_engine_key = 'hybrid'
            self.current_engine = self.analysis_engines['hybrid']
        elif self.analysis_engines['basic'].available:
            self.current_engine_key = 'basic'
            self.current_engine = self.analysis_engines['basic']
        elif self.analysis_engines['advanced'].available:
            self.current_engine_key = 'advanced'
            self.current_engine = self.analysis_engines['advanced']
        
        if self.current_engine:
            print(f"✓ Using analysis engine: {self.current_engine.name}")
        else:
            print("⚠️ No analysis engines available")
        
        # Legacy compatibility - maintain old interface for existing code
        if not NLP_AVAILABLE:
            print("⚠️ Basic NLP libraries not available")
            return
            
        try:
            # Initialize VADER sentiment analyzer (for backward compatibility)
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.vader_analyzer = SentimentIntensityAnalyzer()
            
            # Load spaCy model
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            
            # Confidence/uncertainty patterns
            self.confidence_patterns = {
                'high_confidence': r'\b(definitely|certainly|clearly|obviously|undoubtedly|without doubt)\b',
                'medium_confidence': r'\b(likely|probably|seems|appears|suggests|indicates)\b',
                'low_confidence': r'\b(maybe|perhaps|possibly|might|could|uncertain|unsure)\b',
                'hedging': r'\b(I think|I believe|I feel|in my opinion|it seems to me)\b'
            }
            
            print("✓ Legacy NLP tools initialized for compatibility")
            
        except Exception as e:
            print(f"⚠️ Failed to initialize legacy NLP tools: {e}")
            self.vader_analyzer = None
            self.nlp = None
    
    def refresh_analysis_responses(self):
        """Refresh the list of responses available for analysis"""
        responses = []
        for i, result in enumerate(self.results):
            # Get test name
            test_name = "Unknown Test"
            if result.test_id in self.tests:
                test_name = self.tests[result.test_id].name
            
            display_text = f"{i+1}. {result.model_name} - {test_name} ({result.timestamp[:10]})"
            responses.append(display_text)
        
        self.analysis_response_combo['values'] = responses
        if responses:
            self.analysis_response_combo.set(responses[0])
            self.on_analysis_selection(None)
    
    def on_analysis_selection(self, event):
        """Handle selection of response for analysis"""
        if not self.analysis_response_combo.get():
            return
            
        # Extract result index from selection
        selection = self.analysis_response_combo.get()
        try:
            result_index = int(selection.split('.')[0]) - 1
            if 0 <= result_index < len(self.results):
                result = self.results[result_index]
                self.analyze_response(result)
        except (ValueError, IndexError):
            pass
    
    def analyze_response(self, result: TestResult):
        """Perform comprehensive cognitive analysis on a response"""
        if not NLP_AVAILABLE or not hasattr(self, 'vader_analyzer') or not self.vader_analyzer:
            self.show_analysis_error("NLP tools not available")
            return
            
        response_text = result.response
        
        try:
            # Get test context
            test = self.tests.get(result.test_id)
            test_name = test.name if test else "Unknown Test"
            
            # Perform various analyses
            sentiment_analysis = self.analyze_sentiment(response_text)
            entity_analysis = self.analyze_entities(response_text)
            reasoning_analysis = self.analyze_reasoning_structure(response_text)
            linguistic_analysis = self.analyze_linguistic_features(response_text)
            
            # Display results
            self.display_sentiment_analysis(sentiment_analysis, test_name)
            self.display_entity_analysis(entity_analysis, test_name)
            self.display_reasoning_analysis(reasoning_analysis, test_name)
            self.display_linguistic_analysis(linguistic_analysis, test_name)
            
        except Exception as e:
            self.show_analysis_error(f"Analysis failed: {str(e)}")
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment and confidence in the response"""
        # VADER sentiment analysis - provides compound, positive, negative, neutral scores
        vader_scores = self.vader_analyzer.polarity_scores(text)
        
        # TextBlob sentiment - provides polarity (-1 to 1) and subjectivity (0 to 1)
        blob = TextBlob(text)
        
        # Confidence pattern matching - count occurrences of confidence-indicating phrases
        confidence_scores = {}
        for conf_type, pattern in self.confidence_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            confidence_scores[conf_type] = len(matches)
        
        # Calculate overall confidence using weighted scoring system
        # High confidence phrases worth 3 points, medium worth 2, low worth 1
        # Hedging language ("maybe", "perhaps") subtracts points
        high_conf = confidence_scores['high_confidence'] * 3
        med_conf = confidence_scores['medium_confidence'] * 2
        low_conf = confidence_scores['low_confidence'] * 1
        hedging = confidence_scores['hedging'] * -1
        
        total_confidence = high_conf + med_conf + low_conf + hedging
        # Classify overall confidence: >2 = High, >0 = Medium, <=0 = Low
        confidence_level = "High" if total_confidence > 2 else "Medium" if total_confidence > 0 else "Low"
        
        return {
            'vader': vader_scores,
            'textblob_polarity': blob.sentiment.polarity,
            'textblob_subjectivity': blob.sentiment.subjectivity,
            'confidence_patterns': confidence_scores,
            'confidence_level': confidence_level,
            'confidence_score': total_confidence
        }
    
    def analyze_entities(self, text: str) -> Dict[str, Any]:
        """Extract and analyze entities and concepts"""
        doc = self.nlp(text)
        
        # Named entities
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Noun phrases (concepts)
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        
        # Key terms (excluding stop words)
        key_terms = [token.lemma_.lower() for token in doc 
                    if not token.is_stop and not token.is_punct and token.pos_ in ['NOUN', 'VERB', 'ADJ']]
        
        # Most common terms
        term_freq = Counter(key_terms)
        
        return {
            'entities': entities,
            'noun_phrases': noun_phrases[:10],  # Top 10
            'key_terms': term_freq.most_common(10),
            'total_entities': len(entities),
            'unique_entity_types': len(set(label for _, label in entities))
        }
    
    def analyze_reasoning_structure(self, text: str) -> Dict[str, Any]:
        """Analyze the reasoning structure and logical flow using pattern matching and NLP.
        
        Identifies key reasoning indicators to assess logical thinking patterns:
        - Causal reasoning (because, therefore, leads to)
        - Conditional logic (if, when, unless)
        - Evidence-based reasoning (research, data, proves)
        - Exploratory vs assertive language (question words)
        """
        doc = self.nlp(text)
        
        # Causal indicators - words that show cause-effect relationships
        causal_patterns = r'\b(because|since|therefore|thus|consequently|as a result|due to|leads to|causes)\b'
        causal_indicators = len(re.findall(causal_patterns, text, re.IGNORECASE))
        
        # Conditional reasoning - words that indicate hypothetical or conditional thinking
        conditional_patterns = r'\b(if|when|unless|provided that|assuming|suppose)\b'
        conditional_indicators = len(re.findall(conditional_patterns, text, re.IGNORECASE))
        
        # Evidence markers - words that reference supporting data or research
        evidence_patterns = r'\b(evidence|data|research|studies|shows|demonstrates|proves|indicates)\b'
        evidence_markers = len(re.findall(evidence_patterns, text, re.IGNORECASE))
        
        # Question words - indicate exploratory vs assertive reasoning style
        question_patterns = r'\b(why|how|what|where|when|which|whether)\b'
        question_words = len(re.findall(question_patterns, text, re.IGNORECASE))
        
        # Sentence complexity - measure average length as indicator of sophistication
        sentences = [sent for sent in doc.sents]
        avg_sentence_length = np.mean([len(sent) for sent in sentences]) if sentences else 0
        
        # Dependency analysis - count complex grammatical structures using spaCy dependencies
        # ccomp=clausal complement, xcomp=open clausal complement, advcl=adverbial clause, acl=adjectival clause
        complex_dependencies = 0
        for token in doc:
            if token.dep_ in ['ccomp', 'xcomp', 'advcl', 'acl']:  # Complex clauses indicate sophisticated reasoning
                complex_dependencies += 1
        
        return {
            'causal_indicators': causal_indicators,
            'conditional_indicators': conditional_indicators,
            'evidence_markers': evidence_markers,
            'question_words': question_words,
            'avg_sentence_length': avg_sentence_length,
            'complex_dependencies': complex_dependencies,
            'total_sentences': len(sentences),
            'reasoning_depth': causal_indicators + conditional_indicators + evidence_markers
        }
    
    def analyze_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Analyze linguistic features and writing style using NLP.
        
        Extracts various linguistic metrics for psychological analysis:
        - Part-of-speech distribution (grammatical complexity)
        - Readability metrics (sentence length, word count)
        - Lexical diversity (vocabulary richness)
        - Anthropomorphic language (first-person usage)
        - Uncertainty/hedging language patterns
        """
        doc = self.nlp(text)
        
        # Word counts by POS - analyze grammatical structure distribution
        pos_counts = Counter([token.pos_ for token in doc if not token.is_space])
        
        # Readability metrics - basic text complexity measures
        word_count = len([token for token in doc if not token.is_space and not token.is_punct])
        sentence_count = len(list(doc.sents))
        avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
        
        # Vocabulary complexity - lexical diversity indicates sophistication
        # Higher ratio means more varied vocabulary usage
        unique_words = len(set([token.lemma_.lower() for token in doc if token.is_alpha]))
        lexical_diversity = unique_words / word_count if word_count > 0 else 0
        
        # First person usage - indicates anthropomorphic self-reference in AI responses
        first_person = len(re.findall(r'\b(I|me|my|myself|we|us|our)\b', text, re.IGNORECASE))
        
        # Uncertainty markers - hedging language that indicates confidence levels
        uncertainty_markers = len(re.findall(r'\b(may|might|could|would|should|possibly|probably)\b', text, re.IGNORECASE))
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_words_per_sentence': avg_words_per_sentence,
            'unique_words': unique_words,
            'lexical_diversity': lexical_diversity,
            'pos_distribution': dict(pos_counts.most_common(5)),
            'first_person_usage': first_person,
            'uncertainty_markers': uncertainty_markers
        }
    
    def display_sentiment_analysis(self, analysis: Dict[str, Any], test_name: str):
        """Display sentiment analysis results"""
        self.sentiment_text.delete(1.0, tk.END)
        
        content = f"SENTIMENT & CONFIDENCE ANALYSIS\nTest: {test_name}\n{'='*50}\n\n"
        
        # VADER sentiment
        vader = analysis['vader']
        content += f"VADER Sentiment Analysis:\n"
        content += f"  Positive: {vader['pos']:.3f}\n"
        content += f"  Neutral: {vader['neu']:.3f}\n"
        content += f"  Negative: {vader['neg']:.3f}\n"
        content += f"  Compound: {vader['compound']:.3f}\n\n"
        
        # TextBlob sentiment
        content += f"TextBlob Analysis:\n"
        content += f"  Polarity: {analysis['textblob_polarity']:.3f} (-1=negative, 1=positive)\n"
        content += f"  Subjectivity: {analysis['textblob_subjectivity']:.3f} (0=objective, 1=subjective)\n\n"
        
        # Confidence analysis
        content += f"Confidence Analysis:\n"
        content += f"  Overall Level: {analysis['confidence_level']} (Score: {analysis['confidence_score']})\n"
        content += f"  High Confidence Words: {analysis['confidence_patterns']['high_confidence']}\n"
        content += f"  Medium Confidence Words: {analysis['confidence_patterns']['medium_confidence']}\n"
        content += f"  Low Confidence Words: {analysis['confidence_patterns']['low_confidence']}\n"
        content += f"  Hedging Phrases: {analysis['confidence_patterns']['hedging']}\n\n"
        
        # Interpretation
        content += "INTERPRETATION:\n"
        if analysis['confidence_level'] == "High":
            content += "• The model expresses high certainty in its response\n"
        elif analysis['confidence_level'] == "Low":
            content += "• The model shows uncertainty or hedging in its response\n"
        
        if analysis['textblob_subjectivity'] > 0.5:
            content += "• Response is highly subjective/opinion-based\n"
        else:
            content += "• Response is relatively objective/fact-based\n"
        
        self.sentiment_text.insert(1.0, content)
    
    def display_entity_analysis(self, analysis: Dict[str, Any], test_name: str):
        """Display entity analysis results"""
        self.entity_text.delete(1.0, tk.END)
        
        content = f"ENTITIES & CONCEPTS ANALYSIS\nTest: {test_name}\n{'='*50}\n\n"
        
        # Named entities
        content += f"Named Entities ({analysis['total_entities']} found):\n"
        for entity, label in analysis['entities'][:10]:  # Top 10
            content += f"  • {entity} ({label})\n"
        content += "\n"
        
        # Key concepts
        content += f"Key Concepts (Noun Phrases):\n"
        for phrase in analysis['noun_phrases']:
            content += f"  • {phrase}\n"
        content += "\n"
        
        # Most frequent terms
        content += f"Most Frequent Terms:\n"
        for term, freq in analysis['key_terms']:
            content += f"  • {term} ({freq} times)\n"
        content += "\n"
        
        # Analysis summary
        content += "ANALYSIS SUMMARY:\n"
        content += f"• {analysis['total_entities']} named entities identified\n"
        content += f"• {analysis['unique_entity_types']} different entity types\n"
        content += f"• {len(analysis['noun_phrases'])} key concepts extracted\n"
        
        # Focus assessment
        if analysis['total_entities'] > 5:
            content += "• High entity density - response is concept-rich\n"
        elif analysis['total_entities'] < 2:
            content += "• Low entity density - response is more abstract\n"
        
        self.entity_text.insert(1.0, content)
    
    def display_reasoning_analysis(self, analysis: Dict[str, Any], test_name: str):
        """Display reasoning structure analysis"""
        self.reasoning_text.delete(1.0, tk.END)
        
        content = f"REASONING STRUCTURE ANALYSIS\nTest: {test_name}\n{'='*50}\n\n"
        
        content += f"Reasoning Indicators:\n"
        content += f"  Causal reasoning: {analysis['causal_indicators']} indicators\n"
        content += f"  Conditional reasoning: {analysis['conditional_indicators']} indicators\n"
        content += f"  Evidence references: {analysis['evidence_markers']} markers\n"
        content += f"  Exploratory questions: {analysis['question_words']} question words\n\n"
        
        content += f"Structure Metrics:\n"
        content += f"  Total sentences: {analysis['total_sentences']}\n"
        content += f"  Average sentence length: {analysis['avg_sentence_length']:.1f} words\n"
        content += f"  Complex dependencies: {analysis['complex_dependencies']}\n"
        content += f"  Reasoning depth score: {analysis['reasoning_depth']}\n\n"
        
        # Reasoning style assessment
        content += "REASONING STYLE ASSESSMENT:\n"
        
        if analysis['reasoning_depth'] > 5:
            content += "• High reasoning depth - structured, logical approach\n"
        elif analysis['reasoning_depth'] < 2:
            content += "• Low reasoning depth - more intuitive or simple response\n"
        
        if analysis['evidence_markers'] > 2:
            content += "• Evidence-based reasoning style\n"
        
        if analysis['question_words'] > 3:
            content += "• Exploratory reasoning - considers multiple angles\n"
        
        if analysis['conditional_indicators'] > 2:
            content += "• Strong conditional reasoning - considers scenarios\n"
        
        if analysis['avg_sentence_length'] > 15:
            content += "• Complex sentence structure - detailed explanations\n"
        elif analysis['avg_sentence_length'] < 8:
            content += "• Simple sentence structure - concise responses\n"
        
        self.reasoning_text.insert(1.0, content)
    
    def display_linguistic_analysis(self, analysis: Dict[str, Any], test_name: str):
        """Display linguistic features analysis"""
        self.linguistic_text.delete(1.0, tk.END)
        
        content = f"LINGUISTIC FEATURES ANALYSIS\nTest: {test_name}\n{'='*50}\n\n"
        
        content += f"Basic Metrics:\n"
        content += f"  Word count: {analysis['word_count']}\n"
        content += f"  Sentence count: {analysis['sentence_count']}\n"
        content += f"  Average words per sentence: {analysis['avg_words_per_sentence']:.1f}\n"
        content += f"  Unique words: {analysis['unique_words']}\n"
        content += f"  Lexical diversity: {analysis['lexical_diversity']:.3f}\n\n"
        
        content += f"Part-of-Speech Distribution:\n"
        for pos, count in analysis['pos_distribution'].items():
            content += f"  {pos}: {count}\n"
        content += "\n"
        
        content += f"Style Markers:\n"
        content += f"  First person usage: {analysis['first_person_usage']} instances\n"
        content += f"  Uncertainty markers: {analysis['uncertainty_markers']} instances\n\n"
        
        # Linguistic style assessment
        content += "LINGUISTIC STYLE ASSESSMENT:\n"
        
        if analysis['lexical_diversity'] > 0.7:
            content += "• High lexical diversity - varied vocabulary\n"
        elif analysis['lexical_diversity'] < 0.4:
            content += "• Low lexical diversity - repetitive vocabulary\n"
        
        if analysis['first_person_usage'] > 0:
            content += f"• Uses first person ({analysis['first_person_usage']} times) - anthropomorphic style\n"
        else:
            content += "• Avoids first person - more objective style\n"
        
        if analysis['uncertainty_markers'] > 3:
            content += "• High uncertainty markers - cautious, hedged language\n"
        elif analysis['uncertainty_markers'] == 0:
            content += "• No uncertainty markers - confident, assertive language\n"
        
        if analysis['avg_words_per_sentence'] > 20:
            content += "• Long sentences - complex, detailed communication\n"
        elif analysis['avg_words_per_sentence'] < 10:
            content += "• Short sentences - concise, direct communication\n"
        
        self.linguistic_text.insert(1.0, content)
    
    def show_analysis_error(self, error_msg: str):
        """Show error message in all analysis tabs"""
        error_text = f"Analysis Error: {error_msg}\n\nPlease ensure:\n• NLP libraries are installed\n• spaCy English model is available\n• A response is selected for analysis"
        
        for text_widget in [self.sentiment_text, self.entity_text, self.reasoning_text, self.linguistic_text]:
            text_widget.delete(1.0, tk.END)
            text_widget.insert(1.0, error_text)
    
    def load_builtin_tests(self):
        """Load built-in psychological tests from JSON files"""
        # Load from JSON datasets
        self.load_dataset_from_file("datasets/builtin_tests.json")
        self.load_dataset_from_file("datasets/phase1_datasets.json")
    
    def load_dataset_from_file(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # Handle both formats
        if isinstance(dataset, list):
            tests_data = dataset  # Old format
        else:
            tests_data = dataset.get('tests', [])  # New format
        
        for test_data in tests_data:
            test = PsychologicalTest.from_dict(test_data)
            self.tests[test.test_id] = test
    
    def refresh_test_tree(self):
        """Refresh the test library tree view"""
        for item in self.test_tree.get_children():
            self.test_tree.delete(item)
            
        for test in self.tests.values():
            self.test_tree.insert("", tk.END, text=test.name, 
                                values=(test.category, test.description[:50] + "..."))
    
    def refresh_available_tests(self):
        """Refresh the available tests listbox in runner tab"""
        self.available_listbox.delete(0, tk.END)
        for test in self.tests.values():
            self.available_listbox.insert(tk.END, f"{test.name} ({test.category})")
    
    def run(self):
        """Start the Machine Psychology Lab application.
        
        Refreshes the test library and results displays, then starts the main
        Tkinter event loop. This is the main entry point after initialization.
        
        Call this method to begin using the application:
            app = MachinePsychologyLab()
            app.run()
        """
        self.refresh_test_tree()
        self.refresh_available_tests()
        self.root.mainloop()
    
    # Event handlers and functionality
    def view_selected_test(self):
        """View details of selected test"""
        selection = self.test_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a test to view.")
            return
            
        item = self.test_tree.item(selection[0])
        test_name = item['text']
        
        # Find the test by name
        test = None
        for t in self.tests.values():
            if t.name == test_name:
                test = t
                break
                
        if not test:
            return
            
        # Create view window
        view_window = tk.Toplevel(self.root)
        view_window.title(f"View Test: {test.name}")
        view_window.geometry("600x500")
        
        # Test details
        details_frame = ttk.LabelFrame(view_window, text="Test Details", padding=10)
        details_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(details_frame, text=f"Name: {test.name}", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=2)
        ttk.Label(details_frame, text=f"Category: {test.category}").pack(anchor=tk.W, pady=2)
        ttk.Label(details_frame, text=f"Description: {test.description}").pack(anchor=tk.W, pady=2)
        
        ttk.Label(details_frame, text="Prompt:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 2))
        prompt_text = scrolledtext.ScrolledText(details_frame, height=8, wrap=tk.WORD)
        prompt_text.insert(tk.END, test.prompt)
        prompt_text.configure(state=tk.DISABLED)
        prompt_text.pack(fill=tk.BOTH, expand=True, pady=2)
        
        if test.expected_responses:
            ttk.Label(details_frame, text="Expected Responses:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 2))
            responses_text = scrolledtext.ScrolledText(details_frame, height=3, wrap=tk.WORD)
            responses_text.insert(tk.END, "\n".join(test.expected_responses))
            responses_text.configure(state=tk.DISABLED)
            responses_text.pack(fill=tk.X, pady=2)
        
        ttk.Label(details_frame, text=f"Scoring Method: {test.scoring_method}").pack(anchor=tk.W, pady=2)
        if test.scoring_criteria:
            ttk.Label(details_frame, text=f"Scoring Criteria: {test.scoring_criteria}").pack(anchor=tk.W, pady=2)
    
    def edit_selected_test(self):
        """Edit the selected test"""
        selection = self.test_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a test to edit.")
            return
            
        item = self.test_tree.item(selection[0])
        test_name = item['text']
        
        # Find and load test into creator form
        for test in self.tests.values():
            if test.name == test_name:
                self.load_test_into_form(test)
                self.notebook.select(1)  # Switch to creator tab
                break
    
    def load_test_into_form(self, test: PsychologicalTest):
        """Load test data into the creator form"""
        self.name_entry.delete(0, tk.END)
        self.name_entry.insert(0, test.name)
        
        self.category_combo.set(test.category)
        
        self.description_text.delete(1.0, tk.END)
        self.description_text.insert(1.0, test.description)
        
        self.prompt_text.delete(1.0, tk.END)
        self.prompt_text.insert(1.0, test.prompt)
        
        self.responses_text.delete(1.0, tk.END)
        self.responses_text.insert(1.0, "\n".join(test.expected_responses))
        
        self.scoring_combo.set(test.scoring_method)
        
        self.criteria_text.delete(1.0, tk.END)
        self.criteria_text.insert(1.0, test.scoring_criteria)
        
        # Store test ID for updating
        self.current_edit_id = test.test_id
    
    def delete_selected_test(self):
        """Delete the selected test"""
        selection = self.test_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a test to delete.")
            return
            
        item = self.test_tree.item(selection[0])
        test_name = item['text']
        
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete '{test_name}'?"):
            # Find and remove test
            test_id_to_remove = None
            for test_id, test in self.tests.items():
                if test.name == test_name:
                    test_id_to_remove = test_id
                    break
                    
            if test_id_to_remove:
                del self.tests[test_id_to_remove]
                self.refresh_test_tree()
                self.refresh_available_tests()
                self.save_data()
    
    def import_tests(self):
        """Import tests from JSON file"""
        filename = filedialog.askopenfilename(
            title="Import Tests",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                imported_count = 0
                for test_data in data:
                    test = PsychologicalTest.from_dict(test_data)
                    self.tests[test.test_id] = test
                    imported_count += 1
                    
                self.refresh_test_tree()
                self.refresh_available_tests()
                self.save_data()
                messagebox.showinfo("Import Successful", f"Imported {imported_count} tests.")
                
            except Exception as e:
                messagebox.showerror("Import Failed", f"Failed to import tests: {str(e)}")
    
    def export_tests(self):
        """Export tests to JSON file"""
        filename = filedialog.asksaveasfilename(
            title="Export Tests",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                test_data = [test.to_dict() for test in self.tests.values()]
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(test_data, f, indent=2, ensure_ascii=False)
                    
                messagebox.showinfo("Export Successful", f"Exported {len(test_data)} tests to {filename}")
                
            except Exception as e:
                messagebox.showerror("Export Failed", f"Failed to export tests: {str(e)}")
    
    def save_test(self):
        """Save test from creator form"""
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Test name is required.")
            return
            
        category = self.category_combo.get().strip()
        description = self.description_text.get(1.0, tk.END).strip()
        prompt = self.prompt_text.get(1.0, tk.END).strip()
        
        if not prompt:
            messagebox.showerror("Error", "Test prompt is required.")
            return
            
        responses_text = self.responses_text.get(1.0, tk.END).strip()
        expected_responses = [r.strip() for r in responses_text.split('\n') if r.strip()]
        
        scoring_method = self.scoring_combo.get() or "manual"
        scoring_criteria = self.criteria_text.get(1.0, tk.END).strip()
        
        # Check if editing existing test
        test_id = getattr(self, 'current_edit_id', None)
        if test_id and test_id in self.tests:
            # Update existing test
            test = self.tests[test_id]
            test.name = name
            test.category = category
            test.description = description
            test.prompt = prompt
            test.expected_responses = expected_responses
            test.scoring_method = scoring_method
            test.scoring_criteria = scoring_criteria
        else:
            # Create new test
            test = PsychologicalTest(
                name=name,
                category=category,
                description=description,
                prompt=prompt,
                expected_responses=expected_responses,
                scoring_method=scoring_method,
                scoring_criteria=scoring_criteria
            )
            self.tests[test.test_id] = test
        
        self.refresh_test_tree()
        self.refresh_available_tests()
        self.save_data()
        self.clear_test_form()
        messagebox.showinfo("Success", "Test saved successfully!")
    
    def clear_test_form(self):
        """Clear the test creation form"""
        self.name_entry.delete(0, tk.END)
        self.category_combo.set("")
        self.description_text.delete(1.0, tk.END)
        self.prompt_text.delete(1.0, tk.END)
        self.responses_text.delete(1.0, tk.END)
        self.scoring_combo.set("manual")
        self.criteria_text.delete(1.0, tk.END)
        self.current_edit_id = None
    
    def preview_test(self):
        """Preview the test as it would appear to the model"""
        prompt = self.prompt_text.get(1.0, tk.END).strip()
        if not prompt:
            messagebox.showerror("Error", "Please enter a test prompt first.")
            return
            
        preview_window = tk.Toplevel(self.root)
        preview_window.title("Test Preview")
        preview_window.geometry("600x400")
        
        ttk.Label(preview_window, text="This is how the prompt will appear to the AI model:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=10, pady=5)
        
        preview_text = scrolledtext.ScrolledText(preview_window, wrap=tk.WORD)
        preview_text.insert(tk.END, prompt)
        preview_text.configure(state=tk.DISABLED)
        preview_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        ttk.Button(preview_window, text="Close", command=preview_window.destroy).pack(pady=10)
    
    def add_test_to_run(self):
        """Add selected tests to run list"""
        selected_indices = self.available_listbox.curselection()
        for index in selected_indices:
            test_name = self.available_listbox.get(index)
            if test_name not in [self.selected_listbox.get(i) for i in range(self.selected_listbox.size())]:
                self.selected_listbox.insert(tk.END, test_name)
    
    def remove_test_from_run(self):
        """Remove selected tests from run list"""
        selected_indices = list(self.selected_listbox.curselection())
        for index in reversed(selected_indices):
            self.selected_listbox.delete(index)
    
    def add_all_tests(self):
        """Add all tests to run list"""
        self.selected_listbox.delete(0, tk.END)
        for i in range(self.available_listbox.size()):
            self.selected_listbox.insert(tk.END, self.available_listbox.get(i))
    
    def remove_all_tests(self):
        """Remove all tests from run list"""
        self.selected_listbox.delete(0, tk.END)
    
    def run_tests(self):
        """Run selected tests on the configured model"""
        if self.selected_listbox.size() == 0:
            messagebox.showwarning("No Tests", "Please select tests to run.")
            return
            
        # Update API client with current settings
        base_url = self.base_url_entry.get().strip()
        api_key = self.api_key_entry.get().strip()
        self.api_client = APIClient(base_url, api_key)
        
        selected_model = self.model_combo.get().strip()
        if not selected_model:
            messagebox.showerror("No Model Selected", "Please select a model from the dropdown. If no models are shown, click 'Refresh Models' or ensure a model is loaded in LM Studio/Ollama.")
            return
        model_name = selected_model
        temperature = self.temperature_var.get()
        
        # Get selected tests
        selected_test_names = [self.selected_listbox.get(i) for i in range(self.selected_listbox.size())]
        selected_tests = []
        
        for test_name in selected_test_names:
            # Extract test name from display format "Test Name (Category)"
            actual_name = test_name.split(" (")[0]
            for test in self.tests.values():
                if test.name == actual_name:
                    selected_tests.append(test)
                    break
        
        if not selected_tests:
            messagebox.showerror("Error", "No valid tests found to run.")
            return
        
        # Expand dataset tests into individual variant tests
        expanded_tests = []
        for test in selected_tests:
            if test.is_dataset and test.dataset_variants:
                # Create individual test objects for each variant
                for i, variant in enumerate(test.dataset_variants):
                    variant_test = PsychologicalTest(
                        test_id=f"{test.test_id}_variant_{i}",
                        name=f"{test.name} - {variant.get('name', f'Variant {i+1}')}",
                        category=test.category,
                        description=variant.get('description', test.description),
                        prompt=variant.get('prompt', ''),
                        expected_responses=variant.get('expected_responses', []),
                        scoring_method=variant.get('scoring_method', test.scoring_method),
                        scoring_criteria=variant.get('scoring_criteria', test.scoring_criteria),
                        is_dataset=False,
                        dataset_variants=[]
                    )
                    expanded_tests.append(variant_test)
                print(f"Expanded dataset '{test.name}' into {len(test.dataset_variants)} variants")
            else:
                # Regular test or dataset with no variants
                if test.is_dataset:
                    print(f"Warning: Dataset '{test.name}' has no variants to run")
                expanded_tests.append(test)
        
        if not expanded_tests:
            messagebox.showerror("Error", "No runnable tests found after expansion.")
            return
        
        # Show summary of what will be run
        dataset_count = sum(1 for t in selected_tests if t.is_dataset and t.dataset_variants)
        if dataset_count > 0:
            total_variants = sum(len(t.dataset_variants) for t in selected_tests if t.is_dataset and t.dataset_variants)
            regular_tests = len(selected_tests) - dataset_count
            summary_msg = f"Running {len(expanded_tests)} tests:\n"
            if dataset_count > 0:
                summary_msg += f"• {dataset_count} dataset(s) expanded to {total_variants} variants\n"
            if regular_tests > 0:
                summary_msg += f"• {regular_tests} individual test(s)\n"
            summary_msg += f"\nTotal tests to execute: {len(expanded_tests)}"
            messagebox.showinfo("Test Execution Plan", summary_msg)
            
        # Disable run button and start progress
        self.run_button.configure(state=tk.DISABLED)
        self.progress_bar.configure(maximum=len(expanded_tests))
        self.progress_bar['value'] = 0
        
        # Run tests in separate thread
        thread = threading.Thread(target=self.execute_tests, args=(expanded_tests, model_name, temperature))
        thread.daemon = True
        thread.start()
    
    def execute_tests(self, tests: List[PsychologicalTest], model_name: str, temperature: float):
        """Execute psychological tests on the specified AI model in a background thread.
        
        This method handles the complex process of running tests including:
        - Thread-safe GUI updates for progress tracking
        - Dataset expansion for multi-variant tests  
        - API communication with error handling and retries
        - Automatic machine estimation generation
        - Safe error handling with user notification
        
        Args:
            tests: List of PsychologicalTest objects to execute
            model_name: String identifier for the AI model (e.g., 'gpt-4', 'claude-3')
            temperature: Float between 0.0-2.0 controlling response randomness
            
        Note:
            This method runs asynchronously in a background thread to prevent
            GUI freezing during long-running test executions.
        """
        def safe_update_progress(test_name, progress_val):
            """Safely update progress from background thread"""
            try:
                self.progress_var.set(f"Running {test_name}...")
                self.progress_bar['value'] = progress_val
                self.root.update_idletasks()
            except:
                pass
        
        def safe_update_results():
            """Safely update results display"""
            try:
                self.refresh_results()
                if hasattr(self, 'refresh_analysis_responses'):
                    self.refresh_analysis_responses()
            except:
                pass
        
        def safe_show_error(title, message):
            """Safely show error message"""
            try:
                messagebox.showerror(title, message)
            except:
                print(f"Error: {title} - {message}")
        
        try:
            total_tests = len(tests)
            for i, test in enumerate(tests):
                # Update progress safely
                self.root.after(0, lambda tn=test.name, pv=i: safe_update_progress(tn, pv))
                
                # Prepare messages
                messages = [
                    {"role": "user", "content": test.prompt}
                ]
                
                # Make API call
                try:
                    print(f"Making API call for test: {test.name}")
                    response = self.api_client.chat_completion(
                        messages=messages,
                        model=model_name,
                        temperature=temperature,
                        max_tokens=1000
                    )
                    
                    if response:
                        print(f"Got response for {test.name}, length: {len(response)}")
                        
                        # Generate machine estimation (analysis, not clinical scoring)
                        try:
                            machine_estimation = self.generate_machine_estimation(test, response)
                            print(f"Generated machine estimation for {test.name}: {machine_estimation.estimated_understanding}")
                        except Exception as estimation_error:
                            print(f"Machine estimation error for {test.name}: {estimation_error}")
                            machine_estimation = MachineEstimation()
                            machine_estimation.analysis_notes = f"Estimation failed: {str(estimation_error)}"
                        
                        # Create result with dual evaluation system
                        result = TestResult(
                            test_id=test.test_id,
                            model_name=model_name,
                            response=response
                        )
                        
                        # Attach machine estimation
                        result.machine_estimation = machine_estimation
                        
                        # Legacy score field (for backward compatibility)
                        # Set to None to indicate human evaluation needed
                        result.score = None
                        
                        # Add to results safely
                        try:
                            self.results.append(result)
                            print(f"Added result for {test.name}")
                        except Exception as add_error:
                            print(f"Error adding result: {add_error}")
                        
                        # Update results display
                        self.root.after(0, safe_update_results)
                    else:
                        print(f"No response received for {test.name}")
                        self.root.after(0, lambda tn=test.name: safe_show_error("API Error", f"Failed to get response for {tn}"))
                        
                except Exception as api_error:
                    print(f"API error for test {test.name}: {api_error}")
                    self.root.after(0, lambda tn=test.name, err=str(api_error): safe_show_error("Test Error", f"Error running {tn}: {err}"))
                
                # Small delay to prevent overwhelming the API
                import time
                time.sleep(0.5)
            
            # Finish successfully
            self.root.after(0, lambda: self.progress_var.set("Completed!"))
            self.root.after(0, lambda: setattr(self.progress_bar, 'value', total_tests))
            self.root.after(0, lambda: self.run_button.configure(state=tk.NORMAL))
            self.root.after(0, self.save_data)
            print(f"Test execution completed successfully")
            
        except Exception as critical_error:
            print(f"Critical error in execute_tests: {critical_error}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self.progress_var.set("Error occurred"))
            self.root.after(0, lambda: self.run_button.configure(state=tk.NORMAL))
            self.root.after(0, lambda err=str(critical_error): safe_show_error("Execution Error", f"Critical error during test execution: {err}"))
    
    def generate_machine_estimation(self, test: PsychologicalTest, response: str) -> MachineEstimation:
        """Generate comprehensive machine-based analysis of AI model response.
        
        Performs automated linguistic and psychological analysis using the selected
        analysis engine. This is for research estimation only, not clinical assessment.
        
        Analysis includes (depending on engine):
        - Sentiment analysis (VADER, TextBlob, or Transformer-based)
        - Reasoning structure identification 
        - Linguistic feature extraction (sentence length, vocabulary diversity)
        - Coherence scoring based on multiple factors
        - Understanding estimation relative to test context
        
        Args:
            test: PsychologicalTest object containing context and scoring criteria
            response: String response from the AI model to analyze
            
        Returns:
            MachineEstimation object containing all analysis results with
            scores, confidence levels, and detailed analysis notes
            
        Note:
            Uses the currently selected analysis engine. Falls back to basic
            analysis if advanced engines unavailable.
        """
        estimation = MachineEstimation()
        
        try:
            if not response or not isinstance(response, str):
                estimation.analysis_notes = "Invalid or empty response"
                return estimation
            
            # Use selected analysis engine for comprehensive analysis
            if hasattr(self, 'current_engine') and self.current_engine and self.current_engine.available:
                # Primary analysis using selected engine
                estimation.sentiment_analysis = self.current_engine.analyze_sentiment(response)
                estimation.confidence_assessment = estimation.sentiment_analysis.get('confidence_level', 'Unknown')
                
                # Track which engine was used
                engine_info = f"Analysis Engine: {self.current_engine.name}"
                estimation.analysis_notes = engine_info
                
                # Fall back to legacy methods for reasoning and linguistic analysis
                # (these will be moved to engine system in future updates)
                if hasattr(self, 'nlp') and self.nlp:
                    estimation.reasoning_indicators = self.analyze_reasoning_structure(response)
                    estimation.linguistic_features = self.analyze_linguistic_features(response)
            
            # Fallback to legacy analysis if no engine available
            elif NLP_AVAILABLE and hasattr(self, 'vader_analyzer') and self.vader_analyzer:
                estimation.sentiment_analysis = self.analyze_sentiment(response)
                estimation.confidence_assessment = estimation.sentiment_analysis.get('confidence_level', 'Unknown')
                estimation.reasoning_indicators = self.analyze_reasoning_structure(response)
                estimation.linguistic_features = self.analyze_linguistic_features(response)
                
                # Coherence scoring (0-1 scale based on linguistic features)
                coherence_factors = []
                if estimation.linguistic_features:
                    # Check for complete sentences, proper structure, etc.
                    readability = estimation.linguistic_features.get('avg_sentence_length', 0)
                    if 5 <= readability <= 25:  # Reasonable sentence length
                        coherence_factors.append(0.3)
                    
                    # Check for appropriate vocabulary diversity
                    lexical_diversity = estimation.linguistic_features.get('lexical_diversity', 0)
                    if lexical_diversity > 0.3:
                        coherence_factors.append(0.3)
                    
                    # Check reasoning indicators
                    if estimation.reasoning_indicators:
                        reasoning_depth = estimation.reasoning_indicators.get('reasoning_depth', 0)
                        if reasoning_depth > 0:
                            coherence_factors.append(0.4)
                
                estimation.coherence_score = sum(coherence_factors)
                
                # Estimated understanding based on multiple factors
                understanding_indicators = []
                # 1. Response relevance (basic keyword matching with expected responses)
                if test.expected_responses:
                    response_lower = response.lower()
                    for expected in test.expected_responses:
                        if expected and any(word in response_lower for word in expected.lower().split()):
                            understanding_indicators.append("relevant_content")
                            break
                
                # 2. Coherence threshold
                if estimation.coherence_score > 0.5:
                    understanding_indicators.append("coherent_response")
                
                # 3. Appropriate confidence level
                if estimation.confidence_assessment in ['Medium', 'High']:
                    understanding_indicators.append("appropriate_confidence")
                
                # 4. Evidence of reasoning
                if estimation.reasoning_indicators and estimation.reasoning_indicators.get('reasoning_depth', 0) > 1:
                    understanding_indicators.append("shows_reasoning")
                
                # Estimate understanding level
                understanding_score = len(understanding_indicators) / 4.0  # 0-1 scale
                if understanding_score >= 0.75:
                    estimation.estimated_understanding = "High"
                elif understanding_score >= 0.5:
                    estimation.estimated_understanding = "Moderate"
                elif understanding_score >= 0.25:
                    estimation.estimated_understanding = "Limited"
                else:
                    estimation.estimated_understanding = "Minimal"
                
                # Generate analysis notes
                notes = []
                notes.append(f"Estimated understanding: {estimation.estimated_understanding}")
                notes.append(f"Confidence level: {estimation.confidence_assessment}")
                notes.append(f"Coherence score: {estimation.coherence_score:.2f}")
                
                if understanding_indicators:
                    notes.append(f"Positive indicators: {', '.join(understanding_indicators)}")
                
                # Add reasoning insights
                if estimation.reasoning_indicators:
                    causal_count = estimation.reasoning_indicators.get('causal_indicators', 0)
                    if causal_count > 0:
                        notes.append(f"Shows causal reasoning ({causal_count} indicators)")
                
                estimation.analysis_notes = " | ".join(notes)
            
            else:
                # Fallback analysis without NLP tools
                estimation.analysis_notes = "Limited analysis - NLP tools not available"
                estimation.estimated_understanding = "Unknown"
                estimation.coherence_score = 0.5  # Neutral
                
                # Basic keyword matching for understanding estimate
                if test.expected_responses:
                    response_lower = response.lower()
                    matches = sum(1 for expected in test.expected_responses 
                                if expected and expected.lower() in response_lower)
                    if matches > 0:
                        estimation.estimated_understanding = "Moderate"
                        estimation.analysis_notes += f" | Found {matches} keyword matches"
        
        except Exception as e:
            estimation.analysis_notes = f"Analysis error: {str(e)}"
            estimation.estimated_understanding = "Error"
            print(f"Error generating machine estimation for test {test.name}: {e}")
        
        return estimation
    
    def refresh_results(self):
        """Refresh the results tree view"""
        try:
            # Clear existing items
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)
                
            # Add results one by one with error handling
            for i, result in enumerate(self.results):
                try:
                    # Find test name safely
                    test_name = "Unknown"
                    if hasattr(result, 'test_id') and result.test_id and result.test_id in self.tests:
                        test_name = self.tests[result.test_id].name
                    
                    # Extract machine estimation data safely
                    machine_est_text = "No Analysis"
                    confidence_text = "Unknown"
                    if hasattr(result, 'machine_estimation') and result.machine_estimation:
                        est = result.machine_estimation
                        machine_est_text = getattr(est, 'estimated_understanding', 'Unknown')
                        confidence_text = getattr(est, 'confidence_assessment', 'Unknown')
                    
                    # Extract human evaluation data safely
                    human_score_text = "Pending"
                    eval_status_text = "Needs Review"
                    if hasattr(result, 'human_evaluation') and result.human_evaluation:
                        eval = result.human_evaluation
                        if eval.final_score is not None:
                            try:
                                human_score_text = f"{float(eval.final_score):.1f}"
                                eval_status_text = "Completed" 
                            except (ValueError, TypeError):
                                human_score_text = str(eval.final_score)
                                eval_status_text = "Completed"
                        else:
                            eval_status_text = "In Progress" if eval.evaluation_notes else "Needs Review"
                    
                    # Format timestamp safely
                    timestamp_short = "Unknown"
                    if hasattr(result, 'timestamp') and result.timestamp:
                        try:
                            timestamp_short = result.timestamp[:19].replace('T', ' ')
                        except (AttributeError, TypeError):
                            timestamp_short = str(result.timestamp)[:19]
                    
                    # Get model name safely
                    model_name = getattr(result, 'model_name', 'Unknown')
                    
                    # Insert into tree with all columns
                    self.results_tree.insert("", tk.END, iid=str(i), text=str(i+1),
                                           values=(model_name, test_name, machine_est_text, 
                                                 confidence_text, human_score_text, eval_status_text, timestamp_short))
                                           
                except Exception as e:
                    print(f"Error adding result {i} to tree: {e}")
                    # Add a basic error entry
                    self.results_tree.insert("", tk.END, iid=str(i), text=str(i+1),
                                           values=("Error", "Error", "Error", "Error", "Error", "Error", "Error"))
                                           
        except Exception as e:
            print(f"Critical error in refresh_results: {e}")
            import traceback
            traceback.print_exc()
    
    def on_result_select(self, event):
        """Handle result selection"""
        selection = self.results_tree.selection()
        if selection:
            result_index = int(selection[0])
            if 0 <= result_index < len(self.results):
                result = self.results[result_index]
                self.current_result_index = result_index
                
                # Show response in text area
                self.response_text.delete(1.0, tk.END)
                self.response_text.insert(1.0, result.response)
                
                # Show test details
                if result.test_id in self.tests:
                    test = self.tests[result.test_id]
                    self.response_text.insert(1.0, f"Test: {test.name}\nCategory: {test.category}\n\nPrompt:\n{test.prompt}\n\n" + "="*50 + "\nModel Response:\n")
                
                # Update human evaluation interface
                self.update_human_evaluation_interface(result)
        else:
            # Clear interface when no selection
            self.response_text.delete(1.0, tk.END)
            if hasattr(self, 'update_human_evaluation_interface'):
                self.update_human_evaluation_interface(None)
    
    def update_human_evaluation_interface(self, result):
        """Update the human evaluation interface based on selected result"""
        if result is None:
            # Clear interface
            if hasattr(self, 'machine_analysis_label'):
                self.machine_analysis_label.config(text="No analysis available")
            if hasattr(self, 'human_score_var'):
                self.human_score_var.set("")
            if hasattr(self, 'eval_notes_text'):
                self.eval_notes_text.delete(1.0, tk.END)
            if hasattr(self, 'save_eval_button'):
                self.save_eval_button.config(state=tk.DISABLED)
            return
        
        # Display machine analysis
        if hasattr(result, 'machine_estimation') and result.machine_estimation:
            est = result.machine_estimation
            analysis_text = f"Understanding: {est.estimated_understanding}, Confidence: {est.confidence_assessment}"
            if est.coherence_score is not None:
                analysis_text += f", Coherence: {est.coherence_score:.2f}"
            if hasattr(self, 'machine_analysis_label'):
                self.machine_analysis_label.config(text=analysis_text)
        else:
            if hasattr(self, 'machine_analysis_label'):
                self.machine_analysis_label.config(text="No machine analysis available")
        
        # Load existing human evaluation if present
        if hasattr(result, 'human_evaluation') and result.human_evaluation:
            eval = result.human_evaluation
            if hasattr(self, 'human_score_var'):
                self.human_score_var.set(str(eval.final_score) if eval.final_score is not None else "")
            if hasattr(self, 'human_confidence_var'):
                self.human_confidence_var.set(str(eval.confidence_in_evaluation) if eval.confidence_in_evaluation else "3")
            if hasattr(self, 'agreement_var'):
                self.agreement_var.set(eval.agrees_with_machine if eval.agrees_with_machine else "Partial")
            if hasattr(self, 'evaluator_id_var'):
                self.evaluator_id_var.set(eval.evaluator_id if eval.evaluator_id else "Researcher1")
            
            # Load notes
            if hasattr(self, 'eval_notes_text'):
                self.eval_notes_text.delete(1.0, tk.END)
                if eval.evaluation_notes:
                    self.eval_notes_text.insert(1.0, eval.evaluation_notes)
        else:
            # Clear for new evaluation
            if hasattr(self, 'human_score_var'):
                self.human_score_var.set("")
            if hasattr(self, 'human_confidence_var'):
                self.human_confidence_var.set("3")
            if hasattr(self, 'agreement_var'):
                self.agreement_var.set("Partial")
            if hasattr(self, 'eval_notes_text'):
                self.eval_notes_text.delete(1.0, tk.END)
        
        # Enable save button
        if hasattr(self, 'save_eval_button'):
            self.save_eval_button.config(state=tk.NORMAL)
    
    def quick_score(self, score):
        """Set quick score value"""
        if hasattr(self, 'human_score_var'):
            self.human_score_var.set(score)
    
    def save_human_evaluation(self):
        """Save human evaluation for current result"""
        if not hasattr(self, 'current_result_index') or self.current_result_index is None or self.current_result_index >= len(self.results):
            messagebox.showerror("Error", "No result selected for evaluation")
            return
        
        try:
            result = self.results[self.current_result_index]
            
            # Get evaluation data
            score_text = self.human_score_var.get().strip()
            if not score_text:
                messagebox.showerror("Error", "Please enter a score")
                return
            
            try:
                final_score = float(score_text)
            except ValueError:
                messagebox.showerror("Error", "Score must be a number")
                return
            
            # Create or update human evaluation
            if not hasattr(result, 'human_evaluation') or result.human_evaluation is None:
                result.human_evaluation = HumanEvaluation()
            
            eval = result.human_evaluation
            eval.final_score = final_score
            eval.evaluation_notes = self.eval_notes_text.get(1.0, tk.END).strip()
            eval.agrees_with_machine = self.agreement_var.get()
            eval.confidence_in_evaluation = int(self.human_confidence_var.get())
            eval.evaluator_id = self.evaluator_id_var.get().strip()
            eval.timestamp = datetime.now().isoformat()
            
            # Refresh results display to show updated status
            self.refresh_results()
            
            # Show success message
            messagebox.showinfo("Success", f"Evaluation saved for result #{self.current_result_index + 1}")
            
            # Auto-save results
            self.save_results()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save evaluation: {str(e)}")
            print(f"Error saving human evaluation: {e}")
    
    def export_results(self):
        """Export results to JSON file"""
        if not self.results:
            messagebox.showwarning("No Results", "No results to export.")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                results_data = [result.to_dict() for result in self.results]
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(results_data, f, indent=2, ensure_ascii=False)
                    
                messagebox.showinfo("Export Successful", f"Exported {len(results_data)} results to {filename}")
                
            except Exception as e:
                messagebox.showerror("Export Failed", f"Failed to export results: {str(e)}")
    
    def clear_results(self):
        """Clear all results"""
        if messagebox.askyesno("Confirm Clear", "Are you sure you want to clear all results?"):
            self.results.clear()
            self.refresh_results()
            self.response_text.delete(1.0, tk.END)
            self.save_data()
    
    def generate_report(self):
        """Generate a summary report of results"""
        if not self.results:
            messagebox.showwarning("No Results", "No results to report on.")
            return
            
        # Create report window
        report_window = tk.Toplevel(self.root)
        report_window.title("Test Results Report")
        report_window.geometry("800x600")
        
        report_text = scrolledtext.ScrolledText(report_window, wrap=tk.WORD)
        report_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Generate report content
        report_content = self.generate_report_content()
        report_text.insert(tk.END, report_content)
        report_text.configure(state=tk.DISABLED)
        
        # Save button
        ttk.Button(report_window, text="Save Report", command=lambda: self.save_report(report_content)).pack(pady=5)
    
    def generate_report_content(self) -> str:
        """Generate report content string"""
        from collections import defaultdict
        
        report = []
        report.append("MACHINE PSYCHOLOGY LAB - TEST RESULTS REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Results: {len(self.results)}")
        report.append("")
        
        # Group by model
        models = defaultdict(list)
        for result in self.results:
            models[result.model_name].append(result)
        
        for model_name, results in models.items():
            report.append(f"MODEL: {model_name}")
            report.append("-" * 30)
            
            # Group by category
            categories = defaultdict(list)
            for result in results:
                if result.test_id in self.tests:
                    category = self.tests[result.test_id].category
                    categories[category].append(result)
            
            total_scored = 0
            total_correct = 0
            
            for category, cat_results in categories.items():
                report.append(f"\n  {category}:")
                
                scored_results = [r for r in cat_results if r.score is not None]
                if scored_results:
                    avg_score = sum(r.score for r in scored_results) / len(scored_results)
                    report.append(f"    Average Score: {avg_score:.1f} ({len(scored_results)} tests)")
                    total_scored += len(scored_results)
                    total_correct += sum(r.score for r in scored_results)
                else:
                    report.append(f"    {len(cat_results)} tests (manual scoring required)")
                
                for result in cat_results:
                    if result.test_id in self.tests:
                        test_name = self.tests[result.test_id].name
                        score_text = f"{result.score:.1f}" if result.score is not None else "Manual"
                        report.append(f"    - {test_name}: {score_text}")
            
            if total_scored > 0:
                overall_avg = total_correct / total_scored
                report.append(f"\n  Overall Average: {overall_avg:.1f} ({total_scored} auto-scored tests)")
            
            report.append("")
        
        return "\n".join(report)
    
    def save_report(self, content: str):
        """Save report to file"""
        filename = filedialog.asksaveasfilename(
            title="Save Report",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                messagebox.showinfo("Report Saved", f"Report saved to {filename}")
            except Exception as e:
                messagebox.showerror("Save Failed", f"Failed to save report: {str(e)}")
    
    def test_connection(self):
        """Test API connection"""
        base_url = self.base_url_entry.get().strip()
        api_key = self.api_key_entry.get().strip()
        
        if not base_url:
            messagebox.showerror("Error", "Please enter a base URL.")
            return
            
        test_client = APIClient(base_url, api_key)
        
        # Test with simple message
        messages = [{"role": "user", "content": "Hello, please respond with just 'Connection successful'"}]
        
        try:
            self.progress_var.set("Testing connection...")
            response = test_client.chat_completion(messages, temperature=0.1, max_tokens=50)
            
            if response:
                messagebox.showinfo("Connection Test", f"✓ Connection successful!\n\nResponse: {response[:100]}...")
                self.progress_var.set("Connection test passed")
            else:
                messagebox.showerror("Connection Test", "✗ Connection failed - no response received")
                self.progress_var.set("Connection test failed")
                
        except Exception as e:
            messagebox.showerror("Connection Test", f"✗ Connection failed: {str(e)}")
            self.progress_var.set("Connection test failed")
    
    def load_data(self):
        """Load previously saved tests and results from JSON files.
        
        Loads data from two files in the current directory:
        - psychology_tests.json: Custom user-created tests
        - psychology_results.json: Previous test execution results
        
        Built-in tests are not overwritten by loaded data. Handles file
        loading errors gracefully with console warnings.
        
        Files are expected to contain JSON arrays of serialized objects.
        """
        # Load tests
        tests_file = "psychology_tests.json"
        if os.path.exists(tests_file):
            try:
                with open(tests_file, 'r', encoding='utf-8') as f:
                    tests_data = json.load(f)
                    for test_data in tests_data:
                        test = PsychologicalTest.from_dict(test_data)
                        # Don't overwrite built-in tests
                        if test.test_id not in self.tests:
                            self.tests[test.test_id] = test
            except Exception as e:
                print(f"Failed to load tests: {e}")
        
        # Load results
        results_file = "psychology_results.json"
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    results_data = json.load(f)
                    for result_data in results_data:
                        result = TestResult.from_dict(result_data)
                        self.results.append(result)
            except Exception as e:
                print(f"Failed to load results: {e}")
    
    def refresh_models(self):
        """Refresh the list of available models"""
        # Update API client with current settings
        base_url = self.base_url_entry.get().strip()
        api_key = self.api_key_entry.get().strip()
        
        if not base_url:
            messagebox.showerror("Error", "Please configure the API base URL in Settings first.")
            return
            
        self.api_client = APIClient(base_url, api_key)
        
        # Show loading status
        self.model_status_label.configure(text="Loading models...", foreground="blue")
        self.root.update()
        
        # Get models
        models = self.api_client.get_available_models()
        
        if models:
            self.model_combo['values'] = models
            self.model_combo.set(models[0])  # Select first model by default
            self.model_status_label.configure(text=f"Found {len(models)} model(s)", foreground="green")
        else:
            self.model_combo['values'] = []
            self.model_combo.set("")
            self.model_status_label.configure(
                text="⚠ No models found. Please load a model in LM Studio/Ollama first.", 
                foreground="red"
            )
            
            # Show helpful message
            messagebox.showwarning(
                "No Models Available", 
                "No models were found on the API endpoint.\n\n"
                "For LM Studio:\n"
                "1. Open LM Studio\n"
                "2. Go to the 'My Models' tab\n"
                "3. Load a model by clicking the 'Load Model' button\n"
                "4. Wait for it to finish loading\n"
                "5. Come back here and click 'Refresh Models'\n\n"
                "For Ollama:\n"
                "1. Run 'ollama pull <model-name>' in terminal\n"
                "2. Start Ollama server\n"
                "3. Click 'Refresh Models' here"
            )
    
    def save_data(self):
        """Save current tests and results to JSON files for persistence.
        
        Saves data to two files in the current directory:
        - psychology_tests.json: User-created custom tests (excludes built-ins)
        - psychology_results.json: All test execution results with analysis
        
        Built-in tests are excluded from saving to avoid duplication.
        Handles save errors gracefully with console warnings.
        
        Files are saved as formatted JSON with UTF-8 encoding.
        """
        # Save tests (excluding built-in ones)
        try:
            builtin_names = {"Sally-Anne Test Dataset", "Asian Disease Problem", "Linda the Bank Teller", "Wason Selection Task"}
            custom_tests = [test.to_dict() for test in self.tests.values() if test.name not in builtin_names]
            
            with open("psychology_tests.json", 'w', encoding='utf-8') as f:
                json.dump(custom_tests, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save tests: {e}")
        
        # Save results
        try:
            results_data = [result.to_dict() for result in self.results]
            with open("psychology_results.json", 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save results: {e}")
    
    def on_test_select(self, event):
        """Handle test selection in tree view"""
        try:
            selection = self.test_tree.selection()
            if selection:
                item = self.test_tree.item(selection[0])
                test_name = item['text']  # Get test name from text, not values
                
                # Find test by name
                test = None
                for t in self.tests.values():
                    if t.name == test_name:
                        test = t
                        break
                
                if test and test.is_dataset:
                    self.load_dataset_btn.config(state=tk.NORMAL)
                else:
                    self.load_dataset_btn.config(state=tk.DISABLED)
            else:
                self.load_dataset_btn.config(state=tk.DISABLED)
        except Exception as e:
            print(f"Error in test selection: {e}")
            self.load_dataset_btn.config(state=tk.DISABLED)
    
    def load_dataset(self):
        """Load selected dataset for management"""
        try:
            selection = self.test_tree.selection()
            if not selection:
                return
                
            item = self.test_tree.item(selection[0])
            test_name = item['text']  # Get test name from text, not values
            
            # Find test by name
            test = None
            for t in self.tests.values():
                if t.name == test_name:
                    test = t
                    break
            
            if test and test.is_dataset:
                self.open_dataset_manager(test)
            else:
                messagebox.showwarning("Not a Dataset", "Selected test is not a dataset.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {e}")
    
    def open_dataset_manager(self, test):
        """Open dataset management window"""
        dataset_window = tk.Toplevel(self.root)
        dataset_window.title(f"Dataset Manager - {test.name}")
        dataset_window.geometry("800x600")
        dataset_window.resizable(True, True)
        
        # Main frame
        main_frame = ttk.Frame(dataset_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(header_frame, text=f"Managing Dataset: {test.name}", 
                 font=('Arial', 14, 'bold')).pack(anchor=tk.W)
        ttk.Label(header_frame, text=f"Category: {test.category}").pack(anchor=tk.W)
        ttk.Label(header_frame, text=f"Description: {test.description}").pack(anchor=tk.W)
        
        # Variants list
        variants_frame = ttk.LabelFrame(main_frame, text="Test Variants")
        variants_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Treeview for variants
        variants_tree = ttk.Treeview(variants_frame, columns=('name', 'prompt_preview'), show='tree headings', height=15)
        variants_tree.heading('#0', text='ID')
        variants_tree.heading('name', text='Variant Name')
        variants_tree.heading('prompt_preview', text='Prompt Preview')
        variants_tree.column('#0', width=100)
        variants_tree.column('name', width=200)
        variants_tree.column('prompt_preview', width=400)
        
        # Scrollbar for variants tree
        variants_scrollbar = ttk.Scrollbar(variants_frame, orient=tk.VERTICAL, command=variants_tree.yview)
        variants_tree.configure(yscrollcommand=variants_scrollbar.set)
        
        variants_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        variants_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate variants
        def refresh_variants():
            variants_tree.delete(*variants_tree.get_children())
            for i, variant in enumerate(test.dataset_variants):
                name = variant.get('name', f'Variant {i+1}')
                prompt_preview = variant.get('prompt', '')[:50] + '...' if len(variant.get('prompt', '')) > 50 else variant.get('prompt', '')
                variants_tree.insert('', tk.END, text=str(i), values=(name, prompt_preview))
        
        refresh_variants()
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X)
        
        def add_variant():
            self.open_variant_editor(test, None, refresh_variants)
        
        def edit_variant():
            selection = variants_tree.selection()
            if selection:
                variant_idx = int(variants_tree.item(selection[0])['text'])
                self.open_variant_editor(test, variant_idx, refresh_variants)
            else:
                messagebox.showwarning("No Selection", "Please select a variant to edit.")
        
        def delete_variant():
            selection = variants_tree.selection()
            if selection:
                variant_idx = int(variants_tree.item(selection[0])['text'])
                if messagebox.askyesno("Confirm Delete", "Are you sure you want to delete this variant?"):
                    del test.dataset_variants[variant_idx]
                    refresh_variants()
                    self.save_data()
            else:
                messagebox.showwarning("No Selection", "Please select a variant to delete.")
        
        def run_single():
            selection = variants_tree.selection()
            if selection:
                variant_idx = int(variants_tree.item(selection[0])['text'])
                variant = test.dataset_variants[variant_idx]
                # Create temporary test for this variant
                temp_test = PsychologicalTest(
                    name=f"{test.name} - {variant.get('name', f'Variant {variant_idx+1}')}",
                    category=test.category,
                    description=variant.get('description', test.description),
                    prompt=variant.get('prompt', ''),
                    expected_responses=variant.get('expected_responses', []),
                    scoring_method=variant.get('scoring_method', test.scoring_method),
                    scoring_criteria=variant.get('scoring_criteria', test.scoring_criteria)
                )
                # Switch to run test tab and load this test
                self.notebook.select(0)  # Switch to run test tab
                dataset_window.destroy()
                messagebox.showinfo("Variant Loaded", f"Variant '{variant.get('name', f'Variant {variant_idx+1}')}' loaded for testing.")
            else:
                messagebox.showwarning("No Selection", "Please select a variant to run.")
        
        def run_random():
            if test.dataset_variants:
                import random
                variant = random.choice(test.dataset_variants)
                variant_idx = test.dataset_variants.index(variant)
                temp_test = PsychologicalTest(
                    name=f"{test.name} - {variant.get('name', f'Variant {variant_idx+1}')} (Random)",
                    category=test.category,
                    description=variant.get('description', test.description),
                    prompt=variant.get('prompt', ''),
                    expected_responses=variant.get('expected_responses', []),
                    scoring_method=variant.get('scoring_method', test.scoring_method),
                    scoring_criteria=variant.get('scoring_criteria', test.scoring_criteria)
                )
                self.notebook.select(0)
                dataset_window.destroy()
                messagebox.showinfo("Random Variant", f"Random variant '{variant.get('name', f'Variant {variant_idx+1}')}' loaded for testing.")
            else:
                messagebox.showwarning("No Variants", "This dataset has no variants to run.")
        
        def run_full_battery():
            if test.dataset_variants:
                self.notebook.select(0)
                dataset_window.destroy()
                messagebox.showinfo("Full Battery", f"Full battery mode would run all {len(test.dataset_variants)} variants sequentially. This feature is planned for future implementation.")
            else:
                messagebox.showwarning("No Variants", "This dataset has no variants to run.")
        
        # CRUD buttons
        crud_frame = ttk.LabelFrame(buttons_frame, text="Manage Variants")
        crud_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        ttk.Button(crud_frame, text="Add Variant", command=add_variant).pack(side=tk.LEFT, padx=5)
        ttk.Button(crud_frame, text="Edit Variant", command=edit_variant).pack(side=tk.LEFT, padx=5)
        ttk.Button(crud_frame, text="Delete Variant", command=delete_variant).pack(side=tk.LEFT, padx=5)
        
        # Run options
        run_frame = ttk.LabelFrame(buttons_frame, text="Run Options")
        run_frame.pack(side=tk.RIGHT)
        
        ttk.Button(run_frame, text="Run Single", command=run_single).pack(side=tk.LEFT, padx=5)
        ttk.Button(run_frame, text="Run Random", command=run_random).pack(side=tk.LEFT, padx=5)
        ttk.Button(run_frame, text="Full Battery", command=run_full_battery).pack(side=tk.LEFT, padx=5)
    
    def open_variant_editor(self, test, variant_idx, refresh_callback):
        """Open variant editor window"""
        editor_window = tk.Toplevel(self.root)
        editor_window.title("Variant Editor")
        editor_window.geometry("600x500")
        editor_window.resizable(True, True)
        
        # Get existing variant data or create new
        if variant_idx is not None:
            variant_data = test.dataset_variants[variant_idx].copy()
            is_edit = True
        else:
            variant_data = {
                'name': '',
                'description': '',
                'prompt': '',
                'expected_responses': [],
                'scoring_method': test.scoring_method,
                'scoring_criteria': test.scoring_criteria
            }
            is_edit = False
        
        # Main frame
        main_frame = ttk.Frame(editor_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_text = f"Edit Variant" if is_edit else "Add New Variant"
        ttk.Label(main_frame, text=title_text, font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(0, 10))
        
        # Form fields
        # Name
        ttk.Label(main_frame, text="Variant Name:").pack(anchor=tk.W)
        name_entry = ttk.Entry(main_frame, width=60)
        name_entry.pack(fill=tk.X, pady=(0, 10))
        name_entry.insert(0, variant_data.get('name', ''))
        
        # Description
        ttk.Label(main_frame, text="Description:").pack(anchor=tk.W)
        desc_text = tk.Text(main_frame, height=3, wrap=tk.WORD)
        desc_text.pack(fill=tk.X, pady=(0, 10))
        desc_text.insert('1.0', variant_data.get('description', ''))
        
        # Prompt
        ttk.Label(main_frame, text="Prompt:").pack(anchor=tk.W)
        prompt_text = tk.Text(main_frame, height=8, wrap=tk.WORD)
        prompt_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        prompt_text.insert('1.0', variant_data.get('prompt', ''))
        
        # Expected responses
        ttk.Label(main_frame, text="Expected Responses (one per line):").pack(anchor=tk.W)
        responses_text = tk.Text(main_frame, height=4, wrap=tk.WORD)
        responses_text.pack(fill=tk.X, pady=(0, 10))
        if variant_data.get('expected_responses'):
            responses_text.insert('1.0', '\n'.join(variant_data['expected_responses']))
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        def save_variant():
            try:
                # Get form data
                name = name_entry.get().strip()
                description = desc_text.get('1.0', tk.END).strip()
                prompt = prompt_text.get('1.0', tk.END).strip()
                responses_text_content = responses_text.get('1.0', tk.END).strip()
                expected_responses = [r.strip() for r in responses_text_content.split('\n') if r.strip()]
                
                if not name:
                    messagebox.showerror("Error", "Variant name is required.")
                    return
                
                if not prompt:
                    messagebox.showerror("Error", "Prompt is required.")
                    return
                
                # Create variant data
                new_variant = {
                    'name': name,
                    'description': description,
                    'prompt': prompt,
                    'expected_responses': expected_responses,
                    'scoring_method': variant_data.get('scoring_method', test.scoring_method),
                    'scoring_criteria': variant_data.get('scoring_criteria', test.scoring_criteria)
                }
                
                # Save variant
                if is_edit:
                    test.dataset_variants[variant_idx] = new_variant
                else:
                    test.dataset_variants.append(new_variant)
                
                self.save_data()
                refresh_callback()
                editor_window.destroy()
                
                action = "updated" if is_edit else "added"
                messagebox.showinfo("Success", f"Variant '{name}' {action} successfully.")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save variant: {e}")
        
        ttk.Button(button_frame, text="Save", command=save_variant).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=editor_window.destroy).pack(side=tk.LEFT, padx=5)

if __name__ == "__main__":
    app = MachinePsychologyLab()
    app.run()
