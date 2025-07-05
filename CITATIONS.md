# Citations and References

## Foundational Research

This project was inspired by and builds upon the following research:

### Primary Inspiration
**"Machine Psychology"**
- Authors: Thilo Hagendorff, Ishita Dasgupta, Marcel Binz, Stephanie C.Y. Chan, Andrew Lampinen, Jane X. Wang, Zeynep Akata, Eric Schulz
- arXiv: 2303.13988v6
- URL: https://arxiv.org/abs/2303.13988
- Date: August 9, 2024

*This foundational paper established the field of "machine psychology" - applying behavioral experiments from psychology to understand LLM behavior. It provides the theoretical framework and methodological foundation for systematic psychological assessment of AI models using established paradigms from cognitive science, developmental psychology, and behavioral sciences.*

### Related Machine Psychology Research
- **Binz, M. & Schulz, E.** (2023). Using cognitive psychology to understand GPT-3. Proceedings of the National Academy of Sciences, 120(6), 1-10.
- **Sap, M., Le Bras, R., Fried, D., & Choi, Y.** (2022). Neural Theory-of-Mind? On the Limits of Social Intelligence in Large LMs. Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing.
- **Strachan, J. W. A. et al.** (2024). Testing theory of mind in large language models and humans. Nature Human Behaviour, 8, 1285–1295.

## Analysis Engine Citations

When publishing research using this tool, please cite the appropriate analysis engines:

### Basic Analysis Engine
- **VADER Sentiment Analysis**: Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14).
- **TextBlob**: Loria, S. (2018). textblob Documentation. Release 0.15.2.
- **spaCy**: Honnibal, M. & Montani, I. (2017). spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing.

### Advanced Analysis Engine
- **RoBERTa Sentiment Model**: Barbieri, F., Camacho-Collados, J., Espinosa Anke, L., & Neves, L. (2022). TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification. In Findings of the Association for Computational Linguistics: EMNLP 2020.
  - Model: `cardiffnlp/twitter-roberta-base-sentiment-latest`
  - HuggingFace: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
- **Sentence-BERT**: Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP).
  - Model: `all-MiniLM-L6-v2`
  - HuggingFace: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

### Supporting Libraries
- **Transformers**: Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., ... & Rush, A. M. (2020). Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 conference on empirical methods in natural language processing: system demonstrations (pp. 38-45).
- **PyTorch**: Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. In Advances in neural information processing systems (pp. 8024-8035).

## Psychological Test References

### Theory of Mind Tests
- **Sally-Anne Test**: Baron-Cohen, S., Leslie, A. M., & Frith, U. (1985). Does the autistic child have a "theory of mind"? A case study. Cognition, 21(1), 37-46.

### Cognitive Bias Tests
- **Asian Disease Problem**: Tversky, A., & Kahneman, D. (1981). The framing of decisions and the psychology of choice. Science, 211(4481), 453-458.
- **Linda the Bank Teller**: Tversky, A., & Kahneman, D. (1983). Extensional versus intuitive reasoning: The conjunction fallacy in probability judgment. Psychological Review, 90(4), 293-315.

### Logical Reasoning Tests
- **Wason Selection Task**: Wason, P. C. (1968). Reasoning about a rule. Quarterly Journal of Experimental Psychology, 20(3), 273-281.

## Software Dependencies

### Core Scientific Computing
- **NumPy**: Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020).
- **SciPy**: Virtanen, P., Gommers, R., Oliphant, T.E. et al. SciPy 1.0: fundamental algorithms for scientific computing in Python. Nat Methods 17, 261–272 (2020).
- **scikit-learn**: Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. Journal of machine learning research, 12(Oct), 2825-2830.

## Recommended Citation for This Tool

If you use the Machine Psychology Lab in your research, please cite:

```bibtex
@software{machine_psychology_lab,
  title={Machine Psychology Lab: A Research Tool for Systematic Psychological Evaluation of AI Models},
  author={[Your Name/Institution]},
  year={2025},
  url={[Your Repository URL]},
  note={GPL-3.0 License}
}
```

## Academic Usage Guidelines

When publishing research using this tool:

1. **Specify the analysis engine(s) used** (Basic, Advanced, or Hybrid)
2. **Cite the underlying models and libraries** as listed above
3. **Reference the methodological approach** of dual evaluation (machine estimation + human evaluation)
4. **Acknowledge the foundational research** that inspired this work
5. **Include appropriate disclaimers** about machine analysis being interpretive, not diagnostic

## License and Attribution

This software is released under the GPL-3.0 License. The foundational research paper and all cited models retain their original licenses and attribution requirements.

For questions about citations or academic use, please refer to the individual papers and model documentation linked above.
