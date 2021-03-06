# Classification of Disaster Tweets (Natural Language Processing)
This is the repository for the source code of the disaster tweets classification using NLP project found on https://www.kaggle.com/competitions/nlp-getting-started/  
My Jupyter Notebook on Kaggle is available here https://www.kaggle.com/code/alinaageichik/eda-bert-roberta-xlmroberta-and-ensembles

This project is implemented using Python 3.8.5, CUDA 11.6.  
Prerequisites can be found here: https://github.com/alite13/NLP-Disaster-Tweets-Classification/tree/main/BERT%2C%20RoBERTa%2C%20XLMRoBERTa%2C%20Ensembles#readme  

## Results:  
| Model | Loss Function | Optimizer | Epochs | Accuracy |
| --- | --- | --- | --- | --- |
| BERT | nn.CrossEntropy() | AdamW | 4 | 83.09% |
| RoBERTa | nn.CrossEntropy() | AdamW | 4 | 81.79% |
| XLMRoBERTa | nn.CrossEntropy() | AdamW | 4 | 79.89% |
| BERT + BERT (Ensemble) | nn.CrossEntropy() | AdamW | 4 | 79.04% |
| RoBERTa + RoBERTa (Ensemble) | nn.CrossEntropy() | AdamW | 4 | 82.37% |
| XLMRoBERTa + XLMRoBERTa (Ensemble) | nn.CrossEntropy() | AdamW | 4 | 81.2% |

## References

This project on Kaggle: https://www.kaggle.com/competitions/nlp-getting-started/overview  
BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding: https://arxiv.org/abs/1810.04805
RoBERTa: A Robustly Optimized BERT Pretraining Approach: https://arxiv.org/pdf/1907.11692.pdf
Unsupervised Cross-lingual Representation Learning at Scale: https://arxiv.org/pdf/1911.02116.pdf
