# Explainable Sentiment Analysis with DistilBERT and SHAP

## Overview

Modern Natural Language Processing (NLP) models based on Transformer architectures achieve remarkable performance on text classification tasks. However, these models often behave as **black boxes**, making it difficult to understand why a particular prediction was made.

This project addresses that challenge by combining a **Transformer-based sentiment classifier (DistilBERT)** with **SHAP (SHapley Additive exPlanations)**, an Explainable AI (XAI) technique that attributes model predictions to individual input words.

The objective is not only to build an accurate sentiment classifier but also to provide transparent explanations for every prediction, making the model easier to interpret, debug, and trust.

---

# Project Objectives

The primary goals of this project are:

- Fine-tune a DistilBERT model for binary sentiment classification.
- Classify movie reviews as **Positive** or **Negative**.
- Explain individual predictions using SHAP.
- Visualize the contribution of each word to the final prediction.
- Demonstrate Explainable AI techniques for modern Transformer models.

---

# Motivation

Deep learning models have become increasingly powerful but also increasingly difficult to interpret.

In sensitive applications such as

- Healthcare
- Finance
- Education
- Legal systems

it is often insufficient to know **what** prediction was made.

We must also understand

> **Why was this prediction made?**

Explainable AI attempts to answer this question.

This project demonstrates one of the most widely used explanation methods: **SHAP**.

---

# Dataset

The project uses the **IMDb Movie Review Dataset** provided by Hugging Face.

The dataset contains approximately

- 25,000 training reviews
- 25,000 testing reviews

Each review is labeled as

- Positive
- Negative

Example

Positive

> "The acting was absolutely amazing."

Negative

> "The movie was boring and too long."

---

# Model

The classifier is built using **DistilBERT**, a compressed version of Google's BERT model.

DistilBERT is produced using **Knowledge Distillation**, where a smaller student model learns from a larger teacher model.

Compared with BERT

- ~40% fewer parameters
- Faster inference
- Lower memory usage
- Maintains approximately 97% of BERT's performance

This makes DistilBERT suitable for research experiments and deployment on limited hardware.

---

# Transformer Architecture

Unlike traditional Recurrent Neural Networks (RNNs), Transformers process an entire sentence simultaneously.

The main innovation is the **Self-Attention Mechanism**.

Instead of reading words sequentially,

```
The movie was surprisingly good
```

the model learns relationships among all words at once.

For every token, attention scores are computed using

\[
Attention(Q,K,V)=Softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

where

- Q = Query
- K = Key
- V = Value

This allows the model to capture long-range dependencies between words.

---

# Sentiment Classification

The Transformer produces contextual embeddings for every token.

The embedding corresponding to the **[CLS]** token is used as a representation of the entire sentence.

A linear classification layer predicts

\[
P(y|x)=Softmax(z)
\]

where

- x = input sentence
- z = logits
- y = sentiment class

The predicted class is

\[
\hat y=\arg\max P(y|x)
\]

---

# Training

The model is fine-tuned using supervised learning.

The optimization objective is Cross-Entropy Loss

\[
L=-\sum_i y_i\log(\hat y_i)
\]

where

- y = true label
- \(\hat y\) = predicted probability

Optimization is performed using AdamW.

---

# Explainable AI

Although Transformer models achieve high accuracy, their internal reasoning is difficult to understand.

This project integrates **SHAP** to explain model predictions.

SHAP is based on **Shapley Values**, originally developed in cooperative game theory.

The main idea is

> Every feature (word) receives credit according to its contribution to the prediction.

---

# Mathematical Foundation of SHAP

Suppose a prediction is generated from

\[
f(x)
\]

where

\[
x=(x_1,x_2,\ldots,x_n)
\]

Each feature receives a Shapley value

\[
\phi_i
\]

The SHAP explanation model is

\[
f(x)=\phi_0+\sum_{i=1}^{n}\phi_i
\]

where

- \(\phi_0\) = baseline prediction
- \(\phi_i\) = contribution of feature i

The Shapley value is computed as

\[
\phi_i=
\sum_{S\subseteq N\setminus\{i\}}
\frac{|S|!(n-|S|-1)!}{n!}
\left[
f(S\cup\{i\})-f(S)
\right]
\]

This equation measures the average contribution of a feature over every possible subset of features.

Although exact computation is computationally expensive, SHAP provides efficient approximations.

---

# Interpretation

Suppose the input is

> "The movie was fantastic but the ending was terrible."

SHAP may produce

| Word | SHAP Value |
|--------|-----------:|
| fantastic | +0.82 |
| movie | +0.03 |
| ending | -0.15 |
| terrible | -0.76 |

Interpretation

- **fantastic** strongly increases positive sentiment.
- **terrible** pushes the prediction toward negative sentiment.
- Neutral words contribute very little.

---

# Project Workflow

```
Input Text

↓

Tokenizer

↓

DistilBERT

↓

Sentiment Prediction

↓

SHAP Explainer

↓

Word Importance Visualization
```

---

# Technologies

- Python
- PyTorch
- Hugging Face Transformers
- SHAP
- Hugging Face Datasets
- NumPy
- Scikit-learn
- Matplotlib

---

# Results

The project successfully demonstrates

- Transformer-based sentiment classification
- Explainable AI using SHAP
- Word-level interpretation of predictions
- Visualization of feature importance
- Transparent decision making

---

# Future Improvements

Possible extensions include

- Compare SHAP with LIME
- Evaluate RoBERTa and DeBERTa
- Multi-class sentiment classification
- Explain Large Language Models
- Benchmark multiple XAI methods
- Deploy the model as a web application
- Evaluate explanation stability

---

# Learning Outcomes

Through this project, I gained practical experience with

- Transformer architectures
- Transfer learning
- Hugging Face ecosystem
- Explainable AI
- SHAP
- Model interpretability
- Deep learning for NLP
- Scientific machine learning workflows

---

# References

- Vaswani et al. (2017), *Attention Is All You Need*
- Devlin et al. (2019), *BERT: Pre-training of Deep Bidirectional Transformers*
- Sanh et al. (2019), *DistilBERT*
- Lundberg & Lee (2017), *A Unified Approach to Interpreting Model Predictions*
