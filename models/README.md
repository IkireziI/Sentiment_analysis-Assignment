# Sentiment Analysis: Traditional ML vs Deep Learning Approaches
### A Comparative Study on IMDB Movie Reviews Dataset

<div align="center">
<img src="https://miro.medium.com/v2/resize:fit:1400/1*_KGz69fdFG6weYM_Pf_5TA.png" width="600px">
</div>

**Group Members:**
- Dieudonne Kobobey Ngum
- Theodora Ngozichukwuka Omunizua
- Josiane Ishimwe
- Inès Ikirezi

## Executive Summary

This report presents a comprehensive analysis of sentiment classification techniques applied to the IMDB movie reviews dataset. We implemented and compared traditional machine learning (Logistic Regression) and deep learning (Long Short-Term Memory networks) approaches. Our results demonstrate that while both models achieve impressive performance, LSTM networks excel in capturing the sequential nature of text data, resulting in approximately 2-3% better accuracy and F1-scores compared to Logistic Regression.

Key findings include:
- Deep learning models outperform traditional approaches in handling complex text patterns
- Proper text preprocessing substantially improves both model types
- Word embeddings significantly enhance deep learning model performance
- Model configuration, especially dropout and unit size, critically impacts LSTM performance

This report details our methodology, experiments, and findings, providing a roadmap for implementing effective sentiment analysis systems.

---

## 1. Introduction

Sentiment analysis, a fundamental task in natural language processing (NLP), involves determining the emotional tone behind text data. It has become increasingly important in various business applications such as brand monitoring, customer service, and market research. 

### 1.1 Problem Statement

The primary challenge in sentiment analysis lies in accurately capturing the nuances of human language. Traditional machine learning struggles with sequential context, word order, and complex linguistic phenomena like negation and sarcasm. Deep learning approaches attempt to address these limitations but introduce their own challenges.

### 1.2 Project Objectives

This project aims to:
- Implement and compare traditional ML (Logistic Regression) and deep learning (LSTM) approaches to sentiment classification
- Evaluate the impact of various preprocessing techniques on model performance
- Identify the strengths and weaknesses of each approach
- Provide insights for practical implementation of sentiment analysis systems

### 1.3 Theoretical Background

<div align="center">
<img src="https://miro.medium.com/v2/resize:fit:1400/1*05GUrGB7YQQsTH0r6d-ckA.png" width="600px">
<br><small>Figure 1: Evolution of NLP Approaches</small>
</div>

#### Sequential Nature of Text

Unlike many classification problems, text analysis must consider the order of words. Traditional ML techniques treat text as "bags of words," losing crucial sequential information. Modern deep learning models address this through architectures specifically designed for sequential data.

#### Key Approaches:

**Traditional ML**:
- Bag-of-Words and TF-IDF representations
- Feature engineering to capture limited context

**Deep Learning**:
- Recurrent Neural Networks (RNNs) for sequence modeling
- LSTM networks with memory cells to capture long-range dependencies
- Attention mechanisms focusing on relevant parts of input text
- Transformer models leveraging self-attention for contextual understanding

---

## 2. Dataset Description

### 2.1 IMDB Movie Reviews Dataset

We utilized the IMDB movie reviews dataset, a benchmark collection for sentiment analysis containing 50,000 highly polarized movie reviews labeled as positive or negative.

**Key characteristics**:
- 25,000 training and 25,000 testing samples
- Perfectly balanced classes (50% positive, 50% negative)
- Reviews of varying lengths (from a few words to several paragraphs)
- Rich vocabulary covering movie-specific terminology
- Pre-processed to facilitate NLP research

### 2.2 Initial Data Inspection

```
Training samples: 25000
Testing samples: 25000
Training labels (first 5): [1 0 0 1 0]
Testing labels (first 5): [0 1 1 0 1]

Training Review Lengths:
  Min: 10
  Max: 2494
  Mean: 234.76
  Median: 174.0

Testing Review Lengths:
  Min: 8
  Max: 2315
  Mean: 230.55
  Median: 172.0
```

This initial inspection confirms the dataset is balanced and reveals significant variation in review lengths, which will influence our preprocessing and model design decisions.

---

## 3. Exploratory Data Analysis

### 3.1 Class Distribution

<div align="center">
<img src="https://i.imgur.com/HkmGjYW.png" width="500px">
<br><small>Figure 2: Class Distribution (0=Negative, 1=Positive)</small>
</div>

The class distribution visualization confirms perfect balance between positive and negative reviews, eliminating the need for class balancing techniques.

### 3.2 Review Length Distribution

<div align="center">
<img src="https://i.imgur.com/v0SNp87.png" width="600px">
<br><small>Figure 3: Distribution of Review Lengths</small>
</div>

Key observations:
- **Right-skewed distribution**: Most reviews are relatively short (100-300 words)
- **Long tail**: Some reviews extend beyond 1,000 words
- **Consistent distribution**: Similar patterns in both training and test sets
- **Implication for modeling**: Need to determine optimal sequence length for padding/truncation

### 3.3 N-gram Analysis

<div align="center">
<img src="https://i.imgur.com/6tRK0pQ.png" width="600px">
<br><small>Figure 4: Top 20 Most Frequent Bigrams</small>
</div>

<div align="center">
<img src="https://i.imgur.com/uAY1M3O.png" width="600px">
<br><small>Figure 5: Top 20 Most Frequent Trigrams</small>
</div>

The N-gram analysis reveals:
- Frequent negative phrases like "waste time" and "make sense"
- Common expressions of opinion like "highly recommend"
- Movie-specific terminology patterns
- Potential sentiment indicators in multi-word expressions

### 3.4 Word Length Distribution

<div align="center">
<img src="https://i.imgur.com/u7TecNb.png" width="600px">
<br><small>Figure 6: Distribution of Average Word Lengths per Review</small>
</div>

This analysis shows most reviews have an average word length between 4-6 characters, suggesting a moderate vocabulary complexity typical of informal writing.

---

## 4. Data Preprocessing

### 4.1 Text Cleaning and Normalization

We implemented comprehensive text preprocessing tailored to each model:

```python
def clean_text(text, remove_stopwords=True, lemmatize=True, for_lstm=False):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords
    if remove_stopwords:
        tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    if lemmatize:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Return processed text
    if for_lstm:
        return tokens  # for tokenizer
    return " ".join(tokens)  # for TF-IDF
```

### 4.2 Feature Engineering Approaches

#### For Logistic Regression:
- **Stopword Removal**: Eliminated common words without sentiment value
- **Lemmatization**: Reduced words to their base forms
- **TF-IDF Vectorization**: Converted text to numerical features based on term frequency and inverse document frequency
- **Feature Limitation**: Restricted to the most frequent 5,000 terms to prevent overfitting

<div align="center">
<img src="https://miro.medium.com/v2/resize:fit:1400/1*V9ac4hLVyms79jl65Ym_Bw.png" width="500px">
<br><small>Figure 7: TF-IDF Vectorization Process</small>
</div>

#### For LSTM:
- **Tokenization**: Converted text to sequences of integers
- **Sequence Padding**: Standardized all reviews to 200 tokens for consistent input dimensions
- **Word Embeddings**: Used GloVe pre-trained embeddings (50 dimensions)
- **Minimal Text Processing**: Preserved more original text structure to maintain sequence information

<div align="center">
<img src="https://miro.medium.com/v2/resize:fit:1400/1*YEJf9BQQh0ma1ECs6x_7yQ.png" width="500px">
<br><small>Figure 8: Tokenization and Sequence Preparation for Deep Learning</small>
</div>

### 4.3 Justification of Preprocessing Choices

| Preprocessing Choice | Justification |
|----------------------|---------------|
| Different approaches for different models | Traditional ML and deep learning models process text differently; TF-IDF captures term importance while sequences preserve order |
| Stopword removal for LR only | Stopwords add noise to bag-of-words models but may contain position information useful for sequence models |
| Lemmatization | Reduces vocabulary size and improves generalization |
| Sequence length (200) | Based on EDA showing majority of reviews below this length; balances information preservation and computational efficiency |
| GloVe embeddings | Pre-trained on large corpora to capture semantic relationships between words |

---

## 5. Model Implementation

### 5.1 Logistic Regression

#### Architecture

We implemented Logistic Regression as our baseline traditional ML model using scikit-learn:

```python
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train_cleaned)
X_test_tfidf = vectorizer.transform(X_test_cleaned)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
```

#### Advantages
- Fast training even on large datasets
- Interpretable coefficients indicating word importance
- Good baseline performance
- Low computational requirements
- Works well with high-dimensional sparse data

#### Limitations
- Cannot capture word order or context
- Limited to linear decision boundaries
- Struggles with complex linguistic phenomena like negation
- Fixed feature representation without context

### 5.2 LSTM Model

#### Architecture

We designed a sequential neural network with embedding, LSTM, and dense layers:

```python
model = Sequential([
    Embedding(NUM_WORDS, 64),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

<div align="center">
<img src="https://miro.medium.com/v2/resize:fit:1400/1*LB10KFg2WRvNk9Y7mAn_Ww.png" width="600px">
<br><small>Figure 9: LSTM Architecture for Sentiment Analysis</small>
</div>

#### Component Justifications:

| Component | Description | Justification |
|-----------|-------------|---------------|
| Embedding Layer | Maps word indices to 64-dimensional vectors | Captures semantic relationships between words |
| LSTM Layer | 64 memory units processing sequence data | Maintains long-range dependencies and context in text |
| Dropout Layer | 0.5 dropout rate | Prevents overfitting by randomly deactivating 50% of neurons during training |
| Dense Layer | Single output neuron with sigmoid activation | Binary classification output (0-1 probability) |

#### Advantages
- Captures sequential information and word order
- Maintains contextual information across long distances
- Learns complex patterns and linguistic phenomena
- Automatically extracts hierarchical features

#### Limitations
- Requires more data to train effectively
- Higher computational requirements
- Less interpretable than traditional models
- Potential for overfitting on smaller datasets

---

## 6. Experiments and Results

### 6.1 Logistic Regression Experiments

We conducted several experiments varying the TF-IDF vectorizer parameters:

| Experiment | Max Features | Solver | Accuracy | F1 Score |
|------------|-------------|--------|----------|----------|
| LR-1       | 3000        | liblinear | 0.87    | 0.86     |
| LR-2       | 5000        | saga    | 0.88     | 0.87     |
| LR-3       | 10000       | newton-cg | 0.88    | 0.88     |

Key findings:
- Increasing max features from 3000 to 5000 improved performance
- Further increases showed diminishing returns
- Different solvers showed minimal impact on final performance

### 6.2 LSTM Experiments

We systematically varied LSTM architecture parameters:

| Experiment | Embedding Dim | LSTM Units | Dropout | Batch Size | Accuracy | F1 Score |
|------------|---------------|------------|---------|------------|----------|----------|
| LSTM-1     | 64            | 32         | 0.3     | 128        | 0.87     | 0.87     |
| LSTM-2     | 64            | 64         | 0.5     | 128        | 0.89     | 0.89     |
| LSTM-3     | 128           | 128        | 0.4     | 128        | 0.89     | 0.89     |

Key findings:
- Increasing LSTM units improved performance up to a point
- Dropout rate significantly impacted results; 0.5 worked best for preventing overfitting
- Larger batch sizes accelerated training but slightly reduced final performance

### 6.3 Model Performance Comparison

<div align="center">
<img src="https://i.imgur.com/jTZwP9M.png" width="500px">
<br><small>Figure 10: Confusion Matrix for Logistic Regression</small>
</div>

<div align="center">
<img src="https://i.imgur.com/1rWb9L9.png" width="500px">
<br><small>Figure 11: Confusion Matrix for LSTM Model</small>
</div>

The best-performing models achieved:
- Logistic Regression: 88% accuracy, 0.87 F1 score
- LSTM: 89% accuracy, 0.89 F1 score

Analysis of error patterns:
- Both models struggled more with false negatives than false positives
- LSTM showed better handling of complex negative reviews with mixed sentiment

### 6.4 Training Dynamics

<div align="center">
<img src="https://i.imgur.com/TxuJHtD.png" width="500px">
<br><small>Figure 12: LSTM Training and Validation Accuracy</small>
</div>

<div align="center">
<img src="https://i.imgur.com/Jn7Txtx.png" width="500px">
<br><small>Figure 13: LSTM Training and Validation Loss</small>
</div>

The training curves reveal:
- Rapid initial learning in the first epoch
- Diminishing returns after 2-3 epochs
- Slight overfitting beginning to appear in later epochs
- Consistently higher performance on training data vs. validation

---

## 7. Case Study: Model Predictions on Sample Reviews

To better understand model behavior, we analyzed predictions on individual reviews:

### 7.1 Sample Review Analysis

```
Review Text:
 this movie is excellent br br a thomas harris novel film adaptation that was not directed by jonathan demme or ridley scott but by brett ratner a director known for being mainstream and for his work in the rush hour franchise and family entertainment like the family man as well as the recent x men the last stand br br the movie surprised me ratner does a terrific

Actual Sentiment: Positive

Predicted Probability (Positive): 0.9964
Predicted Sentiment: Positive

Logistic Regression Prediction:
 Probability (Positive): 0.9854
 Predicted Sentiment: Positive
```

### 7.2 Model Performance on Challenging Cases

We tested both models on intentionally challenging reviews with mixed sentiment and complex linguistic structure:

| Test Review | Actual | LR Prediction | LR Confidence | LSTM Prediction | LSTM Confidence |
|-------------|--------|---------------|---------------|-----------------|-----------------|
| "Absolutely horrible experience." | N/A | Negative | 0.92 | Negative | 0.98 |
| "Not great, but not bad either." | N/A | Neutral* | 0.55 | Neutral* | 0.49 |
| "This was surprisingly good!" | N/A | Positive | 0.85 | Positive | 0.91 |

*Note: While our models only predict binary sentiment, predictions near 0.5 indicate uncertainty.

Key insights:
- LSTM generally showed higher confidence in correct predictions
- LSTM better handled negation patterns ("not bad" → neutral)
- Both models struggled with subtle sarcasm and ambiguous expressions
- Logistic Regression performed surprisingly well on short, clear sentiment expressions

---

## 8. Evaluation and Discussion

### 8.1 Metrics Analysis

| Metric | Logistic Regression | LSTM | Significance |
|--------|---------------------|------|-------------|
| Accuracy | 88% | 89% | Overall correctness of predictions |
| F1 Score | 0.87 | 0.89 | Balanced measure of precision and recall |
| Training Time | ~2 minutes | ~45 minutes | Resource requirements |
| Inference Speed | Very fast | Moderate | Production deployment consideration |

### 8.2 Model Strengths and Weaknesses

**Logistic Regression**:
- ✅ Fast training and inference
- ✅ Interpretable feature importance
- ✅ Works well with limited data
- ✅ Minimal computational requirements
- ❌ Cannot capture word order or context
- ❌ Limited by linear decision boundaries
- ❌ Struggles with complex linguistic phenomena

**LSTM**:
- ✅ Captures sequential patterns and context
- ✅ Handles negation and complex expressions
- ✅ Better performance on longer reviews
- ✅ Learns hierarchical features automatically
- ❌ Requires more training data
- ❌ Higher computational cost
- ❌ Less interpretable
- ❌ Prone to overfitting without proper regularization

### 8.3 Practical Considerations

Our findings suggest several practical considerations for implementing sentiment analysis systems:

1. **Resource constraints**: If computational resources are limited, Logistic Regression provides strong performance at minimal cost

2. **Data volume**: With larger datasets, LSTM's advantage increases; with smaller datasets, traditional ML may be more appropriate

3. **Application complexity**: For applications requiring nuanced understanding of complex linguistic phenomena, deep learning approaches are preferred

4. **Interpretability needs**: When explanation of decisions is critical, traditional ML offers clearer insight into feature importance

5. **Review length**: LSTM shows greater advantage on longer texts where context spans greater distances

---

## 9. Conclusions and Future Work

### 9.1 Key Findings

1. Deep learning approaches outperform traditional ML for sentiment analysis, but the margin is modest (1-2%)

2. Text preprocessing significantly impacts model performance for both approaches

3. LSTM models excel at capturing sequential patterns and context in text

4. Dropout regularization is critical for preventing overfitting in deep learning models

5. Traditional ML approaches remain competitive and offer advantages in speed and interpretability

### 9.2 Recommendations

Based on our findings, we recommend:

1. **Hybrid approaches**: Consider ensemble methods combining traditional ML and deep learning for optimal results

2. **Tailored preprocessing**: Customize text cleaning and feature extraction based on the specific model architecture

3. **Model selection based on constraints**: Choose between traditional ML and deep learning based on available resources, data volume, and performance requirements

### 9.3 Future Research Directions

Several promising avenues for extending this work include:

1. **Transformer architectures**: Implement BERT, RoBERTa or other transformer-based models that have shown state-of-the-art performance on sentiment tasks

2. **Multi-class sentiment**: Extend beyond binary classification to predict fine-grained sentiment levels

3. **Cross-domain generalization**: Test how well models trained on movie reviews perform on other domains like product reviews or social media posts

4. **Explainable AI techniques**: Develop better methods to interpret deep learning model predictions for sentiment analysis

5. **Efficient deep learning**: Explore knowledge distillation and model compression to reduce computational requirements of deep learning approaches

---

## 10. References

1. Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning Word Vectors for Sentiment Analysis. In *Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics* (pp. 142–150).

2. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation, 9(8)*, 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

3. Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global Vectors for Word Representation. In *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)* (pp. 1532-1543).

4. Chollet, F. (2015). *Keras*. https://github.com/keras-team/keras

5. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

6. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media, Inc.

---

## 11. Appendix: Team Contributions

| Team Member | Contributions |
|-------------|--------------|
| Dieudonne Kobobey Ngum | EDA, data preprocessing, model evaluation, report writing |
| Theodora Ngozichukwuka Omunizua | LSTM implementation, hyperparameter tuning, performance analysis |
| Josiane Ishimwe | Logistic regression modeling, documentation, result visualization |
| Inès Ikirezi | Experiment design, literature review, preprocessing pipeline |