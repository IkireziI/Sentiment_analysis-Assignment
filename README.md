# Sentiment Analysis: Traditional ML vs Deep Learning Approaches

![Sentiment Analysis](https://img.shields.io/badge/NLP-Sentiment%20Analysis-blue)
![Models](https://img.shields.io/badge/Models-LogisticRegression%20%7C%20LSTM-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)

##  Project Overview

This project implements and compares sentiment analysis techniques using both traditional machine learning and deep learning approaches. We analyze the IMDB movie reviews dataset to classify sentiment as positive or negative, showcasing the strengths and limitations of different methodologies.

### Key Features
- Comprehensive exploratory data analysis with multiple visualizations
- Text preprocessing pipeline optimized for sentiment analysis
- Implementation of Logistic Regression (traditional ML) and LSTM (deep learning) models
- Systematic hyperparameter tuning and performance evaluation
- In-depth comparison of model performance with detailed metrics

##  Dataset

We used the **IMDB Movie Reviews Dataset**, a benchmark collection for sentiment analysis containing 50,000 movie reviews labeled as positive or negative. This balanced dataset is ideal for binary sentiment classification tasks.

**Dataset characteristics:**
- 25,000 training reviews and 25,000 test reviews
- Equal distribution of positive and negative sentiments
- Reviews of varying lengths and complexity
- Rich vocabulary covering movie-specific terminology

##  Data Analysis & Preprocessing

### Exploratory Data Analysis

Our EDA revealed important insights about the dataset:

- **Review Length Distribution**: Most reviews contain 100-300 words, with some outliers extending to over 1,000 words
- **Class Balance**: Perfect 50/50 distribution between positive and negative reviews
- **Data Quality**: No missing reviews or corrupt entries detected
- **Word Frequency Analysis**: Identified common terms and their sentiment correlations

### Preprocessing Pipeline

We implemented a comprehensive text preprocessing workflow:

1. **Tokenization**: Convert texts into word/token sequences
2. **Stop Word Removal**: Eliminated common English stop words to reduce noise
3. **Vectorization Approaches**:
   - TF-IDF for traditional ML models (capturing word importance)
   - Tokenization with padding for sequential deep learning models
4. **Normalization**: Standardized text features for better model convergence

**Justification**: TF-IDF effectively captures keyword importance for traditional ML, while sequential tokenization preserves word order crucial for LSTMs.

##  Model Implementation

### Logistic Regression (Traditional ML)

We implemented Logistic Regression with TF-IDF features as our baseline traditional ML approach:

```python
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
```

**Justification**: Logistic Regression provides a strong baseline with interpretable results, works well with high-dimensional sparse data from TF-IDF, and trains efficiently.

### LSTM Model (Deep Learning)

We designed an LSTM architecture to capture sequential patterns in text:

```python
model = Sequential([
    Embedding(NUM_WORDS, 64),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

**Architecture details**:
- Embedding layer with 64 dimensions to capture semantic relationships
- LSTM layer with 64 units to process sequential information
- Dropout (0.5) for regularization to prevent overfitting
- Binary classification output with sigmoid activation

**Justification**: LSTM networks excel at capturing long-range dependencies in sequential data like text, preserving word order and context that traditional ML models ignore.

##  Experiments and Results

### Logistic Regression Experiments

| Experiment | Max Features (TF-IDF) | Solver    | Accuracy | F1 Score |
|------------|----------------------|-----------|----------|----------|
| LR-1       | 3000                 | liblinear | 0.87     | 0.86     |
| LR-2       | 5000                 | saga      | 0.88     | 0.87     |
| LR-3       | 10000                | newton-cg | 0.88     | 0.88     |

### LSTM Model Experiments

| Experiment | Embedding Dim | LSTM Units | Dropout | Batch Size | Accuracy | F1 Score |
|------------|---------------|------------|---------|------------|----------|----------|
| LSTM-1     | 64            | 64         | 0.5     | 128        | 0.88     | 0.88     |
| LSTM-2     | 128           | 128        | 0.3     | 128        | 0.89     | 0.89     |
| LSTM-3     | 64            | 32         | 0.5     | 64         | 0.87     | 0.87     |

### Key Insights from Experiments
- Increasing TF-IDF max features from 3000 to 5000 improved performance, but further increases showed diminishing returns
- Larger LSTM units and embedding dimensions generally improved model performance but required more training time
- Dropout was critical for preventing overfitting in LSTM models
- LSTM models consistently outperformed Logistic Regression, especially on longer and more complex reviews

##  Evaluation Metrics

We evaluated our models using multiple complementary metrics:

### Accuracy and F1 Score
- **Accuracy**: Overall correctness of predictions
- **F1 Score**: Harmonic mean of precision and recall, more robust for balanced datasets

### Confusion Matrix Analysis
We visualized confusion matrices to understand error patterns:
- Both models showed balanced false positive and false negative rates
- LSTM performed better at capturing nuanced negative reviews with positive words

### Learning Curves
We tracked training and validation metrics to detect overfitting:
- LSTM showed slight overfitting after 3 epochs
- Logistic Regression maintained stable performance

**Justification**: Using multiple metrics provides a more comprehensive view of model performance than accuracy alone, particularly for understanding error types.

##  Conclusion and Future Work

### Key Findings
- LSTM outperformed Logistic Regression with 1-2% higher accuracy and F1 scores
- Traditional ML approaches provide a strong baseline with much lower computational requirements
- Sequence modeling significantly improves performance on complex, longer reviews

### Limitations & Challenges
- LSTM required significantly more training time and resources
- Both models struggled with sarcasm and ambiguous language
- Limited dataset size potentially restricted model generalization

### Future Improvements
- Implement Transformer-based models (BERT, RoBERTa) for state-of-the-art performance
- Explore ensemble methods combining traditional ML and deep learning
- Add additional features like POS tags and n-grams
- Test cross-domain generalization on different types of reviews

##  Repository Structure

```
sentiment-analysis/
├── data/                      # Data directory
│   └── processed/             # Preprocessed data files
├── models/                    # Saved model files
│   ├── lstm_model.h5          # Trained LSTM model
│   └── logistic_model.pkl     # Trained Logistic Regression model
├── notebooks/                 # Jupyter notebooks
│   └── sentiment_analysis.ipynb  # Main analysis notebook
├── src/                       # Source code
│   ├── preprocessing.py       # Text preprocessing functions
│   ├── models.py              # Model definitions
│   └── evaluation.py          # Evaluation functions
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

##  Setup and Installation

```bash
# Clone the repository
git clone https://github.com/username/sentiment-analysis.git
cd sentiment-analysis

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Jupyter notebook
jupyter notebook notebooks/sentiment_analysis.ipynb
```

##  Usage

### Training Models
To train both models from scratch:
```python
# Run the entire notebook
jupyter nbconvert --execute notebooks/sentiment_analysis.ipynb
```

### Making Predictions
To use the trained models for predictions:
```python
# Load models from saved files
import pickle
from tensorflow.keras.models import load_model

# Load Logistic Regression model
with open('models/logistic_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

# Load LSTM model
lstm_model = load_model('models/lstm_model.h5')

# Make predictions
# See notebook for preprocessing steps
```

##  Team Contributions

- **NGUM**: EDA, data preprocessing, model evaluation
- **B**: LSTM implementation, hyperparameter tuning
- **C**: Logistic regression modeling, documentation
- **D**: Visualization, experiment design, report writing

##  Citations

- Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning Word Vectors for Sentiment Analysis. In *Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics* (pp. 142–150).
- Chollet, F. (2015). *Keras*. https://github.com/keras-team/keras
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research, 12*, 2825-2830.

##  License

This project is licensed under the ML- Techn_Groupwork at ALU.