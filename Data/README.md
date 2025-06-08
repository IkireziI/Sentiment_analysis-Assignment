##  Official Sources for IMDB Movie Reviews Dataset

### 1. **Keras / TensorFlow Datasets (Direct API Access)**

This is the most convenient and officially maintained version for deep learning use:

 **[Keras IMDB Dataset Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb)**
You can load it directly with:

```python
from tensorflow.keras.datasets import imdb
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)
```

Includes:

* 25,000 training + 25,000 test samples
* Pre-tokenized word index
* Balanced classes

---

### 2. **Stanford AI Lab (Original Dataset Source)**

 **[IMDB Dataset from Stanford](https://ai.stanford.edu/~amaas/data/sentiment/)**

> Author: Andrew Maas et al.
> Citation: Maas et al. (2011) — Learning Word Vectors for Sentiment Analysis.

Includes:

* Full text `.txt` files for 50,000 reviews
* Labeled as `pos` or `neg` for binary classification
* Useable with manual preprocessing or TF-IDF

---

### 3. **Kaggle Version (CSV format, user-friendly)**

 **[Kaggle IMDB Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)**

Includes:

* CSV file with `review` and `sentiment` columns
* Already tokenized and labeled
* Easy for beginners or ML pipelines

>  Requires a Kaggle login to download

---

## Recommended Based on Use Case

| Use Case                                 | Recommended Source                |
| ---------------------------------------- | --------------------------------- |
| Deep Learning (LSTM/RNN)                 | ✅ Keras built-in                  |
| TF-IDF + Classical ML                    | ✅ Kaggle CSV or Stanford raw text |
| Full NLP Pipeline (Custom Preprocessing) | ✅ Stanford dataset                |

