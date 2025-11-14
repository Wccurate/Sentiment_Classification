#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TF-IDF feature extraction for text classification.
Converts text data to TF-IDF vectors suitable for sklearn models.
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix


def df_to_tfidf_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_column: str = "text",
    label_column: str = "label",
    min_df: int = 1,
    max_df: float = 1.0,
    max_features: Optional[int] = None,
    ngram_range: Tuple[int, int] = (1, 1),
    lowercase: bool = True,
    stop_words: Optional[str] = None,
    return_vectorizer: bool = False,
) -> Tuple[csr_matrix, np.ndarray, csr_matrix, np.ndarray]:
    """
    Convert DataFrame text to TF-IDF features for sklearn models.
    
    Args:
        train_df: Training DataFrame with text and label columns.
        test_df: Test DataFrame with text and label columns.
        text_column: Name of the text column (default: "text").
        label_column: Name of the label column (default: "label").
        min_df: Minimum document frequency for a term to be included.
                Can be int (absolute count) or float (proportion of documents).
                Default: 1 (include all terms).
        max_df: Maximum document frequency for a term to be included.
                Can be int (absolute count) or float (proportion of documents).
                Default: 1.0 (include all terms).
        max_features: Maximum number of features (vocabulary size).
                      If None, use all features. Default: None.
        ngram_range: Range of n-grams to extract.
                     (1, 1) for unigrams, (1, 2) for unigrams+bigrams.
                     Default: (1, 1).
        lowercase: Convert all text to lowercase before vectorization.
                   Default: True.
        stop_words: Stop words to remove. Can be 'english' or None.
                    Default: None.
        return_vectorizer: If True, also return the fitted vectorizer.
                          Default: False.
    
    Returns:
        If return_vectorizer is False:
            X_train: TF-IDF features for training set (sparse matrix).
            y_train: Labels for training set (numpy array).
            X_test: TF-IDF features for test set (sparse matrix).
            y_test: Labels for test set (numpy array).
        
        If return_vectorizer is True:
            (X_train, y_train, X_test, y_test, vectorizer)
    
    Example:
        >>> train_df = pd.DataFrame({
        ...     'text': ['hello world', 'goodbye world'],
        ...     'label': [0, 1]
        ... })
        >>> test_df = pd.DataFrame({
        ...     'text': ['hello there'],
        ...     'label': [0]
        ... })
        >>> X_train, y_train, X_test, y_test = df_to_tfidf_features(
        ...     train_df, test_df,
        ...     min_df=1,
        ...     max_features=1000,
        ...     ngram_range=(1, 2)
        ... )
    """
    # Extract text and labels
    train_texts = train_df[text_column].values
    train_labels = train_df[label_column].values
    test_texts = test_df[text_column].values
    test_labels = test_df[label_column].values
    
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        ngram_range=ngram_range,
        lowercase=lowercase,
        stop_words=stop_words,
        sublinear_tf=True,  # Apply sublinear tf scaling (1 + log(tf))
        use_idf=True,
        smooth_idf=True,
        norm='l2',  # L2 normalization
    )
    
    # Fit on training data and transform both train and test
    print(f"[+] Fitting TF-IDF vectorizer on training data...")
    print(f"    - min_df={min_df}, max_df={max_df}")
    print(f"    - max_features={max_features}")
    print(f"    - ngram_range={ngram_range}")
    print(f"    - lowercase={lowercase}, stop_words={stop_words}")
    
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    
    # Print vocabulary statistics
    vocab_size = len(vectorizer.vocabulary_)
    print(f"[✓] Vectorization complete!")
    print(f"    - Vocabulary size: {vocab_size}")
    print(f"    - Train shape: {X_train.shape}")
    print(f"    - Test shape: {X_test.shape}")
    print(f"    - Train sparsity: {(1.0 - X_train.nnz / (X_train.shape[0] * X_train.shape[1])) * 100:.2f}%")
    
    y_train = train_labels
    y_test = test_labels
    
    if return_vectorizer:
        return X_train, y_train, X_test, y_test, vectorizer
    else:
        return X_train, y_train, X_test, y_test


def get_top_tfidf_terms(
    vectorizer: TfidfVectorizer,
    X: csr_matrix,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Get the top TF-IDF terms for each document.
    
    Args:
        vectorizer: Fitted TfidfVectorizer instance.
        X: TF-IDF feature matrix (sparse).
        top_n: Number of top terms to retrieve per document.
    
    Returns:
        DataFrame with top terms and their scores for each document.
    """
    feature_names = vectorizer.get_feature_names_out()
    top_terms_list = []
    
    for doc_idx in range(X.shape[0]):
        doc_vec = X[doc_idx].toarray().flatten()
        top_indices = doc_vec.argsort()[-top_n:][::-1]
        top_terms = [(feature_names[i], doc_vec[i]) for i in top_indices if doc_vec[i] > 0]
        top_terms_list.append(top_terms)
    
    return pd.DataFrame({
        'doc_index': range(len(top_terms_list)),
        'top_terms': top_terms_list
    })


if __name__ == "__main__":
    # Example usage and testing
    import sys
    sys.path.append("/Users/wangshibo/Documents/Academic/Course/6405Project/Intent_Recognition_Exp")
    
    from data.loaders.load_raw_data import load_raw_atis, dataset_to_dataframe
    
    print("="*80)
    print("TF-IDF Feature Extraction Example")
    print("="*80)
    
    # Load dataset
    print("\n[1] Loading dataset...")
    dataset, dataset_stats = load_raw_atis()
    train_df, test_df = dataset_to_dataframe(dataset)
    
    print(f"    Train size: {len(train_df)}, Test size: {len(test_df)}")
    print(f"    Number of labels: {dataset_stats['num_labels']}")
    
    # Extract TF-IDF features with different configurations
    print("\n" + "="*80)
    print("[2] Configuration 1: Unigrams only, no filtering")
    print("="*80)
    X_train, y_train, X_test, y_test = df_to_tfidf_features(
        train_df, test_df,
        min_df=1,
        max_df=1.0,
        max_features=None,
        ngram_range=(1, 1),
    )
    
    print("\n" + "="*80)
    print("[3] Configuration 2: Unigrams + Bigrams, max 5000 features")
    print("="*80)
    X_train2, y_train2, X_test2, y_test2, vectorizer = df_to_tfidf_features(
        train_df, test_df,
        min_df=2,
        max_df=0.95,
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english',
        return_vectorizer=True,
    )
    
    print("\n" + "="*80)
    print("[4] Sample TF-IDF terms from first document")
    print("="*80)
    top_terms_df = get_top_tfidf_terms(vectorizer, X_train2[:5], top_n=10)
    for idx, row in top_terms_df.iterrows():
        print(f"\nDocument {idx}:")
        print(f"Text: {train_df.iloc[idx]['text'][:80]}...")
        print(f"Label: {train_df.iloc[idx]['text_label']}")
        print(f"Top TF-IDF terms: {row['top_terms'][:5]}")
    
    print("\n" + "="*80)
    print("[5] Feature matrix statistics")
    print("="*80)
    print(f"Data type: {type(X_train)}")
    print(f"Is sparse: {isinstance(X_train, csr_matrix)}")
    print(f"Memory usage (train): {X_train.data.nbytes / 1024 / 1024:.2f} MB")
    print(f"Labels shape: {y_train.shape}")
    print(f"Unique labels: {len(np.unique(y_train))}")
    
    print("\n" + "="*80)
    print("✓ Feature extraction completed successfully!")
    print("="*80)
