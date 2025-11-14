#!/usr/bin/env python3
"""
SVM with TF-IDF for binary sentiment classification on Project dataset.
Features:
- 5-fold stratified cross-validation
- Text preprocessing (lowercase, punctuation, stopwords)
- TF-IDF vectorization
- PCA dimensionality reduction
- SVM with RBF (Gaussian) kernel
- Final train/test split evaluation
"""

import argparse
import json
import os
import re
import string
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm

# Import dataset loader
from data.loaders.load_raw_data import load_raw_project, dataset_to_dataframe


# ========== Text Preprocessing ==========

# Common English stopwords
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
    'further', 'then', 'once'
}


def preprocess_text(text: str, remove_stopwords: bool = True) -> str:
    """
    Preprocess text:
    1. Convert to lowercase
    2. Remove URLs
    3. Remove punctuation
    4. Remove extra whitespace
    5. Optionally remove stopwords
    
    Args:
        text: Input text string
        remove_stopwords: Whether to remove stopwords
        
    Returns:
        Preprocessed text string
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove digits
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords
    if remove_stopwords:
        tokens = text.split()
        tokens = [word for word in tokens if word not in STOPWORDS and len(word) > 2]
        text = ' '.join(tokens)
    
    return text


def preprocess_dataframe(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """
    Preprocess all texts in a DataFrame.
    
    Args:
        df: Input DataFrame
        text_column: Name of the text column
        
    Returns:
        DataFrame with preprocessed texts (original text saved as 'text_raw')
    """
    df = df.copy()
    print(f"Preprocessing {len(df)} texts...")
    
    # Save original text before preprocessing
    df['text_raw'] = df[text_column].copy()
    
    # Apply preprocessing with progress bar
    tqdm.pandas(desc="Preprocessing")
    df[text_column] = df[text_column].progress_apply(preprocess_text)
    
    # Remove empty texts after preprocessing
    df = df[df[text_column].str.len() > 0]
    print(f"After preprocessing: {len(df)} texts remaining")
    
    return df


# ========== Model Training and Evaluation ==========

def build_pipeline_components(
    X_train: np.ndarray,
    pca_components: int = 100,
    random_state: int = 42
) -> Tuple[StandardScaler, PCA]:
    """
    Build and fit preprocessing components.
    
    Args:
        X_train: Training feature matrix
        pca_components: Number of PCA components
        random_state: Random seed
        
    Returns:
        Fitted scaler and PCA objects
    """
    # Standardization
    scaler = StandardScaler(with_mean=False)  # TF-IDF is sparse, don't center
    X_train_scaled = scaler.fit_transform(X_train)
    
    # PCA dimensionality reduction
    n_components = min(pca_components, X_train.shape[0], X_train.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X_train_scaled)
    
    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"PCA: {n_components} components explain {explained_var:.2%} of variance")
    
    return scaler, pca


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    gamma: str = 'scale',
    random_state: int = 42,
    verbose: bool = False
) -> SVC:
    """
    Train SVM classifier with RBF kernel.
    
    Args:
        X_train: Training features
        y_train: Training labels
        C: Regularization parameter
        gamma: Kernel coefficient
        random_state: Random seed
        verbose: Whether to print training info
        
    Returns:
        Trained SVC model
    """
    clf = SVC(
        kernel='rbf',
        C=C,
        gamma=gamma,
        random_state=random_state,
        verbose=verbose,
        max_iter=-1  # No limit on iterations
    )
    
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(
    clf: SVC,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str = "Test"
) -> dict:
    """
    Evaluate model and return metrics.
    
    Args:
        clf: Trained classifier
        X: Feature matrix
        y: True labels
        split_name: Name of the split (for printing)
        
    Returns:
        Dictionary of metrics
    """
    y_pred = clf.predict(X)
    
    accuracy = accuracy_score(y, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, y_pred, average='binary', pos_label=1
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'split': split_name
    }
    
    return metrics


def cross_validate_svm(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    pca_components: int = 100,
    C: float = 1.0,
    gamma: str = 'scale',
    random_state: int = 42
) -> List[dict]:
    """
    Perform stratified k-fold cross-validation.
    
    Args:
        X: Feature matrix (TF-IDF)
        y: Labels
        n_folds: Number of folds
        pca_components: Number of PCA components
        C: SVM regularization parameter
        gamma: SVM kernel coefficient
        random_state: Random seed
        
    Returns:
        List of metrics dictionaries for each fold
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    fold_metrics = []
    
    print(f"\n{'='*80}")
    print(f"Starting {n_folds}-Fold Stratified Cross-Validation")
    print(f"{'='*80}")
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n--- Fold {fold_idx}/{n_folds} ---")
        
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        print(f"Train samples: {len(y_train_fold)}, Val samples: {len(y_val_fold)}")
        print(f"Train class distribution: {np.bincount(y_train_fold)}")
        print(f"Val class distribution: {np.bincount(y_val_fold)}")
        
        # Build preprocessing pipeline
        scaler, pca = build_pipeline_components(
            X_train_fold,
            pca_components=pca_components,
            random_state=random_state
        )
        
        # Transform data
        X_train_scaled = scaler.transform(X_train_fold)
        X_train_pca = pca.transform(X_train_scaled)
        X_val_scaled = scaler.transform(X_val_fold)
        X_val_pca = pca.transform(X_val_scaled)
        
        # Train SVM
        print("Training SVM with RBF kernel...")
        clf = train_svm(
            X_train_pca,
            y_train_fold,
            C=C,
            gamma=gamma,
            random_state=random_state
        )
        
        # Evaluate on validation set
        metrics = evaluate_model(clf, X_val_pca, y_val_fold, split_name=f"Fold-{fold_idx}")
        fold_metrics.append(metrics)
        
        print(f"Fold {fold_idx} Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
    
    # Compute average metrics
    print(f"\n{'='*80}")
    print("Cross-Validation Summary")
    print(f"{'='*80}")
    
    avg_acc = np.mean([m['accuracy'] for m in fold_metrics])
    avg_prec = np.mean([m['precision'] for m in fold_metrics])
    avg_rec = np.mean([m['recall'] for m in fold_metrics])
    avg_f1 = np.mean([m['f1'] for m in fold_metrics])
    
    std_acc = np.std([m['accuracy'] for m in fold_metrics])
    std_prec = np.std([m['precision'] for m in fold_metrics])
    std_rec = np.std([m['recall'] for m in fold_metrics])
    std_f1 = np.std([m['f1'] for m in fold_metrics])
    
    print(f"Average Accuracy:  {avg_acc:.4f} ± {std_acc:.4f}")
    print(f"Average Precision: {avg_prec:.4f} ± {std_prec:.4f}")
    print(f"Average Recall:    {avg_rec:.4f} ± {std_rec:.4f}")
    print(f"Average F1 Score:  {avg_f1:.4f} ± {std_f1:.4f}")
    
    return fold_metrics


def final_train_test_evaluation(
    df: pd.DataFrame,
    text_column: str = 'text',
    label_column: str = 'label',
    test_size: float = 0.2,
    pca_components: int = 100,
    C: float = 1.0,
    gamma: str = 'scale',
    max_features: int = 5000,
    random_state: int = 42,
    output_dir: str = "outputs/svm_tfidf_results"
) -> dict:
    """
    Final training and testing with stratified split.
    
    Args:
        df: DataFrame with preprocessed text and labels
        text_column: Name of text column
        label_column: Name of label column
        test_size: Proportion of test set
        pca_components: Number of PCA components
        C: SVM regularization parameter
        gamma: SVM kernel coefficient
        max_features: Maximum number of TF-IDF features
        random_state: Random seed
        output_dir: Directory to save results
        
    Returns:
        Dictionary of test metrics
    """
    print(f"\n{'='*80}")
    print("Final Train/Test Split Evaluation")
    print(f"{'='*80}")
    
    # Stratified train/test split
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_column]
    )
    
    # Reset index to ensure proper sample tracking
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    print(f"\nDataset split (stratified):")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    print(f"  Train class distribution: {train_df[label_column].value_counts().to_dict()}")
    print(f"  Test class distribution:  {test_df[label_column].value_counts().to_dict()}")
    
    # TF-IDF Vectorization
    print(f"\nBuilding TF-IDF vectorizer (max_features={max_features})...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=2,            # Ignore terms that appear in less than 2 documents
        max_df=0.95,         # Ignore terms that appear in more than 95% of documents
        sublinear_tf=True    # Apply sublinear tf scaling
    )
    
    X_train_tfidf = vectorizer.fit_transform(train_df[text_column])
    X_test_tfidf = vectorizer.transform(test_df[text_column])
    
    y_train = train_df[label_column].values
    y_test = test_df[label_column].values
    
    print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # Build preprocessing pipeline
    print("\nApplying PCA dimensionality reduction...")
    scaler, pca = build_pipeline_components(
        X_train_tfidf,
        pca_components=pca_components,
        random_state=random_state
    )
    
    X_train_scaled = scaler.transform(X_train_tfidf)
    X_train_pca = pca.transform(X_train_scaled)
    X_test_scaled = scaler.transform(X_test_tfidf)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"After PCA: {X_train_pca.shape}")
    
    # Train SVM
    print("\nTraining SVM with RBF (Gaussian) kernel...")
    print(f"  C={C}, gamma={gamma}")
    clf = train_svm(
        X_train_pca,
        y_train,
        C=C,
        gamma=gamma,
        random_state=random_state,
        verbose=False
    )
    
    # Evaluate on train set
    train_metrics = evaluate_model(clf, X_train_pca, y_train, split_name="Train")
    
    # Evaluate on test set and get predictions
    y_pred_test = clf.predict(X_test_pca)
    test_metrics = evaluate_model(clf, X_test_pca, y_test, split_name="Test")
    
    # Get decision function scores (distance from hyperplane)
    decision_scores = clf.decision_function(X_test_pca)
    
    # Print results
    print(f"\n{'='*80}")
    print("FINAL EVALUATION RESULTS")
    print(f"{'='*80}")
    
    print("\nTrain Set:")
    print(f"  Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"  Precision: {train_metrics['precision']:.4f}")
    print(f"  Recall:    {train_metrics['recall']:.4f}")
    print(f"  F1 Score:  {train_metrics['f1']:.4f}")
    
    print("\nTest Set:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    
    # Classification report
    print("\nDetailed Classification Report (Test Set):")
    print(classification_report(
        y_test,
        y_pred_test,
        target_names=['Negative (0)', 'Positive (1)'],
        digits=4
    ))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    print("\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"               Neg    Pos")
    print(f"Actual Neg   {cm[0, 0]:5d}  {cm[0, 1]:5d}")
    print(f"       Pos   {cm[1, 0]:5d}  {cm[1, 1]:5d}")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model artifacts
    print(f"\nSaving model artifacts to: {output_path}")
    joblib.dump(vectorizer, output_path / "tfidf_vectorizer.pkl")
    joblib.dump(scaler, output_path / "scaler.pkl")
    joblib.dump(pca, output_path / "pca.pkl")
    joblib.dump(clf, output_path / "svm_classifier.pkl")
    
    # Save detailed predictions for each test sample
    print("Saving detailed test predictions...")
    test_predictions = []
    correct_count = 0
    error_count = 0
    
    for idx in range(len(test_df)):
        text_preprocessed = test_df.iloc[idx][text_column]
        text_raw = test_df.iloc[idx]['text_raw'] if 'text_raw' in test_df.columns else text_preprocessed
        true_label = int(y_test[idx])
        pred_label = int(y_pred_test[idx])
        decision_score = float(decision_scores[idx])
        is_correct = (true_label == pred_label)
        
        if is_correct:
            correct_count += 1
        else:
            error_count += 1
        
        sample_result = {
            "sample_id": idx,
            "text_raw": text_raw,
            "text_preprocessed": text_preprocessed,
            "true_label": true_label,
            "true_label_name": "negative" if true_label == 0 else "positive",
            "predicted_label": pred_label,
            "predicted_label_name": "negative" if pred_label == 0 else "positive",
            "decision_score": decision_score,
            "is_correct": is_correct
        }
        test_predictions.append(sample_result)
    
    # Save all test predictions to JSON
    predictions_file = output_path / "test_predictions.json"
    with open(predictions_file, "w", encoding="utf-8") as f:
        json.dump(test_predictions, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(test_predictions)} test predictions to: {predictions_file}")
    print(f"  Correct: {correct_count}")
    print(f"  Errors: {error_count}")
    
    # Save error samples separately
    error_samples = [sample for sample in test_predictions if not sample["is_correct"]]
    if error_samples:
        error_file = output_path / "error_samples.json"
        with open(error_file, "w", encoding="utf-8") as f:
            json.dump(error_samples, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(error_samples)} error samples to: {error_file}")
    
    # Save evaluation results with metadata
    evaluation_results = {
        "model_type": "SVM with RBF kernel",
        "dataset": "Project (Binary Sentiment)",
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "hyperparameters": {
            "C": C,
            "gamma": gamma,
            "kernel": "rbf",
            "max_features": max_features,
            "pca_components": pca_components,
            "random_state": random_state
        },
        "train_metrics": {
            "accuracy": float(train_metrics['accuracy']),
            "precision": float(train_metrics['precision']),
            "recall": float(train_metrics['recall']),
            "f1_score": float(train_metrics['f1'])
        },
        "test_metrics": {
            "accuracy": float(test_metrics['accuracy']),
            "precision": float(test_metrics['precision']),
            "recall": float(test_metrics['recall']),
            "f1_score": float(test_metrics['f1']),
            "total_samples": len(y_test),
            "correct_predictions": correct_count,
            "incorrect_predictions": error_count
        },
        "confusion_matrix": {
            "true_negative": int(cm[0, 0]),
            "false_positive": int(cm[0, 1]),
            "false_negative": int(cm[1, 0]),
            "true_positive": int(cm[1, 1])
        }
    }
    
    eval_results_file = output_path / "evaluation_results.json"
    with open(eval_results_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    print(f"Saved evaluation results to: {eval_results_file}")
    
    # Save metrics to text file (summary)
    with open(output_path / "results_summary.txt", "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("SVM + TF-IDF Classification Results\n")
        f.write("="*80 + "\n\n")
        f.write(f"Dataset: Project (Binary Sentiment Classification)\n")
        f.write(f"Train samples: {len(train_df)}\n")
        f.write(f"Test samples: {len(test_df)}\n\n")
        f.write(f"Model: SVM with RBF (Gaussian) kernel\n")
        f.write(f"Parameters: C={C}, gamma={gamma}\n")
        f.write(f"TF-IDF max_features: {max_features}\n")
        f.write(f"PCA components: {pca_components}\n\n")
        f.write("-"*80 + "\n")
        f.write("Train Set Results:\n")
        f.write(f"  Accuracy:  {train_metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {train_metrics['precision']:.4f}\n")
        f.write(f"  Recall:    {train_metrics['recall']:.4f}\n")
        f.write(f"  F1 Score:  {train_metrics['f1']:.4f}\n\n")
        f.write("-"*80 + "\n")
        f.write("Test Set Results:\n")
        f.write(f"  Accuracy:  {test_metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"  Recall:    {test_metrics['recall']:.4f}\n")
        f.write(f"  F1 Score:  {test_metrics['f1']:.4f}\n\n")
        f.write(f"  Total samples: {len(y_test)}\n")
        f.write(f"  Correct predictions: {correct_count}\n")
        f.write(f"  Incorrect predictions: {error_count}\n\n")
        f.write("-"*80 + "\n")
        f.write("\nConfusion Matrix:\n")
        f.write(f"                 Predicted\n")
        f.write(f"               Neg    Pos\n")
        f.write(f"Actual Neg   {cm[0, 0]:5d}  {cm[0, 1]:5d}\n")
        f.write(f"       Pos   {cm[1, 0]:5d}  {cm[1, 1]:5d}\n\n")
        f.write("-"*80 + "\n")
        f.write("\nDetailed Classification Report (Test Set):\n")
        f.write(classification_report(
            y_test,
            y_pred_test,
            target_names=['Negative (0)', 'Positive (1)'],
            digits=4
        ))
        f.write("\n" + "="*80 + "\n")
    
    print(f"Results saved to: {output_path / 'results_summary.txt'}")
    
    return test_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="SVM + TF-IDF for binary sentiment classification"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/usr1/home/s125mdg41_03/code/Intent_Recognition_Exp/data/raw/project",
        help="Path to project dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ee6483/svm_tfidf_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of test set for final evaluation"
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="Maximum number of TF-IDF features"
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=100,
        help="Number of PCA components"
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="SVM regularization parameter"
    )
    parser.add_argument(
        "--gamma",
        type=str,
        default="scale",
        help="SVM kernel coefficient (scale, auto, or float)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip cross-validation, only do final train/test"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*80)
    print("SVM + TF-IDF Binary Sentiment Classification")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Cross-validation folds: {args.n_folds}")
    print(f"  Test size: {args.test_size}")
    print(f"  Max TF-IDF features: {args.max_features}")
    print(f"  PCA components: {args.pca_components}")
    print(f"  SVM C: {args.C}")
    print(f"  SVM gamma: {args.gamma}")
    print(f"  Random seed: {args.seed}")
    
    # Load dataset
    print(f"\n{'='*80}")
    print("Loading Project Dataset")
    print(f"{'='*80}")
    
    raw_dataset, stats = load_raw_project(data_dir=args.data_dir, return_dicts=False)
    train_df, _ = dataset_to_dataframe(raw_dataset)
    
    print(f"\nDataset statistics:")
    print(f"  Total samples: {len(train_df)}")
    print(f"  Number of classes: {stats['num_labels']}")
    print(f"  Class names: {stats['label_names']}")
    print(f"  Class distribution:")
    print(train_df['label'].value_counts().sort_index())
    
    # Text preprocessing
    print(f"\n{'='*80}")
    print("Text Preprocessing")
    print(f"{'='*80}")
    
    train_df = preprocess_dataframe(train_df, text_column='text')
    
    # Remove any rows with empty text after preprocessing
    original_len = len(train_df)
    train_df = train_df[train_df['text'].str.strip().str.len() > 0]
    if len(train_df) < original_len:
        print(f"Removed {original_len - len(train_df)} empty texts after preprocessing")
    
    # Cross-validation
    if not args.skip_cv:
        print(f"\n{'='*80}")
        print("Step 1: Cross-Validation")
        print(f"{'='*80}")
        
        # Build TF-IDF matrix for CV
        vectorizer_cv = TfidfVectorizer(
            max_features=args.max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        
        X_cv = vectorizer_cv.fit_transform(train_df['text']).toarray()
        y_cv = train_df['label'].values
        
        print(f"TF-IDF matrix for CV: {X_cv.shape}")
        
        # Perform cross-validation
        cv_metrics = cross_validate_svm(
            X_cv,
            y_cv,
            n_folds=args.n_folds,
            pca_components=args.pca_components,
            C=args.C,
            gamma=args.gamma,
            random_state=args.seed
        )
    
    # Final train/test evaluation
    print(f"\n{'='*80}")
    print("Step 2: Final Train/Test Evaluation")
    print(f"{'='*80}")
    
    test_metrics = final_train_test_evaluation(
        train_df,
        text_column='text',
        label_column='label',
        test_size=args.test_size,
        pca_components=args.pca_components,
        C=args.C,
        gamma=args.gamma,
        max_features=args.max_features,
        random_state=args.seed,
        output_dir=args.output_dir
    )
    
    print(f"\n{'='*80}")
    print("All tasks completed successfully!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
