from typing import Tuple, Optional, Union
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def df_to_embedding_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name_or_path: str = '/Users/wangshibo/Documents/Academic/Course/6405Project/Intent_Recognition_Exp/models/huggingface_models/bilingual_embedding_small_lajavaness',
    text_column: str = "text",
    label_column: str = "label",
    batch_size: int = 32,
    normalize_embeddings: bool = True,
    show_progress: bool = True,
    trust_remote_code: bool = True,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert DataFrame text to sentence embeddings for sklearn models.
    
    Args:
        train_df: Training DataFrame with text and label columns.
        test_df: Test DataFrame with text and label columns.
        model_name_or_path: Path to pre-trained SentenceTransformer model or HF model name.
                           Default: "Lajavaness/bilingual-embedding-small".
        text_column: Name of the text column (default: "text").
        label_column: Name of the label column (default: "label").
        batch_size: Batch size for encoding. Larger batch = faster but more memory.
                   Default: 32. Adjust based on GPU memory:
                   - CPU: 16-32
                   - GPU (8GB): 32-64
                   - GPU (16GB+): 64-128
        normalize_embeddings: Whether to normalize embeddings to unit length.
                             Recommended for cosine similarity. Default: True.
        show_progress: Show progress bar during encoding. Default: True.
        trust_remote_code: Trust remote code for custom models. Default: True.
        device: Device to use ('cuda', 'cpu', or None for auto-detect).
                If None, automatically uses GPU if available.
    
    Returns:
        X_train: Embeddings for training set (numpy array, shape: [n_train, embedding_dim]).
        y_train: Labels for training set (numpy array).
        X_test: Embeddings for test set (numpy array, shape: [n_test, embedding_dim]).
        y_test: Labels for test set (numpy array).
    
    Example:
        >>> # Basic usage with default model
        >>> X_train, y_train, X_test, y_test = df_to_embedding_features(
        ...     train_df, test_df
        ... )
        
        >>> # Use custom model with larger batch size
        >>> X_train, y_train, X_test, y_test = df_to_embedding_features(
        ...     train_df, test_df,
        ...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
        ...     batch_size=64
        ... )
        
        >>> # Use local model path
        >>> X_train, y_train, X_test, y_test = df_to_embedding_features(
        ...     train_df, test_df,
        ...     model_name_or_path="/path/to/local/model",
        ...     batch_size=16,
        ...     device="cpu"
        ... )
    """
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[+] Loading SentenceTransformer model: {model_name_or_path}")
    print(f"    - Device: {device}")
    print(f"    - Batch size: {batch_size}")
    print(f"    - Normalize embeddings: {normalize_embeddings}")
    
    # Load model
    model = SentenceTransformer(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        device=device
    )
    
    # Get embedding dimension
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"    - Embedding dimension: {embedding_dim}")
    
    # Extract text and labels
    train_texts = train_df[text_column].tolist()
    train_labels = train_df[label_column].values
    test_texts = test_df[text_column].tolist()
    test_labels = test_df[label_column].values
    
    print(f"\n[+] Encoding training data ({len(train_texts)} samples)...")
    X_train = model.encode(
        train_texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=normalize_embeddings,
        convert_to_numpy=True,
    )
    
    print(f"[+] Encoding test data ({len(test_texts)} samples)...")
    X_test = model.encode(
        test_texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=normalize_embeddings,
        convert_to_numpy=True,
    )
    
    y_train = train_labels
    y_test = test_labels
    
    print(f"\n[✓] Encoding complete!")
    print(f"    - Train embeddings shape: {X_train.shape}")
    print(f"    - Test embeddings shape: {X_test.shape}")
    print(f"    - Train labels shape: {y_train.shape}")
    print(f"    - Test labels shape: {y_test.shape}")
    print(f"    - Memory usage (train): {X_train.nbytes / 1024 / 1024:.2f} MB")
    print(f"    - Memory usage (test): {X_test.nbytes / 1024 / 1024:.2f} MB")
    
    return X_train, y_train, X_test, y_test


def df_to_embedding_features_manual_batching(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name_or_path: str = "Lajavaness/bilingual-embedding-small",
    text_column: str = "text",
    label_column: str = "label",
    batch_size: int = 32,
    normalize_embeddings: bool = True,
    trust_remote_code: bool = True,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert DataFrame text to sentence embeddings with manual batch control.
    This version provides more explicit control over batching and memory usage.
    
    Args:
        Same as df_to_embedding_features.
    
    Returns:
        Same as df_to_embedding_features.
    """
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[+] Loading SentenceTransformer model: {model_name_or_path}")
    print(f"    - Device: {device}")
    print(f"    - Batch size: {batch_size}")
    
    # Load model
    model = SentenceTransformer(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        device=device
    )
    
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"    - Embedding dimension: {embedding_dim}")
    
    # Extract text and labels
    train_texts = train_df[text_column].tolist()
    train_labels = train_df[label_column].values
    test_texts = test_df[text_column].tolist()
    test_labels = test_df[label_column].values
    
    # Manual batching for training data
    print(f"\n[+] Encoding training data ({len(train_texts)} samples) with manual batching...")
    train_embeddings = []
    num_train_batches = (len(train_texts) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(train_texts), batch_size), 
                  total=num_train_batches, 
                  desc="Train batches"):
        batch_texts = train_texts[i:i + batch_size]
        batch_embeddings = model.encode(
            batch_texts,
            batch_size=len(batch_texts),
            show_progress_bar=False,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
        )
        train_embeddings.append(batch_embeddings)
        
        # Clear GPU cache if using CUDA
        if device == "cuda":
            torch.cuda.empty_cache()
    
    X_train = np.vstack(train_embeddings)
    
    # Manual batching for test data
    print(f"[+] Encoding test data ({len(test_texts)} samples) with manual batching...")
    test_embeddings = []
    num_test_batches = (len(test_texts) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(test_texts), batch_size), 
                  total=num_test_batches, 
                  desc="Test batches"):
        batch_texts = test_texts[i:i + batch_size]
        batch_embeddings = model.encode(
            batch_texts,
            batch_size=len(batch_texts),
            show_progress_bar=False,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
        )
        test_embeddings.append(batch_embeddings)
        
        # Clear GPU cache if using CUDA
        if device == "cuda":
            torch.cuda.empty_cache()
    
    X_test = np.vstack(test_embeddings)
    
    y_train = train_labels
    y_test = test_labels
    
    print(f"\n[✓] Encoding complete!")
    print(f"    - Train embeddings shape: {X_train.shape}")
    print(f"    - Test embeddings shape: {X_test.shape}")
    print(f"    - Memory usage (train): {X_train.nbytes / 1024 / 1024:.2f} MB")
    
    return X_train, y_train, X_test, y_test


def get_embedding_model_info(
    model_name_or_path: str,
    trust_remote_code: bool = True,
) -> dict:
    """
    Get information about an embedding model without loading it fully.
    
    Args:
        model_name_or_path: Path or name of the model.
        trust_remote_code: Trust remote code.
    
    Returns:
        Dictionary with model information.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name_or_path, trust_remote_code=trust_remote_code, device=device)
    
    return {
        "model_name": model_name_or_path,
        "embedding_dim": model.get_sentence_embedding_dimension(),
        "max_seq_length": model.max_seq_length,
        "device": device,
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }


if __name__ == "__main__":
    # Example usage and testing
    import sys
    sys.path.append("/Users/wangshibo/Documents/Academic/Course/6405Project/Intent_Recognition_Exp")
    
    from data.loaders.load_raw_data import load_raw_atis, dataset_to_dataframe
    
    print("="*80)
    print("Sentence Embedding Feature Extraction Example")
    print("="*80)
    
    # Check GPU availability
    print(f"\n[0] GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    - GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"    - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load dataset
    print("\n[1] Loading dataset...")
    dataset, dataset_stats = load_raw_atis()
    train_df, test_df = dataset_to_dataframe(dataset)
    
    # Take a small subset for testing
    # train_df_small = train_df.head(100)
    train_df_small = train_df
    # test_df_small = test_df.head(50)
    test_df_small = test_df
    
    print(f"    Train size: {len(train_df_small)}, Test size: {len(test_df_small)}")
    print(f"    Number of labels: {dataset_stats['num_labels']}")
    
    # Get model info
    print("\n" + "="*80)
    print("[2] Model Information")
    print("="*80)
    model_path = "/Users/wangshibo/Documents/Academic/Course/6405Project/Intent_Recognition_Exp/models/huggingface_models/bilingual_embedding_small_lajavaness"
    try:
        info = get_embedding_model_info(model_path)
        for key, value in info.items():
            print(f"    - {key}: {value}")
    except Exception as e:
        print(f"    ! Could not load model info: {e}")
        print("    ! Using HuggingFace Hub model name instead")
        model_path = "Lajavaness/bilingual-embedding-small"
    
    # Extract embeddings - Standard method
    print("\n" + "="*80)
    print("[3] Standard Encoding (Built-in Batching)")
    print("="*80)
    X_train, y_train, X_test, y_test = df_to_embedding_features(
        train_df_small, 
        test_df_small,
        model_name_or_path=model_path,
        batch_size=32,
        normalize_embeddings=True,
    )
    
    # Extract embeddings - Manual batching
    print("\n" + "="*80)
    print("[4] Manual Batching (More Control)")
    print("="*80)
    X_train2, y_train2, X_test2, y_test2 = df_to_embedding_features_manual_batching(
        train_df_small,
        test_df_small,
        model_name_or_path=model_path,
        batch_size=16,
        normalize_embeddings=True,
    )
    
    # Verify embeddings
    print("\n" + "="*80)
    print("[5] Embedding Statistics")
    print("="*80)
    print(f"Data type: {X_train.dtype}")
    print(f"Is normalized: {np.allclose(np.linalg.norm(X_train, axis=1), 1.0)}")
    print(f"Mean norm: {np.mean(np.linalg.norm(X_train, axis=1)):.4f}")
    print(f"Min value: {X_train.min():.4f}, Max value: {X_train.max():.4f}")
    print(f"Embeddings match (standard vs manual): {np.allclose(X_train, X_train2, atol=1e-5)}")
    
    # Test with sklearn model
    print("\n" + "="*80)
    print("[6] Quick sklearn Model Test")
    print("="*80)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Logistic Regression Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report (first 5 classes):")
    print(classification_report(y_test, y_pred, 
                                labels=list(range(min(5, len(np.unique(y_test))))),
                                target_names=[str(i) for i in range(min(5, len(np.unique(y_test))))]))
    
    print("\n" + "="*80)
    print("✓ Embedding extraction completed successfully!")
    print("="*80)
