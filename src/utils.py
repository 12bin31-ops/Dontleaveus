"""Utility functions for the churn prediction pipeline."""

import logging
import yaml
import pickle
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Set up logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, config['logging']['level']))
    
    # File handler
    fh = logging.FileHandler(config['logging']['log_file'])
    fh.setLevel(getattr(logging, config['logging']['level']))
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, config['logging']['level']))
    
    # Formatter
    formatter = logging.Formatter(config['logging']['format'])
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def save_model(model: Any, filepath: str) -> None:
    """Save trained model to disk."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to {filepath}")


def load_model(filepath: str) -> Any:
    """Load trained model from disk."""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Model loaded from {filepath}")
    return model


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, 
                   y_pred_proba: np.ndarray = None) -> Dict[str, float]:
    """
    Evaluate model performance with multiple metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (for ROC-AUC)
    
    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics


def print_model_report(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> None:
    """Print detailed classification report."""
    print(f"\n{'='*50}")
    print(f"📊 {model_name} Performance Report")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, target_names=['No Churn', 'Churn']))
    

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> plt.Figure:
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    ax.set_title(f"{model_name} - Confusion Matrix")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    
    return fig


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, model_name: str = "Model") -> plt.Figure:
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{model_name} - ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def save_metrics_json(metrics: Dict[str, Any], filepath: str) -> None:
    """Save metrics to JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"✅ Metrics saved to {filepath}")


def load_metrics_json(filepath: str) -> Dict[str, Any]:
    """Load metrics from JSON file."""
    with open(filepath, 'r') as f:
        metrics = json.load(f)
    return metrics
