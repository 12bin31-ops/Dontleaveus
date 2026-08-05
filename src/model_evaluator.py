"""Model evaluation and explainability module."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple
from pathlib import Path

import shap
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc, auc_oc_score
)


class ModelEvaluator:
    """Evaluate model performance and provide explainability insights."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.evaluation_results = {}
        self.shap_values = None
        self.shap_explainer = None
        
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                      model_name: str = "Model") -> Dict[str, float]:
        """
        Evaluate model with multiple metrics.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name for logging
        
        Returns:
            Dictionary of metrics
        """
        y_pred = model.predict(X_test)
        
        # Probability predictions for ROC-AUC
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = model.decision_function(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        self.evaluation_results[model_name] = metrics
        
        return metrics
    
    def get_feature_importance(self, model: Any, feature_names: list,
                              top_n: int = 15) -> pd.DataFrame:
        """
        Extract feature importance from model.
        
        Supports tree-based models (Random Forest, XGBoost, LightGBM).
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            raise ValueError("Model doesn't support feature importance extraction")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def plot_feature_importance(self, model: Any, feature_names: list,
                               top_n: int = 15, figsize: Tuple = (10, 6)) -> plt.Figure:
        """Plot top N important features."""
        importance_df = self.get_feature_importance(model, feature_names, top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(range(len(importance_df)), importance_df['importance'])
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'])
        ax.set_xlabel('Importance')
        ax.set_title('Top Features by Importance')
        ax.invert_yaxis()
        
        return fig
    
    def explain_with_shap(self, model: Any, X_test: np.ndarray, 
                         X_train: np.ndarray = None, sample_size: int = 100) -> None:
        """
        Generate SHAP explanations for model predictions.
        
        Args:
            model: Trained model
            X_test: Test features to explain
            X_train: Training data (for background distribution)
            sample_size: Number of samples to use for SHAP
        """
        print("\n" + "="*50)
        print("🔍 GENERATING SHAP EXPLANATIONS")
        print("="*50)
        
        # Use training data as background if provided
        if X_train is not None and len(X_train) > sample_size:
            background = X_train[:sample_size]
        else:
            background = X_test[:sample_size] if len(X_test) > sample_size else X_test
        
        # Create explainer
        try:
            self.shap_explainer = shap.TreeExplainer(model)
        except:
            try:
                self.shap_explainer = shap.KernelExplainer(
                    model.predict, 
                    shap.sample(background, min(50, len(background)))
                )
            except Exception as e:
                print(f"⚠️ Could not create SHAP explainer: {str(e)}")
                return
        
        # Calculate SHAP values
        self.shap_values = self.shap_explainer.shap_values(X_test[:sample_size])
        print(f"✅ SHAP values computed for {sample_size} test samples")
    
    def plot_shap_summary(self, X_test: pd.DataFrame, figsize: Tuple = (12, 8)) -> plt.Figure:
        """Plot SHAP summary plot."""
        if self.shap_values is None:
            raise ValueError("Run explain_with_shap first!")
        
        # Handle binary classification (SHAP returns list of arrays)
        if isinstance(self.shap_values, list):
            shap_values = self.shap_values[1]  # Use positive class
        else:
            shap_values = self.shap_values
        
        fig = plt.figure(figsize=figsize)
        shap.summary_plot(shap_values, X_test, show=False)
        
        return fig
    
    def plot_confusion_matrix(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                             figsize: Tuple = (8, 6)) -> plt.Figure:
        """Plot confusion matrix."""
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'])
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        return fig
    
    def plot_roc_curve(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                      figsize: Tuple = (8, 6)) -> plt.Figure:
        """Plot ROC curve."""
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = model.decision_function(X_test)
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def print_classification_report(self, model: Any, X_test: np.ndarray, 
                                   y_test: np.ndarray) -> str:
        """Print detailed classification report."""
        y_pred = model.predict(X_test)
        
        report = classification_report(y_test, y_pred, 
                                      target_names=['No Churn', 'Churn'])
        print("\n" + "="*50)
        print("📊 CLASSIFICATION REPORT")
        print("="*50)
        print(report)
        
        return report
    
    def save_evaluation_report(self, filepath: str) -> None:
        """Save evaluation results to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(filepath, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        print(f"✅ Evaluation report saved to {filepath}")
