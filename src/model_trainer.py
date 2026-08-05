"""Model training module with hyperparameter tuning."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from datetime import datetime
import json
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import optuna
from optuna.pruners import MedianPruner
from sklearn.model_selection import cross_val_score

from src.utils import save_model, evaluate_model


class ModelTrainer:
    """Train multiple models and perform hyperparameter tuning."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.training_history = []
        
    def get_model_class(self, model_type: str):
        """Get model class by name."""
        model_map = {
            'LogisticRegression': LogisticRegression,
            'RandomForestClassifier': RandomForestClassifier,
            'XGBClassifier': XGBClassifier,
            'LGBMClassifier': LGBMClassifier,
        }
        return model_map.get(model_type)
    
    def train_model(self, model_config: Dict[str, Any], 
                   X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Train a single model."""
        model_type = model_config['type']
        model_name = model_config['name']
        params = model_config.get('params', {})
        
        print(f"\n🤖 Training {model_name}...")
        
        ModelClass = self.get_model_class(model_type)
        model = ModelClass(**params)
        model.fit(X_train, y_train)
        
        print(f"✅ {model_name} training completed")
        return model
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Train all models specified in config."""
        print("\n" + "="*50)
        print("🤖 TRAINING MODELS")
        print("="*50)
        
        results = {}
        
        for model_config in self.config['models']['classifiers']:
            model_name = model_config['name']
            
            try:
                model = self.train_model(model_config, X_train, y_train)
                self.models[model_name] = model
                
                # Store training info
                results[model_name] = {
                    'status': 'success',
                    'type': model_config['type'],
                    'params': model_config.get('params', {}),
                    'timestamp': datetime.now().isoformat()
                }
                
            except Exception as e:
                print(f"❌ Error training {model_name}: {str(e)}")
                results[model_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'results': results
        })
        
        return results
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray, 
                             model_name: str = 'XGBoost') -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training target
            model_name: Model to tune
        
        Returns:
            Best parameters and score
        """
        if not self.config['hyperparameter_tuning']['enabled']:
            print("⚠️ Hyperparameter tuning is disabled in config")
            return None
        
        print(f"\n" + "="*50)
        print(f"🔧 HYPERPARAMETER TUNING: {model_name}")
        print("="*50)
        
        def objective(trial):
            """Objective function for Optuna."""
            if model_name == 'XGBoost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'random_state': 42
                }
                model = XGBClassifier(**params)
                
            elif model_name == 'RandomForest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                    'random_state': 42
                }
                model = RandomForestClassifier(**params)
            
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            
            # Cross-validation score
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
            return scores.mean()
        
        # Create study
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=MedianPruner()
        )
        
        # Optimize
        n_trials = self.config['hyperparameter_tuning']['n_trials']
        timeout = self.config['hyperparameter_tuning']['timeout']
        
        study.optimize(objective, n_trials=n_trials, timeout=timeout, 
                      show_progress_bar=True)
        
        print(f"\n✅ Hyperparameter Tuning Results:")
        print(f"   Best F1 Score: {study.best_value:.4f}")
        print(f"   Best Parameters: {study.best_params}")
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'trials': len(study.trials)
        }
    
    def get_best_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[str, Any, Dict]:
        """Select best model based on test performance."""
        print(f"\n" + "="*50)
        print(f"🏆 SELECTING BEST MODEL")
        print("="*50)
        
        best_score = -1
        best_model_name = None
        best_model = None
        scores_dict = {}
        
        for model_name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            # Use F1 score as primary metric
            metrics = evaluate_model(y_test, y_pred)
            f1 = metrics['f1']
            
            scores_dict[model_name] = metrics
            
            print(f"\n{model_name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            if f1 > best_score:
                best_score = f1
                best_model_name = model_name
                best_model = model
        
        self.best_model = best_model
        self.best_model_name = best_model_name
        
        print(f"\n🎖️ Best Model: {best_model_name} (F1: {best_score:.4f})")
        
        return best_model_name, best_model, scores_dict
    
    def save_best_model(self, filepath: str = "models/best_model.pkl") -> None:
        """Save best model to disk."""
        if self.best_model is None:
            raise ValueError("No best model selected yet!")
        
        save_model(self.best_model, filepath)
        
        # Also save model metadata
        metadata = {
            'model_name': self.best_model_name,
            'model_type': type(self.best_model).__name__,
            'timestamp': datetime.now().isoformat(),
            'config': self.config
        }
        
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"✅ Model metadata saved to {metadata_path}")
    
    def get_models_dict(self) -> Dict[str, Any]:
        """Get all trained models."""
        return self.models
