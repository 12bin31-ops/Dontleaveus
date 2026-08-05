"""
Main pipeline orchestration.

This module coordinates:
1. Data loading and preprocessing
2. Model training and evaluation
3. Prediction on new data
4. LLM-based insight generation
5. Automated report generation
6. Scheduling for continuous operation
"""

import sys
from pathlib import Path
import logging
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, setup_logging, save_model, load_model
from src.data_loader import DataLoader
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator
from src.insight_generator import InsightGenerator
from src.report_generator import ReportGenerator
from pipeline.scheduler import PipelineScheduler, ErrorHandler


class ChurnPredictionPipeline:
    """End-to-end churn prediction pipeline."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize pipeline."""
        self.config = load_config(config_path)
        self.logger = setup_logging(self.config)
        
        self.logger.info("🚀 Initializing Churn Prediction Pipeline")
        
        # Initialize components
        self.data_loader = DataLoader(self.config)
        self.model_trainer = ModelTrainer(self.config)
        self.model_evaluator = ModelEvaluator(self.config)
        self.insight_generator = InsightGenerator(self.config)
        self.report_generator = ReportGenerator(self.config)
        self.scheduler = PipelineScheduler(self.config)
        self.error_handler = ErrorHandler(self.logger)
    
    def run_training_pipeline(self, data_path: str = None) -> Dict[str, Any]:
        """
        Run complete training pipeline.
        
        Steps:
        1. Load and explore data
        2. Preprocess data
        3. Split train/test
        4. Train models
        5. Evaluate models
        6. Generate insights
        7. Create report
        """
        print("\n" + "="*60)
        print("📊 RUNNING TRAINING PIPELINE")
        print("="*60)
        
        if data_path is None:
            data_path = self.config['data']['source']
        
        # Step 1: Load data
        self.logger.info(f"Loading data from {data_path}")
        self.data_loader.load_data(data_path)
        self.data_loader.explore_data()
        
        # Step 2: Preprocess
        self.logger.info("Preprocessing data")
        X, y = self.data_loader.preprocess()
        
        # Step 3: Train/test split
        self.logger.info("Splitting data")
        self.data_loader.train_test_split(X, y)
        X_train, X_test, y_train, y_test = self.data_loader.get_train_test_data()
        
        # Step 4: Train models
        self.logger.info("Training models")
        training_results = self.model_trainer.train_all_models(X_train, y_train)
        
        # Step 5: Hyperparameter tuning (optional)
        if self.config['hyperparameter_tuning']['enabled']:
            self.logger.info("Performing hyperparameter tuning")
            try:
                tuning_results = self.model_trainer.hyperparameter_tuning(
                    X_train, y_train, model_name='XGBoost'
                )
            except Exception as e:
                self.logger.warning(f"Hyperparameter tuning failed: {str(e)}")
        
        # Step 6: Evaluate and select best model
        self.logger.info("Evaluating models")
        best_model_name, best_model, scores = self.model_trainer.get_best_model(X_test, y_test)
        
        # Step 7: Get feature importance
        self.logger.info("Extracting feature importance")
        feature_importance = self.model_evaluator.get_feature_importance(
            best_model, 
            self.data_loader.get_feature_names()
        )
        
        # Step 8: SHAP explanations
        if self.config['evaluation']['use_shap']:
            self.logger.info("Generating SHAP explanations")
            try:
                self.model_evaluator.explain_with_shap(best_model, X_test, X_train)
            except Exception as e:
                self.logger.warning(f"SHAP explanation failed: {str(e)}")
        
        # Step 9: Generate LLM insights
        self.logger.info("Generating LLM-based insights")
        y_pred = best_model.predict(X_test)
        metrics = self.model_evaluator.evaluate_model(best_model, X_test, y_test)
        
        insights = self.insight_generator.batch_generate_insights(
            metrics, 
            feature_importance,
            high_risk_customers=None,  # Generate from predictions below
            feature_names=self.data_loader.get_feature_names()
        )
        
        # Step 10: Identify high-risk customers
        if hasattr(best_model, 'predict_proba'):
            risk_scores = best_model.predict_proba(X_test)[:, 1]
        else:
            risk_scores = best_model.decision_function(X_test)
        
        high_risk_indices = risk_scores.argsort()[-10:][::-1]
        high_risk_customers = [(i, risk_scores[i]) for i in high_risk_indices]
        
        # Step 11: Generate visualizations
        self.logger.info("Generating visualizations")
        charts = {}
        try:
            charts['Feature Importance'] = self.model_evaluator.plot_feature_importance(
                best_model, self.data_loader.get_feature_names()
            )
            charts['Confusion Matrix'] = self.model_evaluator.plot_confusion_matrix(
                best_model, X_test, y_test
            )
            charts['ROC Curve'] = self.model_evaluator.plot_roc_curve(
                best_model, X_test, y_test
            )
        except Exception as e:
            self.logger.warning(f"Chart generation failed: {str(e)}")
        
        # Step 12: Generate report
        self.logger.info("Generating report")
        report_path = self.report_generator.generate_report(
            metrics=metrics,
            feature_importance=feature_importance,
            high_risk_customers=high_risk_customers,
            insights=insights,
            charts=charts
        )
        
        # Step 13: Save model
        self.logger.info("Saving best model")
        self.model_trainer.save_best_model("models/best_model.pkl")
        
        self.logger.info("✅ Training pipeline completed successfully")
        
        return {
            'best_model_name': best_model_name,
            'best_model': best_model,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'insights': insights,
            'report_path': report_path,
            'high_risk_customers': high_risk_customers
        }
    
    def run_inference_pipeline(self, new_data_path: str, model_path: str = "models/best_model.pkl") -> Dict[str, Any]:
        """
        Run inference on new data and generate predictions.
        
        Args:
            new_data_path: Path to new customer data
            model_path: Path to trained model
        """
        print("\n" + "="*60)
        print("🔮 RUNNING INFERENCE PIPELINE")
        print("="*60)
        
        self.logger.info(f"Loading new data from {new_data_path}")
        
        # Load new data
        self.data_loader.load_data(new_data_path)
        X_new, _ = self.data_loader.preprocess()
        
        # Load model
        self.logger.info(f"Loading model from {model_path}")
        model = load_model(model_path)
        
        # Make predictions
        self.logger.info("Generating predictions")
        predictions = model.predict(X_new)
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_new)[:, 1]
        else:
            probabilities = model.decision_function(X_new)
        
        self.logger.info("✅ Inference completed")
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'X_new': X_new
        }
    
    def setup_automated_pipeline(self) -> None:
        """Setup automated scheduling for continuous operation."""
        print("\n" + "="*60)
        print("⏰ SETTING UP AUTOMATED SCHEDULING")
        print("="*60)
        
        # Wrap jobs with error handling
        retrain_job = self.error_handler.wrap_job(
            lambda: self.run_training_pipeline()
        )
        
        # Schedule jobs
        self.scheduler.schedule_retrain(retrain_job)
        self.scheduler.schedule_inference(
            self.error_handler.wrap_job(
                lambda: self.run_inference_pipeline("data/raw/new_customers.csv")
            )
        )
        self.scheduler.schedule_report_generation(
            self.error_handler.wrap_job(
                lambda: print("✅ Report generation job would run here")
            )
        )
        
        # Start scheduler
        self.scheduler.start()
        self.logger.info("✅ Automated scheduling configured")
    
    def run_once(self) -> None:
        """Run training pipeline once (for manual execution)."""
        self.run_training_pipeline()
        self.logger.info("✅ Pipeline execution complete")


def main():
    """Main entry point."""
    pipeline = ChurnPredictionPipeline()
    
    # Run training pipeline
    results = pipeline.run_training_pipeline()
    
    print("\n" + "="*60)
    print("✅ PIPELINE EXECUTION COMPLETE")
    print("="*60)
    print(f"\nBest Model: {results['best_model_name']}")
    print(f"F1 Score: {results['metrics']['f1']:.4f}")
    print(f"Report saved to: {results['report_path']}")


if __name__ == "__main__":
    main()
