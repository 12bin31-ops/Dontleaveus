#!/usr/bin/env python3
"""
Inference script for making predictions on new customer data.

Usage:
    python scripts/run_inference.py --data data/raw/new_customers.csv
    python scripts/run_inference.py --data data/raw/new_customers.csv --model models/best_model.pkl
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.main import ChurnPredictionPipeline
from src.utils import load_model


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on new customer data"
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to new customer data (CSV file)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='models/best_model.pkl',
        help='Path to trained model'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='reports/predictions.csv',
        help='Path to save predictions'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Probability threshold for churn classification'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    print("🚀 Initializing inference pipeline...")
    pipeline = ChurnPredictionPipeline(config_path=args.config)
    
    # Run inference
    print(f"\n🔮 Running inference on: {args.data}")
    results = pipeline.run_inference_pipeline(
        new_data_path=args.data,
        model_path=args.model
    )
    
    # Create results dataframe
    predictions_df = pd.DataFrame({
        'churn_probability': results['probabilities'],
        'churn_prediction': results['predictions'],
        'risk_level': pd.cut(
            results['probabilities'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low', 'Medium', 'High']
        )
    })
    
    # Save predictions
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(args.output, index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("✅ INFERENCE COMPLETE")
    print("="*60)
    
    print(f"\nPredictions Summary:")
    print(f"Total customers: {len(predictions_df)}")
    print(f"\nRisk Distribution:")
    print(predictions_df['risk_level'].value_counts())
    
    print(f"\nChurn Probability Statistics:")
    print(f"  Mean: {results['probabilities'].mean():.4f}")
    print(f"  Min: {results['probabilities'].min():.4f}")
    print(f"  Max: {results['probabilities'].max():.4f}")
    print(f"  Median: {np.median(results['probabilities']):.4f}")
    
    print(f"\nPredictions saved to: {args.output}")
    
    # Show top 10 high-risk customers
    print("\n🚨 Top 10 High-Risk Customers:")
    top_10 = predictions_df.nlargest(10, 'churn_probability')
    print(top_10.to_string())


if __name__ == '__main__':
    main()
