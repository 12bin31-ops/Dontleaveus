#!/usr/bin/env python3
"""
Training script for customer churn prediction model.

Usage:
    python scripts/train_model.py --data data/raw/churn_data.csv
    python scripts/train_model.py --config config/config.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.main import ChurnPredictionPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Train customer churn prediction model"
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='data/raw/churn_data.csv',
        help='Path to training data (CSV file)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--output-model',
        type=str,
        default='models/best_model.pkl',
        help='Path to save trained model'
    )
    
    parser.add_argument(
        '--generate-report',
        action='store_true',
        help='Generate HTML/PDF report after training'
    )
    
    parser.add_argument(
        '--setup-scheduler',
        action='store_true',
        help='Setup automated scheduling'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    print("🚀 Initializing pipeline...")
    pipeline = ChurnPredictionPipeline(config_path=args.config)
    
    # Run training
    print(f"\n📊 Training on data: {args.data}")
    results = pipeline.run_training_pipeline(data_path=args.data)
    
    # Print results
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print("="*60)
    print(f"\nBest Model: {results['best_model_name']}")
    print(f"Model Metrics:")
    for metric, value in results['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nModel saved to: {args.output_model}")
    print(f"Report saved to: {results['report_path']}")
    
    # Setup scheduler if requested
    if args.setup_scheduler:
        print("\n⏰ Setting up automated scheduler...")
        pipeline.setup_automated_pipeline()
        print("✅ Scheduler configured - press Ctrl+C to stop")
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️ Scheduler stopped")


if __name__ == '__main__':
    main()
