#!/usr/bin/env python3


import sys
import logging
from pathlib import Path
import argparse

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pipelines.training import train_flight_price_model

from pycache_handler.handler import py_cache_handler

@py_cache_handler
def main():
    parser = argparse.ArgumentParser(description='Train flight price prediction model')
    parser.add_argument('--data-path', type=str, default='data/google_flights_airfare_data.csv',
                       help='Path to the training data CSV file')
    parser.add_argument('--model-name', type=str, default='xgboost',
                       choices=['xgboost', 'random_forest', 'gradient_boosting', 'ridge'],
                       help='Model type to train')
    parser.add_argument('--tune-hyperparameters', action='store_true',
                       help='Perform hyperparameter tuning')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting model training...")
    
    try:
        # Check if data file exists
        data_path = Path(args.data_path)
        if not data_path.exists():
            logger.error(f"Data file not found: {data_path}")
            return 1
        
        # Train the model
        version, metrics, feature_importance = train_flight_price_model(
            data_path=str(data_path),
            model_name=args.model_name,
            tune_hyperparameters=args.tune_hyperparameters,
            save_model=True
        )
        
        logger.info(f"Model training completed successfully!")
        logger.info(f"Model version: {version}")
        
        # Display metrics
        logger.info("=" * 50)
        logger.info("MODEL PERFORMANCE METRICS")
        logger.info("=" * 50)
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                logger.info(f"{metric_name}: {metric_value:.4f}")
            else:
                logger.info(f"{metric_name}: {metric_value}")
        
        # Display top feature importance
        logger.info("=" * 50)
        logger.info("TOP 10 FEATURE IMPORTANCE")
        logger.info("=" * 50)
        
        # Sort features by importance and display top 10
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_features[:10], 1):
            logger.info(f"{i:2d}. {feature:25s}: {importance:.4f}")
        
        if len(sorted_features) > 10:
            logger.info(f"... and {len(sorted_features) - 10} more features")
        
        logger.info("=" * 50)
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
