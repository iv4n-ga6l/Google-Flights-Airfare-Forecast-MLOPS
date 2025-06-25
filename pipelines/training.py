import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
import json

from pipelines.feature_engineering import FeatureEngineer


class ModelTrainer:
    """Model training pipeline for flight price prediction"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        self.feature_engineer = FeatureEngineer()
        self.scaler = StandardScaler()
        self.model = None
        self.model_metadata = {}
        
        # Available models
        self.available_models = {
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'ridge': Ridge(alpha=1.0)
        }
    
    def load_and_prepare_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load and prepare data for training"""
        self.logger.info(f"Loading data from {data_path}")
        
        # Load data
        df = pd.read_csv(data_path)
        self.logger.info(f"Loaded {len(df)} records")
        
        # Feature engineering
        df_features = self.feature_engineer.fit_transform(df)
        
        # Prepare features and target
        target_col = 'price'
        feature_cols = self.feature_engineer.get_feature_names()
        
        # Ensure all feature columns exist
        available_features = [col for col in feature_cols if col in df_features.columns]
        missing_features = [col for col in feature_cols if col not in df_features.columns]
        
        if missing_features:
            self.logger.warning(f"Missing features: {missing_features}")
        
        X = df_features[available_features]
        y = df_features[target_col]
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=pd.qcut(y, q=5, duplicates='drop')
        )
        
        self.logger.info(f"Training set size: {len(X_train)}")
        self.logger.info(f"Test set size: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_name: str = 'xgboost',
        tune_hyperparameters: bool = False
    ) -> None:
        """Train a machine learning model"""
        self.logger.info(f"Training {model_name} model...")
        
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not available. Choose from: {list(self.available_models.keys())}")
        
        model = self.available_models[model_name]
        
        # Scale features for linear models
        if model_name in ['ridge', 'linear']:
            X_train_scaled = self.scaler.fit_transform(X_train)
            model.fit(X_train_scaled, y_train)
        else:
            model.fit(X_train, y_train)
        
        # Hyperparameter tuning
        if tune_hyperparameters:
            self.logger.info("Performing hyperparameter tuning...")
            model = self._tune_hyperparameters(model, X_train, y_train, model_name)
        
        self.model = model
        self.model_name = model_name
        
        # Cross-validation
        if model_name in ['ridge', 'linear']:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        
        self.logger.info(f"Cross-validation MAE: {-cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
    
    def _tune_hyperparameters(self, model, X_train, y_train, model_name):
        """Perform hyperparameter tuning"""
        param_grids = {
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'ridge': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            }
        }
        
        if model_name in param_grids:
            grid_search = GridSearchCV(
                model, param_grids[model_name],
                cv=3, scoring='neg_mean_absolute_error',
                n_jobs=-1, verbose=1
            )
            
            if model_name in ['ridge', 'linear']:
                X_train_scaled = self.scaler.transform(X_train)
                grid_search.fit(X_train_scaled, y_train)
            else:
                grid_search.fit(X_train, y_train)
            
            self.logger.info(f"Best parameters: {grid_search.best_params_}")
            return grid_search.best_estimator_
        
        return model
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate the trained model"""
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        if self.model_name in ['ridge', 'linear']:
            X_test_scaled = self.scaler.transform(X_test)
            y_pred = self.model.predict(X_test_scaled)
        else:
            y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        self.logger.info("Model evaluation metrics:")
        for metric, value in metrics.items():
            self.logger.info(f"{metric.upper()}: {value:.4f}")
        
        return metrics
    
    def get_feature_importance(self, X_train: pd.DataFrame) -> Dict[str, float]:
        """Get feature importance from the trained model"""
        if self.model is None:
            raise ValueError("Model must be trained before getting feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            feature_names = X_train.columns
            
            feature_importance = dict(zip(feature_names, importance))
            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
            return feature_importance
        else:
            self.logger.warning("Model does not support feature importance")
            return {}
    
    def save_model(self, version: str = None) -> str:
        """Save the trained model and metadata"""
        if self.model is None:
            raise ValueError("Model must be trained before saving")
        
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_path = self.model_dir / f"flight_price_model_{version}.joblib"
        scaler_path = self.model_dir / f"scaler_{version}.joblib"
        feature_engineer_path = self.model_dir / f"feature_engineer_{version}.joblib"
        metadata_path = self.model_dir / f"model_metadata_{version}.json"
        
        # Save model components
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.feature_engineer, feature_engineer_path)
        
        # Save metadata
        self.model_metadata = {
            'version': version,
            'model_name': self.model_name,
            'training_timestamp': datetime.now().isoformat(),
            'model_path': str(model_path),
            'scaler_path': str(scaler_path),
            'feature_engineer_path': str(feature_engineer_path)
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(self.model_metadata, f, indent=2)
        
        self.logger.info(f"Model saved with version {version}")
        return version
    
    def load_model(self, version: str) -> None:
        """Load a saved model"""
        model_path = self.model_dir / f"flight_price_model_{version}.joblib"
        scaler_path = self.model_dir / f"scaler_{version}.joblib"
        feature_engineer_path = self.model_dir / f"feature_engineer_{version}.joblib"
        metadata_path = self.model_dir / f"model_metadata_{version}.json"
        
        if not all(path.exists() for path in [model_path, scaler_path, feature_engineer_path, metadata_path]):
            raise FileNotFoundError(f"Model version {version} not found")
        
        # Load model components
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_engineer = joblib.load(feature_engineer_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.model_metadata = json.load(f)
        
        self.model_name = self.model_metadata.get('model_name', 'unknown')
        self.logger.info(f"Model version {version} loaded successfully")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model must be trained or loaded before making predictions")
        
        # Apply feature engineering
        X_features = self.feature_engineer.transform(X)
        
        # Ensure features match training
        feature_cols = self.feature_engineer.get_feature_names()
        available_features = [col for col in feature_cols if col in X_features.columns]
        X_features = X_features[available_features]
        
        # Handle missing values
        X_features = X_features.fillna(X_features.mean())
        
        # Scale if needed
        if self.model_name in ['ridge', 'linear']:
            X_features = self.scaler.transform(X_features)
        
        predictions = self.model.predict(X_features)
        return predictions
    
    def get_prediction_confidence(self, X: pd.DataFrame, confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Get prediction confidence intervals (for tree-based models)"""
        predictions = self.predict(X)
        
        # For ensemble models, we can use prediction variance
        if hasattr(self.model, 'estimators_'):
            # Get predictions from all estimators
            estimator_predictions = np.array([
                estimator.predict(X) for estimator in self.model.estimators_
            ])
            
            # Calculate confidence intervals
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower_bound = np.percentile(estimator_predictions, lower_percentile, axis=0)
            upper_bound = np.percentile(estimator_predictions, upper_percentile, axis=0)
            
            return lower_bound, upper_bound
        else:
            # For non-ensemble models, use a simple heuristic
            std_error = np.std(predictions) * 0.1  # Rough estimate
            margin = 1.96 * std_error  # 95% confidence interval
            
            lower_bound = predictions - margin
            upper_bound = predictions + margin
            
            return lower_bound, upper_bound


def train_flight_price_model(
    data_path: str,
    model_name: str = 'xgboost',
    tune_hyperparameters: bool = False,
    save_model: bool = True
) -> Tuple[Optional[str], Dict, Dict]:
    """Complete training pipeline
    
    Returns:
        tuple: (version, metrics, feature_importance)
            - version: Model version string (or None if not saved)
            - metrics: Dictionary with model evaluation metrics
            - feature_importance: Dictionary with feature importance scores
    """
    logging.basicConfig(level=logging.INFO)
    
    trainer = ModelTrainer()
    
    # Load and prepare data
    X_train, X_test, y_train, y_test = trainer.load_and_prepare_data(data_path)
    
    # Train model
    trainer.train_model(X_train, y_train, model_name, tune_hyperparameters)
    
    # Evaluate model
    metrics = trainer.evaluate_model(X_test, y_test)
    
    # Get feature importance
    feature_importance = trainer.get_feature_importance(X_train)
    
    # Save model
    version = None
    if save_model:
        version = trainer.save_model()
    
    return version, metrics, feature_importance
