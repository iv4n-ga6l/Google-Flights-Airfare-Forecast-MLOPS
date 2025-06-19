import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
import logging
from pathlib import Path

from pipelines.feature_engineering import create_features_for_prediction
from pipelines.training import ModelTrainer
from app.models import (
    FlightPredictionRequest, 
    FlightPredictionResponse,
    ScenarioPlanningRequest,
    ScenarioPlanningResponse,
    BookingScenario
)


class FlightPricePredictor:
    """Inference pipeline for flight price prediction"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.logger = logging.getLogger(__name__)
        
        self.model_trainer = ModelTrainer(model_dir)
        self.current_version = None
        self.model_loaded = False
    
    def load_latest_model(self) -> str:
        """Load the latest trained model"""
        try:
            # Find all model metadata files
            metadata_files = list(self.model_dir.glob("model_metadata_*.json"))
            
            if not metadata_files:
                raise FileNotFoundError("No trained models found")
            
            # Get the latest model based on timestamp
            latest_metadata_file = max(metadata_files, key=lambda x: x.stat().st_mtime)
            
            # Extract version from filename
            version = latest_metadata_file.stem.replace("model_metadata_", "")
            
            # Load the model
            self.model_trainer.load_model(version)
            self.current_version = version
            self.model_loaded = True
            
            self.logger.info(f"Loaded model version: {version}")
            return version
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def load_model_version(self, version: str) -> None:
        """Load a specific model version"""
        try:
            self.model_trainer.load_model(version)
            self.current_version = version
            self.model_loaded = True
            self.logger.info(f"Loaded model version: {version}")
        except Exception as e:
            self.logger.error(f"Failed to load model version {version}: {str(e)}")
            raise
    
    def predict_price(self, request: FlightPredictionRequest) -> FlightPredictionResponse:
        """Predict flight price for a single request"""
        if not self.model_loaded:
            raise ValueError("Model not loaded. Call load_latest_model() or load_model_version() first.")
        
        try:
            # Create features DataFrame
            features_df = create_features_for_prediction(
                airline=request.airline,
                origin=request.origin,
                destination=request.destination,
                booking_date=request.booking_date,
                departure_date=request.departure_date,
                fare_class=request.fare_class.value
            )
            
            # Make prediction
            predicted_price = self.model_trainer.predict(features_df)[0]
            
            # Get confidence interval
            lower_bound, upper_bound = self.model_trainer.get_prediction_confidence(features_df)
            confidence_interval = (float(lower_bound[0]), float(upper_bound[0]))
            
            # Create response
            response = FlightPredictionResponse(
                predicted_price=float(predicted_price),
                confidence_interval=confidence_interval,
                booking_date=request.booking_date,
                departure_date=request.departure_date,
                route=f"{request.origin}-{request.destination}",
                airline=request.airline,
                fare_class=request.fare_class.value,
                model_version=self.current_version,
                prediction_timestamp=datetime.now()
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def scenario_planning(self, request: ScenarioPlanningRequest) -> ScenarioPlanningResponse:
        """Perform scenario planning for multiple booking dates"""
        if not self.model_loaded:
            raise ValueError("Model not loaded. Call load_latest_model() or load_model_version() first.")
        
        try:
            scenarios = []
            predictions = []
            
            # Get predictions for all booking dates
            for booking_date in request.booking_dates:
                features_df = create_features_for_prediction(
                    airline=request.airline,
                    origin=request.origin,
                    destination=request.destination,
                    booking_date=booking_date,
                    departure_date=request.departure_date,
                    fare_class=request.fare_class.value
                )
                
                predicted_price = self.model_trainer.predict(features_df)[0]
                predictions.append(predicted_price)
                
                # Calculate days before departure
                days_before = (request.departure_date - booking_date).days
                
                scenarios.append({
                    'booking_date': booking_date,
                    'predicted_price': float(predicted_price),
                    'days_before_departure': days_before
                })
            
            # Calculate baseline (current date or first booking date)
            baseline_price = predictions[0] if predictions else 0
            
            # Calculate savings relative to baseline
            scenario_objects = []
            for scenario, price in zip(scenarios, predictions):
                potential_savings = baseline_price - price
                
                booking_scenario = BookingScenario(
                    booking_date=scenario['booking_date'],
                    predicted_price=scenario['predicted_price'],
                    days_before_departure=scenario['days_before_departure'],
                    potential_savings=float(potential_savings)
                )
                scenario_objects.append(booking_scenario)
            
            # Find best and worst booking dates
            best_scenario = min(scenarios, key=lambda x: x['predicted_price'])
            worst_scenario = max(scenarios, key=lambda x: x['predicted_price'])
            
            max_savings = worst_scenario['predicted_price'] - best_scenario['predicted_price']
            
            # Create response
            response = ScenarioPlanningResponse(
                scenarios=scenario_objects,
                best_booking_date=best_scenario['booking_date'],
                worst_booking_date=worst_scenario['booking_date'],
                max_savings=float(max_savings),
                departure_date=request.departure_date,
                route=f"{request.origin}-{request.destination}",
                airline=request.airline,
                baseline_price=float(baseline_price),
                analysis_timestamp=datetime.now()
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Scenario planning failed: {str(e)}")
            raise
    
    def generate_booking_date_recommendations(
        self,
        airline: str,
        origin: str,
        destination: str,
        departure_date: datetime,
        fare_class: str = "Economy",
        days_range: int = 60
    ) -> ScenarioPlanningResponse:
        """Generate optimal booking date recommendations"""
        
        # Create a range of booking dates
        today = datetime.now()
        end_date = min(departure_date - timedelta(days=1), today + timedelta(days=days_range))
        
        booking_dates = []
        current_date = today
        
        while current_date <= end_date:
            booking_dates.append(current_date)
            current_date += timedelta(days=1)
        
        # Create scenario planning request
        request = ScenarioPlanningRequest(
            airline=airline,
            origin=origin,
            destination=destination,
            departure_date=departure_date,
            fare_class=fare_class,
            booking_dates=booking_dates
        )
        
        return self.scenario_planning(request)
    
    def get_price_trend(
        self,
        airline: str,
        origin: str,
        destination: str,
        departure_date: datetime,
        fare_class: str = "Economy",
        days_back: int = 30
    ) -> Dict[str, Any]:
        """Get price trend for the past N days"""
        
        if not self.model_loaded:
            raise ValueError("Model not loaded.")
        
        try:
            # Generate booking dates for the past N days
            today = datetime.now()
            booking_dates = [today - timedelta(days=i) for i in range(days_back, 0, -1)]
            
            prices = []
            dates = []
            
            for booking_date in booking_dates:
                features_df = create_features_for_prediction(
                    airline=airline,
                    origin=origin,
                    destination=destination,
                    booking_date=booking_date,
                    departure_date=departure_date,
                    fare_class=fare_class
                )
                
                predicted_price = self.model_trainer.predict(features_df)[0]
                prices.append(float(predicted_price))
                dates.append(booking_date.isoformat())
            
            # Calculate trend statistics
            price_change = prices[-1] - prices[0] if len(prices) > 1 else 0
            avg_price = np.mean(prices)
            min_price = min(prices)
            max_price = max(prices)
            
            return {
                'dates': dates,
                'prices': prices,
                'price_change': price_change,
                'average_price': avg_price,
                'min_price': min_price,
                'max_price': max_price,
                'volatility': np.std(prices),
                'trend': 'increasing' if price_change > 0 else 'decreasing' if price_change < 0 else 'stable'
            }
            
        except Exception as e:
            self.logger.error(f"Price trend analysis failed: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'model_loaded': self.model_loaded,
            'current_version': self.current_version,
            'model_metadata': self.model_trainer.model_metadata if self.model_loaded else None
        }


# Global predictor instance (singleton pattern for FastAPI)
_predictor_instance = None

def get_predictor() -> FlightPricePredictor:
    """Get the global predictor instance"""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = FlightPricePredictor()
        try:
            _predictor_instance.load_latest_model()
        except Exception as e:
            logging.warning(f"Could not load model on startup: {e}")
    
    return _predictor_instance
