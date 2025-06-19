import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import logging


class FeatureEngineer:
    """Feature engineering pipeline for flight price prediction"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.categorical_encoders = {}
        self.feature_scalers = {}
        self.is_fitted = False
    
    def engineer_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from booking and departure dates"""
        df = df.copy()
        
        # Convert datetime columns
        df['booking_dt'] = pd.to_datetime(df['booking_dt'])
        df['departure_dt'] = pd.to_datetime(df['departure_dt'])
        
        # Days between booking and departure
        df['days_before_departure'] = (df['departure_dt'] - df['booking_dt']).dt.days
        
        # Booking date features
        df['booking_month'] = df['booking_dt'].dt.month
        df['booking_day_of_week'] = df['booking_dt'].dt.dayofweek
        df['booking_day_of_year'] = df['booking_dt'].dt.dayofyear
        df['booking_hour'] = df['booking_dt'].dt.hour
        df['booking_quarter'] = df['booking_dt'].dt.quarter
        
        # Departure date features
        df['departure_month'] = df['departure_dt'].dt.month
        df['departure_day_of_week'] = df['departure_dt'].dt.dayofweek
        df['departure_day_of_year'] = df['departure_dt'].dt.dayofyear
        df['departure_hour'] = df['departure_dt'].dt.hour
        df['departure_quarter'] = df['departure_dt'].dt.quarter
        
        # Seasonal features
        df['is_weekend_departure'] = df['departure_day_of_week'].isin([5, 6]).astype(int)
        df['is_weekend_booking'] = df['booking_day_of_week'].isin([5, 6]).astype(int)
        
        # Holiday/peak season indicators
        df['is_summer'] = df['departure_month'].isin([6, 7, 8]).astype(int)
        df['is_winter_holidays'] = df['departure_month'].isin([12, 1]).astype(int)
        df['is_spring_break'] = df['departure_month'].isin([3, 4]).astype(int)
        
        # Advance booking categories
        df['booking_category'] = pd.cut(
            df['days_before_departure'], 
            bins=[-1, 0, 7, 14, 30, 60, 90, float('inf')],
            labels=['same_day', 'week', '2weeks', 'month', '2months', '3months', 'advance']
        )
        
        return df
    
    def engineer_route_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create route-based features"""
        df = df.copy()
        
        # Route identifier
        df['route'] = df['origin'] + '_' + df['destination']
        
        # Major hub indicators (you can expand this list)
        major_hubs = ['ORD', 'ATL', 'LAX', 'DFW', 'DEN', 'JFK', 'SFO', 'LAS', 'PHX', 'CLT']
        df['origin_is_hub'] = df['origin'].isin(major_hubs).astype(int)
        df['destination_is_hub'] = df['destination'].isin(major_hubs).astype(int)
          # Route popularity (will be calculated during fit)
        if self.is_fitted and hasattr(self, 'route_popularity'):
            df['route_popularity'] = df['route'].map(self.route_popularity).fillna(0)
        else:
            # Placeholder during initial fitting
            df['route_popularity'] = 0.0
        
        return df
    
    def engineer_airline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create airline-based features"""
        df = df.copy()
        
        # Legacy carriers vs Low-cost carriers (simplified categorization)
        legacy_carriers = ['American Airlines', 'Delta Air Lines', 'United Airlines']
        df['is_legacy_carrier'] = df['airline'].isin(legacy_carriers).astype(int)
          # Airline market share (will be calculated during fit)
        if self.is_fitted and hasattr(self, 'airline_market_share'):
            df['airline_market_share'] = df['airline'].map(self.airline_market_share).fillna(0)
        else:
            # Placeholder during initial fitting
            df['airline_market_share'] = 0.0
        
        return df
    
    def engineer_price_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Create price-based features (only for training data)"""
        if not is_training:
            return df
            
        df = df.copy()
        
        # Price buckets
        df['price_bucket'] = pd.qcut(df['price'], q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        
        # Log price for better distribution
        df['log_price'] = np.log1p(df['price'])
        
        return df
    
    def calculate_route_popularity(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate route popularity based on booking frequency"""
        route_counts = df['route'].value_counts()
        total_bookings = len(df)
        route_popularity = (route_counts / total_bookings).to_dict()
        return route_popularity
    
    def calculate_airline_market_share(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate airline market share"""
        airline_counts = df['airline'].value_counts()
        total_bookings = len(df)
        market_share = (airline_counts / total_bookings).to_dict()
        return market_share
    
    def encode_categorical_features(self, df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        """Encode categorical features using target encoding or one-hot encoding"""
        df = df.copy()
        
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.categorical_encoders:
                    # For high-cardinality features, use target encoding
                    if df[col].nunique() > 10:
                        if 'price' in df.columns:  # Training mode
                            encoding = df.groupby(col)['price'].mean().to_dict()
                        else:  # Inference mode - use stored encoding
                            encoding = {}
                    else:
                        # For low-cardinality features, use one-hot encoding
                        encoding = pd.get_dummies(df[col], prefix=col)
                        if isinstance(encoding, pd.DataFrame):
                            df = pd.concat([df, encoding], axis=1)
                            df.drop(columns=[col], inplace=True)
                            continue
                    
                    self.categorical_encoders[col] = encoding
                
                # Apply encoding
                if col in self.categorical_encoders:
                    encoding = self.categorical_encoders[col]
                    if isinstance(encoding, dict):
                        df[f'{col}_encoded'] = df[col].map(encoding).fillna(df[col].map(encoding).mean())                
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the feature engineering pipeline and transform the data"""
        self.logger.info("Starting feature engineering pipeline...")
        
        # Apply all feature engineering steps
        df = self.engineer_temporal_features(df)
        df = self.engineer_route_features(df)
        df = self.engineer_airline_features(df)
        df = self.engineer_price_features(df, is_training=True)
          # Calculate route popularity and airline market share after route column is created
        self.route_popularity = self.calculate_route_popularity(df)
        self.airline_market_share = self.calculate_airline_market_share(df)
        
        # Update route popularity and airline market share columns with calculated values
        df['route_popularity'] = df['route'].map(self.route_popularity).fillna(0)
        df['airline_market_share'] = df['airline'].map(self.airline_market_share).fillna(0)
        
        # Categorical encoding
        categorical_cols = ['airline', 'origin', 'destination', 'fare_class', 'route', 'booking_category']
        df = self.encode_categorical_features(df, categorical_cols)
        
        self.is_fitted = True
        self.logger.info(f"Feature engineering completed. Shape: {df.shape}")
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted pipeline"""
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        # Apply all feature engineering steps
        df = self.engineer_temporal_features(df)
        df = self.engineer_route_features(df)
        df = self.engineer_airline_features(df)
        df = self.engineer_price_features(df, is_training=False)
        
        # Categorical encoding
        categorical_cols = ['airline', 'origin', 'destination', 'fare_class', 'route', 'booking_category']
        df = self.encode_categorical_features(df, categorical_cols)
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of engineered feature names"""
        base_features = [
            'days_before_departure', 'booking_month', 'booking_day_of_week',
            'booking_day_of_year', 'booking_hour', 'booking_quarter',
            'departure_month', 'departure_day_of_week', 'departure_day_of_year',
            'departure_hour', 'departure_quarter', 'is_weekend_departure',
            'is_weekend_booking', 'is_summer', 'is_winter_holidays',
            'is_spring_break', 'origin_is_hub', 'destination_is_hub',
            'route_popularity', 'is_legacy_carrier', 'airline_market_share'
        ]
        
        # Add encoded categorical features
        for col, encoding in self.categorical_encoders.items():
            if isinstance(encoding, dict):
                base_features.append(f'{col}_encoded')
        
        return base_features


def create_features_for_prediction(
    airline: str,
    origin: str,
    destination: str,
    booking_date: datetime,
    departure_date: datetime,
    fare_class: str
) -> pd.DataFrame:
    """Create a DataFrame with features for a single prediction"""
    data = {
        'airline': [airline],
        'origin': [origin],
        'destination': [destination],
        'booking_dt': [booking_date],
        'departure_dt': [departure_date],
        'fare_class': [fare_class]
    }
    
    return pd.DataFrame(data)
