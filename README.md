# Google Flights Airfare Forecast - Real-time ML System

A comprehensive **MLOps system** for predicting airline prices with AI-powered scenario planning and booking optimization. This project implements a complete real-time machine learning pipeline with feature engineering, model training and inference.

## Overview

This is a complete **real-time ML system** for predicting airline prices and providing scenario planning for optimal booking strategies. The system includes feature pipelines, training pipelines, inference pipelines, and a FastAPI-based REST API with a stunning Streamlit UI.

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Feature        │    │  Training       │    │  Inference      │
│  Pipeline       │    │  Pipeline       │    │  Pipeline       │
│                 │    │                 │    │                 │
│ • Data Ingestion│    │ • Feature Store │    │ • Model Registry│
│ • Transformation│───▶│ • Model Training│───▶│ • Real-time API │
│ • Feature Store │    │ • Model Registry│    │ • Predictions   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Components

#### 1. Feature Pipeline (`pipelines/feature_engineering.py`)
- Ingests raw flight data in real-time
- Transforms data into ML-ready features
- Advanced feature engineering (temporal, route, airline insights)
- Stores features in a feature store

#### 2. Training Pipeline (`pipelines/training.py`)
- Ingests training data from feature store
- Multiple ML algorithms (XGBoost, Random Forest, etc.)
- Hyperparameter tuning and cross-validation
- Pushes models to model registry with versioning

#### 3. Inference Pipeline (`pipelines/inference.py`)
- Loads models from registry
- Serves real-time predictions via FastAPI
- Confidence intervals and uncertainty quantification
- Scenario planning capabilities

#### 4. API Layer (`app/main.py`)
- FastAPI-based REST API for real-time predictions
- Comprehensive endpoints for all use cases
- Interactive documentation and validation

#### 5. Web Interface (`demo.py`)
- Streamlit web app
- Interactive visualizations and charts
- User-friendly interface for all features

## Features

### 💡 Intelligent Features
- **Advance Booking Analysis**: Optimal timing for bookings
- **Seasonal Patterns**: Holiday and peak season detection
- **Route Intelligence**: Hub detection and popularity metrics
- **Airline Insights**: Legacy vs budget carrier patterns
- **Temporal Features**: Day of week, time of day effects

### 🤖 Multiple ML Models
- **XGBoost**: High performance gradient boosting
- **Random Forest**: Robust ensemble method
- **Gradient Boosting**: Advanced boosting techniques
- **Ridge Regression**: Linear baseline model

### 🎯 Business Intelligence
- **Price Prediction**: Forecast flight prices with confidence intervals
- **Scenario Planning**: "What-if" analysis to help users find optimal booking dates
- **Booking Recommendations**: AI-powered suggestions for best booking times
- **Price Trends**: Historical price movement analysis
- **Savings Analysis**: Identify cost optimization opportunities

### 🌐 Real-time API
- FastAPI-based REST API for real-time predictions
- Interactive API documentation
- Comprehensive error handling and validation
- Model health monitoring and metrics

## 📊 Dataset

The system uses historical flight booking data with the following features:
- `transaction_id`: Unique transaction identifier
- `flight_id`: Unique flight identifier
- `airline`: Airline name
- `origin`: Origin airport code
- `destination`: Destination airport code
- `booking_dt`: Booking timestamp
- `departure_dt`: Departure timestamp
- `price`: Flight price (target variable)
- `fare_class`: Economy/Business/First class

## ⚡ Quick Start

### 1. Setup Environment
```bash
# Clone or download the project
cd Google-Flights-Airfare-Forecast-MLOPS

# Install dependencies (automatically creates virtual environment)
python setup.py
```

### 2. Train Your First Model
```bash
# Train with default settings
python train_model.py

# Train with hyperparameter tuning
python train_model.py --model-name xgboost --tune-hyperparameters

# Train specific algorithm
python train_model.py --model-name random_forest
```

### 3. Start the API Server
```bash
# Start development server
python start_server.py --reload

# Start production server
python start_server.py --host 0.0.0.0 --port 8000
```

### 4. Launch the Web Interface
```bash
# Start  Streamlit web app
streamlit run demo.py
```

## 🌐 API Endpoints

### Core Prediction Endpoints

#### 🔮 Single Flight Prediction
```http
POST /predict
Content-Type: application/json

{
  "airline": "American Airlines",
  "origin": "ORD",
  "destination": "BOS",
  "booking_date": "2024-01-15T10:00:00",
  "departure_date": "2024-03-15T08:00:00",
  "fare_class": "Economy"
}
```

#### 📅 Scenario Planning
```http
POST /scenario-planning
Content-Type: application/json

{
  "airline": "American Airlines",
  "origin": "ORD",
  "destination": "BOS",
  "departure_date": "2024-03-15T08:00:00",
  "fare_class": "Economy",
  "booking_dates": [
    "2024-01-15T10:00:00",
    "2024-01-20T10:00:00",
    "2024-01-25T10:00:00"
  ]
}
```

#### 🎯 Booking Recommendations
```http
GET /recommendations/{origin}/{destination}?departure_date=2024-03-15T08:00:00&airline=American Airlines
```

#### 📈 Price Trends
```http
GET /price-trend/{origin}/{destination}?departure_date=2024-03-15T08:00:00&days_back=30
```

### System Endpoints

#### 🏥 Health Check
```http
GET /health
```

#### 🤖 Model Information
```http
GET /model/info
```

#### 📊 System Statistics
```http
GET /system/stats
```

## 📈 Performance Metrics

Based on our analysis of the flight data:

- **Mean Absolute Error**: ~$45-65 (depending on model)
- **R² Score**: 0.85-0.92
- **MAPE**: 15-25%
- **Prediction Accuracy**: 75-85% within $50 range

## 🔧 Configuration

### Environment Variables
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=info

# Model Configuration
MODEL_DIR=models
DEFAULT_MODEL=xgboost
AUTO_RELOAD=false
```

### Config File 
(`config.ini`)


## 🐳 Docker Deployment

### Build and Run
```bash
# Build image
docker build -t flight-price-api .

# Run container
docker run -p 8000:8000 -v ./data:/app/data -v ./models:/app/models flight-price-api

# Use Docker Compose
docker-compose up -d
```

### Production Deployment
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  flight-api:
    image: flight-price-api:latest
    ports:
      - "80:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - LOG_LEVEL=warning
    restart: always
```

## 📊 Business Insights

### Key Findings
1. **Optimal Booking Window**: 2-3 months in advance
2. **Day of Week**: Tuesday/Wednesday departures are cheapest
3. **Seasonal Patterns**: Summer and holidays are most expensive
4. **Advance Booking**: Same-day bookings cost 3x more on average
5. **Hub Effect**: Major hubs offer more competitive pricing

### Savings Opportunities
- **Flexible Dates**: Up to $200 savings by shifting departure by 1-2 days
- **Advance Planning**: $100-300 savings by booking 30-60 days ahead
- **Day Selection**: $50-100 savings by choosing weekday departures
- **Seasonal Timing**: $150-400 savings by avoiding peak seasons

## 🧪 Testing

### Run Tests
```bash
# Run all tests
pytest tests/ -v

# Run API tests
pytest tests/test_api.py -v

# Run with coverage
pytest tests/ --cov=app --cov=pipelines
```

## 🔄 Model Updates

### Retrain Model
```bash
# Retrain with new data
python train_model.py --data-path data/new_data.csv

# Retrain with different algorithm
python train_model.py --model-name random_forest --tune-hyperparameters
```

### Model Versioning
```bash
# Models are automatically versioned with timestamps
# Load specific version
python -c "
from pipelines.inference import FlightPricePredictor
predictor = FlightPricePredictor()
predictor.load_model_version('20241215_143022')
"
```


🎉 **Ready to predict flight prices!** 

**Quick Start Commands:**
1. `python setup.py` - Setup environment
2. `python train_model.py` - Train your first model  
3. `python start_server.py --reload` - Start the API
4. `streamlit run demo.py` - Launch the web interface

**Access Points:**
- 🌐 **Web Interface**: http://localhost:8501
- 📖 **API Docs**: http://localhost:8000/docs
- 🔍 **Health Check**: http://localhost:8000/health
