from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import logging
import psutil
from typing import Dict
import time

from app.models import (
    FlightPredictionRequest,
    FlightPredictionResponse,
    ScenarioPlanningRequest,
    ScenarioPlanningResponse,
    HealthResponse,
    ErrorResponse
)
from pipelines.inference import get_predictor


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Google Flights Airfare Forecast API",
    description="Real-time ML system for predicting airline prices and scenario planning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store startup time for uptime calculation
startup_time = time.time()

# Global predictor instance
predictor = None


@app.on_event("startup")
async def startup_event():
    """Initialize the ML model on startup"""
    global predictor
    try:
        logger.info("Starting up Flight Price Prediction API...")
        predictor = get_predictor()
        logger.info("API startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        predictor = None


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            error_code="INTERNAL_ERROR",
            details={"message": str(exc)}
        ).dict()
    )


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Google Flights Airfare Forecast API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global predictor
    
    current_time = time.time()
    uptime_seconds = current_time - startup_time
    
    model_loaded = False
    model_version = "unknown"
    
    if predictor is not None:
        try:
            model_info = predictor.get_model_info()
            model_loaded = model_info.get('model_loaded', False)
            model_version = model_info.get('current_version', 'unknown')
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        model_version=model_version,
        uptime_seconds=uptime_seconds
    )


@app.post("/predict", response_model=FlightPredictionResponse)
async def predict_flight_price(request: FlightPredictionRequest):
    """Predict flight price for a single request"""
    global predictor
    
    if predictor is None or not predictor.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please check back later."
        )
    
    try:
        # Validate input dates
        if request.departure_date <= request.booking_date:
            raise HTTPException(
                status_code=400,
                detail="Departure date must be after booking date"
            )
        
        # Make prediction
        prediction_response = predictor.predict_price(request)
        
        logger.info(f"Prediction made for {request.origin}-{request.destination} on {request.airline}")
        return prediction_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/scenario-planning", response_model=ScenarioPlanningResponse)
async def scenario_planning(request: ScenarioPlanningRequest):
    """Perform scenario planning for multiple booking dates"""
    global predictor
    
    if predictor is None or not predictor.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please check back later."
        )
    
    try:
        # Validate input dates
        for booking_date in request.booking_dates:
            if request.departure_date <= booking_date:
                raise HTTPException(
                    status_code=400,
                    detail=f"Departure date must be after all booking dates. Invalid: {booking_date}"
                )
        
        if len(request.booking_dates) > 365:
            raise HTTPException(
                status_code=400,
                detail="Maximum 365 booking dates allowed"
            )
        
        # Perform scenario planning
        scenario_response = predictor.scenario_planning(request)
        
        logger.info(f"Scenario planning completed for {request.origin}-{request.destination} with {len(request.booking_dates)} dates")
        return scenario_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Scenario planning error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Scenario planning failed: {str(e)}"
        )


@app.get("/recommendations/{origin}/{destination}")
async def get_booking_recommendations(
    origin: str,
    destination: str,
    departure_date: str,
    airline: str = "American Airlines",
    fare_class: str = "Economy",
    days_range: int = 60
):
    """Get optimal booking date recommendations"""
    global predictor
    
    if predictor is None or not predictor.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please check back later."
        )
    
    try:
        # Parse departure date
        departure_dt = datetime.fromisoformat(departure_date.replace('Z', '+00:00'))
        
        # Validate dates
        if departure_dt <= datetime.now():
            raise HTTPException(
                status_code=400,
                detail="Departure date must be in the future"
            )
        
        # Get recommendations
        recommendations = predictor.generate_booking_date_recommendations(
            airline=airline,
            origin=origin.upper(),
            destination=destination.upper(),
            departure_date=departure_dt,
            fare_class=fare_class,
            days_range=min(days_range, 365)  # Limit to 1 year
        )
        
        logger.info(f"Recommendations generated for {origin}-{destination}")
        return recommendations
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recommendations error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate recommendations: {str(e)}"
        )


@app.get("/price-trend/{origin}/{destination}")
async def get_price_trend(
    origin: str,
    destination: str,
    departure_date: str,
    airline: str = "American Airlines",
    fare_class: str = "Economy",
    days_back: int = 30
):
    """Get price trend for the past N days"""
    global predictor
    
    if predictor is None or not predictor.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please check back later."
        )
    
    try:
        # Parse departure date
        departure_dt = datetime.fromisoformat(departure_date.replace('Z', '+00:00'))
        
        # Get price trend
        trend_data = predictor.get_price_trend(
            airline=airline,
            origin=origin.upper(),
            destination=destination.upper(),
            departure_date=departure_dt,
            fare_class=fare_class,
            days_back=min(days_back, 90)  # Limit to 90 days
        )
        
        logger.info(f"Price trend retrieved for {origin}-{destination}")
        return trend_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Price trend error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get price trend: {str(e)}"
        )


@app.get("/model/info")
async def get_model_info():
    """Get information about the current model"""
    global predictor
    
    if predictor is None:
        return {"model_loaded": False, "message": "Predictor not initialized"}
    
    try:
        model_info = predictor.get_model_info()
        return model_info
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )


@app.post("/model/reload")
async def reload_model(background_tasks: BackgroundTasks):
    """Reload the latest model (background task)"""
    global predictor
    
    def reload_task():
        global predictor
        try:
            logger.info("Reloading model...")
            if predictor is None:
                predictor = get_predictor()
            else:
                predictor.load_latest_model()
            logger.info("Model reloaded successfully")
        except Exception as e:
            logger.error(f"Failed to reload model: {e}")
    
    background_tasks.add_task(reload_task)
    return {"message": "Model reload initiated"}


@app.get("/system/stats")
async def get_system_stats():
    """Get system performance statistics"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "disk_percent": disk.percent,
            "disk_free_gb": round(disk.free / (1024**3), 2),
            "uptime_seconds": time.time() - startup_time
        }
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system stats: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
