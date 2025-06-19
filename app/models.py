from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from enum import Enum


class FareClass(str, Enum):
    """Fare class enumeration"""
    ECONOMY = "Economy"
    BUSINESS = "Business"
    FIRST = "First"


class FlightPredictionRequest(BaseModel):
    """Request model for flight price prediction"""
    airline: str = Field(..., description="Airline name")
    origin: str = Field(..., description="Origin airport code (e.g., 'ORD')")
    destination: str = Field(..., description="Destination airport code (e.g., 'BOS')")
    booking_date: datetime = Field(..., description="Booking date and time")
    departure_date: datetime = Field(..., description="Departure date and time")
    fare_class: FareClass = Field(default=FareClass.ECONOMY, description="Fare class")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class FlightPredictionResponse(BaseModel):
    """Response model for flight price prediction"""
    predicted_price: float = Field(..., description="Predicted flight price")
    confidence_interval: tuple[float, float] = Field(..., description="95% confidence interval")
    booking_date: datetime = Field(..., description="Booking date used for prediction")
    departure_date: datetime = Field(..., description="Departure date")
    route: str = Field(..., description="Flight route (origin-destination)")
    airline: str = Field(..., description="Airline name")
    fare_class: str = Field(..., description="Fare class")
    model_version: str = Field(..., description="Model version used for prediction")
    prediction_timestamp: datetime = Field(default_factory=datetime.now, description="When prediction was made")


class ScenarioPlanningRequest(BaseModel):
    """Request model for scenario planning"""
    airline: str = Field(..., description="Airline name")
    origin: str = Field(..., description="Origin airport code")
    destination: str = Field(..., description="Destination airport code")
    departure_date: datetime = Field(..., description="Departure date and time")
    fare_class: FareClass = Field(default=FareClass.ECONOMY, description="Fare class")
    booking_dates: List[datetime] = Field(..., description="List of potential booking dates to analyze")


class BookingScenario(BaseModel):
    """Individual booking scenario result"""
    booking_date: datetime = Field(..., description="Booking date")
    predicted_price: float = Field(..., description="Predicted price for this booking date")
    days_before_departure: int = Field(..., description="Days between booking and departure")
    potential_savings: float = Field(..., description="Savings compared to baseline (negative means higher cost)")


class ScenarioPlanningResponse(BaseModel):
    """Response model for scenario planning"""
    scenarios: List[BookingScenario] = Field(..., description="List of booking scenarios")
    best_booking_date: datetime = Field(..., description="Optimal booking date for lowest price")
    worst_booking_date: datetime = Field(..., description="Worst booking date for highest price")
    max_savings: float = Field(..., description="Maximum potential savings")
    departure_date: datetime = Field(..., description="Departure date")
    route: str = Field(..., description="Flight route")
    airline: str = Field(..., description="Airline name")
    baseline_price: float = Field(..., description="Baseline price (current/reference booking date)")
    analysis_timestamp: datetime = Field(default_factory=datetime.now, description="When analysis was performed")


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    model_version: str = Field(..., description="Current model version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    details: Optional[dict] = Field(None, description="Additional error details")
