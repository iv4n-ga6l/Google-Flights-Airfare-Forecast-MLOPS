import pytest
import httpx
from datetime import datetime, timedelta
from app.models import FlightPredictionRequest, ScenarioPlanningRequest, FareClass


class TestFlightPriceAPI:
    """Test cases for the Flight Price Prediction API"""
    
    base_url = "http://localhost:8000"
    
    @pytest.fixture
    def client(self):
        """HTTP client fixture"""
        return httpx.Client(base_url=self.base_url)
    
    def test_health_endpoint(self, client):
        """Test the health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "uptime_seconds" in data
    
    def test_root_endpoint(self, client):
        """Test the root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_predict_endpoint(self, client):
        """Test the price prediction endpoint"""
        # Create a valid prediction request
        booking_date = datetime.now()
        departure_date = booking_date + timedelta(days=30)
        
        request_data = {
            "airline": "American Airlines",
            "origin": "ORD",
            "destination": "BOS",
            "booking_date": booking_date.isoformat(),
            "departure_date": departure_date.isoformat(),
            "fare_class": "Economy"
        }
        
        response = client.post("/predict", json=request_data)
        
        # Check if model is loaded
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "predicted_price" in data
        assert "confidence_interval" in data
        assert "model_version" in data
        assert data["predicted_price"] > 0
    
    def test_predict_invalid_dates(self, client):
        """Test prediction with invalid dates"""
        booking_date = datetime.now()
        departure_date = booking_date - timedelta(days=1)  # Invalid: departure before booking
        
        request_data = {
            "airline": "American Airlines",
            "origin": "ORD",
            "destination": "BOS",
            "booking_date": booking_date.isoformat(),
            "departure_date": departure_date.isoformat(),
            "fare_class": "Economy"
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 400
    
    def test_scenario_planning_endpoint(self, client):
        """Test the scenario planning endpoint"""
        booking_date = datetime.now()
        departure_date = booking_date + timedelta(days=30)
        
        booking_dates = [
            (booking_date + timedelta(days=i)).isoformat()
            for i in range(0, 7)  # 7 different booking dates
        ]
        
        request_data = {
            "airline": "American Airlines",
            "origin": "ORD",
            "destination": "BOS",
            "departure_date": departure_date.isoformat(),
            "fare_class": "Economy",
            "booking_dates": booking_dates
        }
        
        response = client.post("/scenario-planning", json=request_data)
        
        # Check if model is loaded
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "scenarios" in data
        assert "best_booking_date" in data
        assert "max_savings" in data
        assert len(data["scenarios"]) == len(booking_dates)
    
    def test_recommendations_endpoint(self, client):
        """Test the booking recommendations endpoint"""
        departure_date = (datetime.now() + timedelta(days=30)).isoformat()
        
        response = client.get(
            f"/recommendations/ORD/BOS?departure_date={departure_date}&airline=American Airlines"
        )
        
        # Check if model is loaded
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "scenarios" in data
        assert "best_booking_date" in data
    
    def test_price_trend_endpoint(self, client):
        """Test the price trend endpoint"""
        departure_date = (datetime.now() + timedelta(days=30)).isoformat()
        
        response = client.get(
            f"/price-trend/ORD/BOS?departure_date={departure_date}&airline=American Airlines"
        )
        
        # Check if model is loaded
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "prices" in data
        assert "dates" in data
        assert "trend" in data
    
    def test_model_info_endpoint(self, client):
        """Test the model info endpoint"""
        response = client.get("/model/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "model_loaded" in data
    
    def test_system_stats_endpoint(self, client):
        """Test the system stats endpoint"""
        response = client.get("/system/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "cpu_percent" in data
        assert "memory_percent" in data
        assert "uptime_seconds" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
