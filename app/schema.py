from datetime import datetime
from pydantic import BaseModel, Field


class TripRequest(BaseModel):
    tpep_pickup_datetime: datetime = Field(..., examples=["2025-01-15T08:30:00"])
    PULocationID: int = Field(..., ge=1, le=265, examples=[161])
    DOLocationID: int = Field(..., ge=1, le=265, examples=[236])
    trip_distance: float = Field(..., gt=0, examples=[2.4])


class TripResponse(BaseModel):
    duration_minutes: float
    model_version: str