# ============================================================
# main.py
# VAN Network Issue Prediction - FastAPI
# ============================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import predict_network_issue


# ============================================================
# CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="VAN Network Issue Prediction API",
    description=(
        "REST API for predicting potential network issues "
        "using a machine learning model."
    ),
    version="1.0.0"
)


# ============================================================
# REQUEST DATA MODEL
# ============================================================

class NetworkData(BaseModel):

    Network_Type: str = Field(
        ...,
        description="Network type such as 5G, 4G or WiFi"
    )

    Topology: str = Field(
        ...,
        description="Network topology such as Urban, Highway or Grid"
    )

    Speed_kmh: float

    Vehicle_Density_vehicles_km2: float

    Signal_Strength_dBm: float

    Distance_to_Receiver_m: float

    CPU_Utilization_pct: float = Field(
        ...,
        ge=0,
        le=100
    )

    Memory_Utilization_pct: float = Field(
        ...,
        ge=0,
        le=100
    )

    Bandwidth_Utilization_pct: float = Field(
        ...,
        ge=0,
        le=100
    )

    Network_Traffic_Mbps: float

    Channel_Interference_pct: float = Field(
        ...,
        ge=0,
        le=100
    )

    Interface_Errors: int = Field(
        ...,
        ge=0
    )

    Active_Connections: int = Field(
        ...,
        ge=0
    )

    Latency_ms: float = Field(
        ...,
        ge=0
    )

    Packet_Loss_pct: float = Field(
        ...,
        ge=0,
        le=100
    )

    Throughput_Mbps: float = Field(
        ...,
        ge=0
    )

    Connection_Drops: int = Field(
        ...,
        ge=0
    )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "application": "VAN Network Issue Prediction",
        "status": "running",
        "version": "1.0.0"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": "VAN Network Prediction API"
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict_network(data: NetworkData):

    try:

        # Convert Pydantic object into dictionary

        input_data = data.model_dump()

        # Send data to our ML prediction module

        result = predict_network_issue(
            input_data
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )