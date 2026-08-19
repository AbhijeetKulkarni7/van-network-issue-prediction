# ============================================================
# app.py
# VAN Network Issue Prediction - Streamlit Dashboard
# ============================================================

import requests
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000/predict"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_PATH = (
    PROJECT_ROOT
    / "model"
    / "model_metadata.joblib"
)

FEATURE_IMPORTANCE_PATH = (
    PROJECT_ROOT
    / "model"
    / "feature_importance.csv"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="VAN Network Monitoring",
    page_icon="🚗",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 36px;
        font-weight: bold;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL METADATA
# ============================================================

try:

    metadata = joblib.load(
        METADATA_PATH
    )

except Exception:

    metadata = {}


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🚗 VAN Network Issue Prediction'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Network Health Monitoring'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "⚙️ Network Configuration"
)

st.sidebar.info(
    "Enter the current VAN/network "
    "parameters and click Predict."
)


# ============================================================
# NETWORK INFORMATION
# ============================================================

st.subheader(
    "🌐 Network Information"
)

col1, col2, col3 = st.columns(3)


with col1:

    network_type = st.selectbox(
        "Network Type",
        [
            "5G",
            "4G",
            "WiFi"
        ]
    )


with col2:

    topology = st.selectbox(
        "Topology",
        [
            "Urban",
            "Highway",
            "Grid"
        ]
    )


with col3:

    speed = st.number_input(
        "Vehicle Speed (km/h)",
        min_value=0.0,
        max_value=200.0,
        value=60.0
    )


# ============================================================
# VEHICLE / MOBILITY
# ============================================================

st.subheader(
    "🚗 Vehicle & Mobility Metrics"
)

col1, col2, col3 = st.columns(3)


with col1:

    vehicle_density = st.number_input(
        "Vehicle Density (vehicles/km²)",
        min_value=0.0,
        max_value=300.0,
        value=60.0
    )


with col2:

    distance = st.number_input(
        "Distance to Receiver (m)",
        min_value=0.0,
        max_value=1000.0,
        value=100.0
    )


with col3:

    signal_strength = st.number_input(
        "Signal Strength (dBm)",
        min_value=-120.0,
        max_value=-20.0,
        value=-60.0
    )


# ============================================================
# SYSTEM RESOURCES
# ============================================================

st.subheader(
    "💻 System Resource Utilization"
)

col1, col2, col3 = st.columns(3)


with col1:

    cpu = st.slider(
        "CPU Utilization (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )


with col2:

    memory = st.slider(
        "Memory Utilization (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )


with col3:

    active_connections = st.number_input(
        "Active Connections",
        min_value=0,
        max_value=500,
        value=50
    )


# ============================================================
# NETWORK UTILIZATION
# ============================================================

st.subheader(
    "📡 Network Performance"
)

col1, col2, col3 = st.columns(3)


with col1:

    bandwidth = st.slider(
        "Bandwidth Utilization (%)",
        min_value=0.0,
        max_value=100.0,
        value=40.0
    )


with col2:

    traffic = st.number_input(
        "Network Traffic (Mbps)",
        min_value=0.0,
        max_value=200.0,
        value=40.0
    )


with col3:

    interference = st.slider(
        "Channel Interference (%)",
        min_value=0.0,
        max_value=100.0,
        value=10.0
    )


# ============================================================
# NETWORK HEALTH
# ============================================================

st.subheader(
    "📊 Network Health Metrics"
)

col1, col2, col3, col4 = st.columns(4)


with col1:

    latency = st.number_input(
        "Latency (ms)",
        min_value=0.0,
        max_value=500.0,
        value=30.0
    )


with col2:

    packet_loss = st.number_input(
        "Packet Loss (%)",
        min_value=0.0,
        max_value=100.0,
        value=1.0
    )


with col3:

    throughput = st.number_input(
        "Throughput (Mbps)",
        min_value=0.0,
        max_value=150.0,
        value=70.0
    )


with col4:

    interface_errors = st.number_input(
        "Interface Errors",
        min_value=0,
        max_value=100,
        value=1
    )


# ============================================================
# CONNECTION DROPS
# ============================================================

connection_drops = st.number_input(
    "Connection Drops",
    min_value=0,
    max_value=50,
    value=0
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔍 Predict Network Issue",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # Prepare input data

    input_data = {

        "Network_Type":
            network_type,

        "Topology":
            topology,

        "Speed_kmh":
            speed,

        "Vehicle_Density_vehicles_km2":
            vehicle_density,

        "Signal_Strength_dBm":
            signal_strength,

        "Distance_to_Receiver_m":
            distance,

        "CPU_Utilization_pct":
            cpu,

        "Memory_Utilization_pct":
            memory,

        "Bandwidth_Utilization_pct":
            bandwidth,

        "Network_Traffic_Mbps":
            traffic,

        "Channel_Interference_pct":
            interference,

        "Interface_Errors":
            interface_errors,

        "Active_Connections":
            active_connections,

        "Latency_ms":
            latency,

        "Packet_Loss_pct":
            packet_loss,

        "Throughput_Mbps":
            throughput,

        "Connection_Drops":
            connection_drops

    }


    # Show progress

    with st.spinner(
        "Analyzing network conditions..."
    ):

        try:

            response = requests.post(
                API_URL,
                json=input_data,
                timeout=10
            )


            # Check API response

            if response.status_code == 200:

                result = response.json()


                st.success(
                    "Prediction completed successfully!"
                )


                st.divider()


                # ==================================================
                # RESULT
                # ==================================================

                st.subheader(
                    "🎯 Prediction Result"
                )


                col1, col2, col3 = st.columns(3)


                with col1:

                    st.metric(
                        "Prediction",
                        result["status"]
                    )


                with col2:

                    st.metric(
                        "Risk Probability",
                        f"{result['risk_percentage']}%"
                    )


                with col3:

                    st.metric(
                        "Risk Level",
                        result["risk_level"]
                    )


                # ==================================================
                # RISK MESSAGE
                # ==================================================

                if result["risk_level"] == "HIGH":

                    st.error(
                        "🚨 HIGH RISK: "
                        "Potential network issue detected."
                    )

                elif result["risk_level"] == "MEDIUM":

                    st.warning(
                        "⚠️ MEDIUM RISK: "
                        "Network conditions require monitoring."
                    )

                else:

                    st.success(
                        "✅ LOW RISK: "
                        "Network conditions appear healthy."
                    )


                # ==================================================
                # PROBABILITY BAR
                # ==================================================

                st.subheader(
                    "Network Issue Probability"
                )

                st.progress(
                    min(
                        result["risk_percentage"] / 100,
                        1.0
                    )
                )


                # ==================================================
                # INPUT SUMMARY
                # ==================================================

                st.subheader(
                    "📋 Input Metrics"
                )

                display_data = {

                    "Metric": [

                        "CPU Utilization",

                        "Memory Utilization",

                        "Bandwidth Utilization",

                        "Latency",

                        "Packet Loss",

                        "Throughput",

                        "Interface Errors",

                        "Signal Strength",

                        "Vehicle Density",

                        "Channel Interference"

                    ],

                    "Value": [

                        f"{cpu}%",

                        f"{memory}%",

                        f"{bandwidth}%",

                        f"{latency} ms",

                        f"{packet_loss}%",

                        f"{throughput} Mbps",

                        interface_errors,

                        f"{signal_strength} dBm",

                        f"{vehicle_density} vehicles/km²",

                        f"{interference}%"

                    ]

                }


                display_df = pd.DataFrame(
                    display_data
                )


                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )


            else:

                st.error(
                    "API returned an error."
                )

                st.code(
                    response.text
                )


        except requests.exceptions.ConnectionError:

            st.error(
                "❌ Could not connect to FastAPI."
            )

            st.info(
                "Make sure FastAPI is running with:\n\n"
                "uvicorn api.main:app --reload"
            )


        except Exception as e:

            st.error(
                f"Unexpected error: {str(e)}"
            )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.divider()

st.subheader(
    "📈 ML Model Feature Importance"
)


try:

    feature_df = pd.read_csv(
        FEATURE_IMPORTANCE_PATH
    )

    feature_df = (
        feature_df
        .head(10)
        .sort_values(
            "Importance"
        )
    )


    st.bar_chart(
        feature_df.set_index(
            "Feature"
        )["Importance"]
    )


except Exception:

    st.info(
        "Feature importance data is not available."
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.subheader(
    "🤖 Model Information"
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Model",
        metadata.get(
            "model_name",
            "Random Forest"
        )
    )


with col2:

    threshold = metadata.get(
        "recommended_threshold",
        0.45
    )

    st.metric(
        "Decision Threshold",
        threshold
    )


with col3:

    st.metric(
        "Target",
        "Network Issue"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "VAN Network Issue Prediction | "
    "Machine Learning Prototype"
)