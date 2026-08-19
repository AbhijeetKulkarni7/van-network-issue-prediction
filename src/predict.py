# ============================================================
# prediction.py
# VAN Network Issue Prediction
# ============================================================

import os
import joblib
import pandas as pd


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = (
    "model/van_network_issue_model.joblib"
)

METADATA_PATH = (
    "model/model_metadata.joblib"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(

            "Trained model not found. "
            "Run train_model.py first."

        )

    model = joblib.load(
        MODEL_PATH
    )

    return model


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata():

    if not os.path.exists(METADATA_PATH):

        raise FileNotFoundError(

            "Model metadata not found. "
            "Run train_model.py first."

        )

    metadata = joblib.load(
        METADATA_PATH
    )

    return metadata


# ============================================================
# PREDICT NETWORK ISSUE
# ============================================================

def predict_network_issue(input_data):

    """
    Predict whether a network issue is likely.

    Parameters
    ----------
    input_data : dict
        Network parameters.

    Returns
    -------
    dict
        Prediction result.
    """

    # Load trained model
    model = load_model()

    # Load metadata
    metadata = load_metadata()

    # Convert dictionary into DataFrame

    input_df = pd.DataFrame(
        [input_data]
    )

    # Get probability
    probability = model.predict_proba(
        input_df
    )[0][1]

    # Get recommended threshold
    threshold = metadata.get(
        "recommended_threshold",
        0.45
    )

    # Classification
    if probability >= threshold:

        prediction = 1

        status = "NETWORK ISSUE"

    else:

        prediction = 0

        status = "NORMAL"


    # Risk level

    if probability < 0.40:

        risk_level = "LOW"

    elif probability < 0.70:

        risk_level = "MEDIUM"

    else:

        risk_level = "HIGH"


    # Return result

    result = {

        "prediction":
            prediction,

        "status":
            status,

        "probability":
            round(
                float(probability),
                4
            ),

        "risk_percentage":
            round(
                float(probability * 100),
                2
            ),

        "risk_level":
            risk_level,

        "threshold":
            threshold

    }


    return result


# ============================================================
# TEST PREDICTION
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)

    print(
        "VAN NETWORK ISSUE PREDICTION"
    )

    print("=" * 70)


    # Example 1:
    # Relatively healthy network

    normal_network = {

        "Network_Type":
            "5G",

        "Topology":
            "Highway",

        "Speed_kmh":
            60,

        "Vehicle_Density_vehicles_km2":
            40,

        "Signal_Strength_dBm":
            -55,

        "Distance_to_Receiver_m":
            80,

        "CPU_Utilization_pct":
            40,

        "Memory_Utilization_pct":
            45,

        "Bandwidth_Utilization_pct":
            35,

        "Network_Traffic_Mbps":
            35,

        "Channel_Interference_pct":
            10,

        "Interface_Errors":
            1,

        "Active_Connections":
            40,

        "Latency_ms":
            25,

        "Packet_Loss_pct":
            0.5,

        "Throughput_Mbps":
            75,

        "Connection_Drops":
            0

    }


    print(
        "\nTesting normal network..."
    )


    result = predict_network_issue(
        normal_network
    )


    print(
        "\nPrediction Result:"
    )

    print(result)


    # ========================================================
    # Example 2:
    # Poor network
    # ========================================================

    problematic_network = {

        "Network_Type":
            "5G",

        "Topology":
            "Urban",

        "Speed_kmh":
            85,

        "Vehicle_Density_vehicles_km2":
            180,

        "Signal_Strength_dBm":
            -95,

        "Distance_to_Receiver_m":
            400,

        "CPU_Utilization_pct":
            90,

        "Memory_Utilization_pct":
            88,

        "Bandwidth_Utilization_pct":
            95,

        "Network_Traffic_Mbps":
            110,

        "Channel_Interference_pct":
            75,

        "Interface_Errors":
            35,

        "Active_Connections":
            220,

        "Latency_ms":
            220,

        "Packet_Loss_pct":
            12,

        "Throughput_Mbps":
            10,

        "Connection_Drops":
            8

    }


    print(
        "\nTesting problematic network..."
    )


    result = predict_network_issue(
        problematic_network
    )


    print(
        "\nPrediction Result:"
    )

    print(result)


    print(
        "\nPrediction module working successfully! 🚀"
    )