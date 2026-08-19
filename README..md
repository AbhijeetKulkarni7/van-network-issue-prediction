# 🚗 VAN Network Issue Prediction using Machine Learning

A machine learning-based prototype for predicting potential network issues in a Vehicular Ad Hoc Network (VANET) using network, vehicle, communication, and system-level parameters.

The project provides an end-to-end solution covering data preprocessing, machine learning model training, prediction, REST API integration, and an interactive monitoring dashboard.

---

## 📌 Project Overview

Vehicular networks operate in highly dynamic environments where network conditions can change rapidly due to vehicle mobility, congestion, interference, bandwidth limitations, packet loss, and system resource utilization.

This project aims to predict whether the current network condition is:

- 🟢 **Normal**
- 🔴 **Network Issue**

The prediction is generated using a trained **Random Forest Classification model**.

The trained model is exposed through a **FastAPI REST API**, while **Streamlit** provides an interactive dashboard for end users.

---

## 🎯 Objectives

- Predict potential VANET network issues using Machine Learning.
- Analyze important vehicle and network parameters.
- Build a reusable ML preprocessing and prediction pipeline.
- Compare different classification algorithms.
- Expose the ML model through a REST API.
- Provide an interactive monitoring dashboard.
- Generate prediction probability and risk level.
- Design a scalable AWS deployment architecture.

---

## 🏗️ System Architecture

```text
                    VANET / Network Data
                             │
                             ▼
                    ┌─────────────────┐
                    │ Data Processing │
                    │ & Preprocessing  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Random Forest   │
                    │ Classification  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Prediction      │
                    │ Probability     │
                    └────────┬────────┘
                             │
                       Threshold 0.45
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
                 NORMAL          NETWORK ISSUE
                    │                 │
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │    FastAPI      │
                    │    REST API     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    Streamlit    │
                    │   Dashboard     │
                    └─────────────────┘






🧠 Machine Learning Approach

This project is formulated as a Binary Classification problem.

Target Variable
Value	Meaning
0	Normal Network
1	Network Issue
Models Evaluated
Logistic Regression
Random Forest Classifier
Selected Model

Random Forest Classifier

Random Forest was selected because it provided better performance in our evaluation and can capture nonlinear relationships and interactions between network parameters.

📊 Dataset

The current prototype uses a synthetic dataset containing 10,000 records.

The dataset was generated using realistic ranges for vehicle, communication, network, and system parameters to simulate different network conditions.

Key Features
🚗 Vehicle & Mobility Features
Speed
Vehicle Density
Distance to Receiver
📡 Network & Communication Features
Latency
Packet Loss
Bandwidth Utilization
Signal Strength
Channel Interference
Throughput
Network Traffic
💻 System & Infrastructure Features
CPU Utilization
Memory Utilization
Interface Errors
Active Connections
Connection Drops
🌐 Categorical Features
Network Type
Network Topology
🔄 Machine Learning Pipeline
Raw Dataset
     │
     ▼
Data Validation
     │
     ▼
Missing Value Handling
     │
     ▼
Categorical Encoding
     │
     ▼
Feature Scaling
     │
     ▼
Train / Test Split
     │
     ▼
Model Training
     │
     ├─────────────────┐
     ▼                 ▼
Logistic           Random
Regression         Forest
     │                 │
     └────────┬────────┘
              ▼
       Model Evaluation
              │
              ▼
      Selected Model
      Random Forest
              │
              ▼
        Probability
              │
              ▼
       Threshold = 0.45
              │
              ▼
       Final Prediction
📈 Model Performance

The models were evaluated using classification metrics.

Random Forest Results
Metric	Result
Accuracy	82.6%
Precision	69.7%
Recall	66.6%
F1 Score	68.1%
ROC-AUC	88.6%

Random Forest was selected as the final model based on its overall performance in the prototype evaluation.

Note: The current dataset is synthetic. These results represent prototype evaluation results and should not be considered real-world production performance. Real-world deployment would require validation using independently labelled network telemetry.

🎚️ Prediction Threshold

The default classification threshold is commonly 0.50.

For this prototype, a threshold of 0.45 was selected to make the model more sensitive to potential network issues.

Probability < 0.45
        │
        ▼
     NORMAL




Probability >= 0.45
        │
        ▼
 NETWORK ISSUE

Lowering the threshold can help reduce False Negatives and improve sensitivity, but it may also increase False Positives.

🚦 Risk Classification

The prediction module converts the model probability into a user-friendly risk level.

The API provides:

Prediction
Status
Probability
Risk Percentage
Risk Level
Classification Threshold
Example Response
{
  "prediction": 1,
  "status": "NETWORK ISSUE",
  "probability": 0.9778,
  "risk_percentage": 97.78,
  "risk_level": "HIGH",
  "threshold": 0.45
}


⚡ FastAPI REST API

FastAPI is used as the backend service for serving ML predictions.

Main Endpoint
POST /predict
Request Flow
Client
  │
  ▼
POST /predict
  │
  ▼
Pydantic Validation
  │
  ▼
Prediction Module
  │
  ▼
Random Forest Model
  │
  ▼
Prediction Response

FastAPI provides automatic OpenAPI/Swagger documentation.

The API documentation is available at:

http://127.0.0.1:8000/docs

📊 Streamlit Dashboard

The project includes an interactive Streamlit dashboard for monitoring and prediction.

The user can enter network parameters and receive a real-time prediction.

Dashboard provides
Network issue prediction
Prediction probability
Risk percentage
Risk level
Network condition
Prediction Flow
User Input
    │
    ▼
Streamlit Dashboard
    │
    │ HTTP POST
    ▼
FastAPI /predict
    │
    ▼
ML Model
    │
    ▼
Prediction
    │
    ▼
JSON Response
    │
    ▼
Streamlit Dashboard


📁 Project Structure
van-network-issue-prediction/
│
├── api/
│   └── main.py
│
├── dashboard/
│   └── app.py
│
├── data/
│   └── van_network_issue_dataset.csv
│
├── model/
│   ├── trained_model.joblib
│   └── ...
│
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   └── prediction.py
│
├── .gitignore
├── requirements.txt
└── README.md



🛠️ Technology Stack

Programming Language
Python
Machine Learning
Pandas
NumPy
Scikit-learn
Logistic Regression
Random Forest
Backend
FastAPI
Pydantic
Uvicorn
Dashboard
Streamlit
Development Tools
VS Code
Git
GitHub
Cloud Architecture
AWS ECS
AWS Fargate
Application Load Balancer
Amazon S3
Amazon CloudWatch


▶️ Run the Project Locally
1. Clone the Repository
git clone git@github.com:AbhijeetKulkarni7/van-network-issue-prediction.git
cd van-network-issue-prediction
2. Create Virtual Environment
python -m venv venv
3. Activate Virtual Environment
Windows CMD / PowerShell
venv\Scripts\activate
Git Bash
source venv/Scripts/activate
4. Install Dependencies
pip install -r requirements.txt
5. Run Preprocessing
python src/preprocessing.py
6. Train the Model
python src/train_model.py
7. Start FastAPI
uvicorn api.main:app --reload

FastAPI documentation:

http://127.0.0.1:8000/docs
8. Start Streamlit

Open another terminal and run:

streamlit run dashboard/app.py

The Streamlit dashboard will open in the browser.

☁️ AWS Deployment Architecture

For production deployment, the FastAPI application can be containerized using Docker and deployed using Amazon ECS with Fargate.

                         AWS Cloud
                             │
                             ▼
                  ┌────────────────────┐
                  │ Application Load   │
                  │     Balancer       │
                  └─────────┬──────────┘
                            │
               ┌────────────┼────────────┐
               ▼            ▼            ▼
          ┌─────────┐  ┌─────────┐  ┌─────────┐
          │ Fargate │  │ Fargate │  │ Fargate │
          │ Task 1  │  │ Task 2  │  │ Task 3  │
          │ FastAPI │  │ FastAPI │  │ FastAPI │
          │ + Model │  │ + Model │  │ + Model │
          └────┬────┘  └────┬────┘  └────┬────┘
               │            │            │
               └────────────┼────────────┘
                            │
                   ┌────────┴────────┐
                   ▼                 ▼
              Amazon S3        CloudWatch
             Model / Data       Monitoring




The Application Load Balancer distributes incoming requests across healthy FastAPI tasks.

🔐 Production Considerations

For a production implementation, the following improvements can be added:

HTTPS/TLS
API authentication and authorization
AWS IAM roles
AWS Secrets Manager
Private S3 buckets
CloudWatch monitoring
Centralized logging
Model versioning
Database integration
Data drift monitoring
Automated model retraining

⚠️ Current Limitations
The current dataset is synthetic.
Real-world VANET/network telemetry validation is required.
The current model is a prototype.
Production deployment on AWS has been designed but not implemented in the current prototype.
Real-time streaming data integration is not currently implemented.
Further model tuning and validation are required for production use.

🔮 Future Enhancements
Integrate real-world VANET/network telemetry.
Add real-time data streaming.
Evaluate Gradient Boosting, XGBoost, LightGBM or other advanced models.
Implement automated model retraining.
Add ML model versioning.
Implement real-time network alerts.
Integrate PostgreSQL for telemetry and prediction history.
Deploy the complete application on AWS.
Add data drift and model performance monitoring.
Implement CI/CD for automated deployment.

👨‍💻 Author
Abhijeet Kulkarni
Assistant System Engineer | TCS

GitHub:
https://github.com/AbhijeetKulkarni7

⭐ Project

If you find this project useful, consider giving the repository a ⭐ on GitHub.
