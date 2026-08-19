# ============================================================
# train_model.py
# VAN Network Issue Prediction
# ============================================================

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Import our preprocessing functions
from preprocessing import (
    prepare_data,
    get_numeric_features,
    create_preprocessor
)


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

DATA_PATH = "data/van_network_issue_dataset.csv"

MODEL_DIR = "model"

TARGET_COLUMN = "Network_Issue"

RANDOM_STATE = 42

TEST_SIZE = 0.20


# Create model directory
os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("1. LOADING DATASET")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")

print(
    "Rows:",
    df.shape[0]
)

print(
    "Columns:",
    df.shape[1]
)

print("\nFirst 5 records:")

print(
    df.head()
)


# ============================================================
# 2. DATA UNDERSTANDING
# ============================================================

print("\n" + "=" * 70)
print("2. DATA UNDERSTANDING")
print("=" * 70)

print("\nData types:")

print(
    df.dtypes
)

print("\nMissing values:")

print(
    df.isnull().sum()
)

print(
    "\nTotal missing values:",
    df.isnull().sum().sum()
)


# ============================================================
# 3. DUPLICATE CHECK
# ============================================================

print("\n" + "=" * 70)
print("3. DUPLICATE CHECK")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print(
    "Duplicate records:",
    duplicate_count
)

if duplicate_count > 0:

    df = df.drop_duplicates()

    print(
        "Duplicates removed."
    )

else:

    print(
        "No duplicates found."
    )


# ============================================================
# 4. TARGET ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("4. TARGET ANALYSIS")
print("=" * 70)

print("\nTarget:", TARGET_COLUMN)

print("\nTarget distribution:")

print(
    df[TARGET_COLUMN].value_counts()
)

print("\nTarget percentage:")

print(
    df[TARGET_COLUMN]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# 5. PREPARE X AND y
# ============================================================

print("\n" + "=" * 70)
print("5. PREPARING FEATURES AND TARGET")
print("=" * 70)

X, y = prepare_data(df)

print("\nFeatures used:")

print(
    X.columns.tolist()
)

print("\nTarget:")

print(
    "0 = Normal"
)

print(
    "1 = Network Issue"
)


# ============================================================
# 6. IDENTIFY NUMERICAL FEATURES
# ============================================================

numeric_features = get_numeric_features(X)

print("\nNumerical features:")

for feature in numeric_features:

    print(
        "-",
        feature
    )


# ============================================================
# 7. CREATE PREPROCESSOR
# ============================================================

print("\n" + "=" * 70)
print("6. CREATING PREPROCESSING PIPELINE")
print("=" * 70)

preprocessor = create_preprocessor(
    numeric_features
)

print(
    "Preprocessing pipeline created."
)


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("7. TRAIN / TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE,

    stratify=y
)


print(
    "\nTraining records:",
    len(X_train)
)

print(
    "Testing records:",
    len(X_test)
)


# ============================================================
# 9. LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("8. TRAINING LOGISTIC REGRESSION")
print("=" * 70)

logistic_model = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",

            LogisticRegression(

                max_iter=1000,

                class_weight="balanced",

                random_state=RANDOM_STATE

            )
        )

    ]
)


logistic_model.fit(

    X_train,

    y_train

)


print(
    "Logistic Regression training completed."
)


# ============================================================
# 10. RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("9. TRAINING RANDOM FOREST")
print("=" * 70)

# Create a fresh preprocessing pipeline
# because each model should have its own fitted pipeline.

rf_preprocessor = create_preprocessor(
    numeric_features
)


random_forest_model = Pipeline(

    steps=[

        (
            "preprocessor",
            rf_preprocessor
        ),

        (
            "model",

            RandomForestClassifier(

                n_estimators=250,

                max_depth=12,

                min_samples_leaf=3,

                class_weight="balanced",

                random_state=RANDOM_STATE,

                n_jobs=-1

            )
        )

    ]
)


random_forest_model.fit(

    X_train,

    y_train

)


print(
    "Random Forest training completed."
)


# ============================================================
# 11. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    model,
    X_test,
    y_test
):

    print("\n" + "=" * 70)

    print(
        f"MODEL EVALUATION: {model_name}"
    )

    print("=" * 70)


    # Predictions

    predictions = model.predict(
        X_test
    )


    # Probability of Network Issue

    probabilities = model.predict_proba(
        X_test
    )[:, 1]


    # Metrics

    accuracy = accuracy_score(
        y_test,
        predictions
    )


    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )


    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )


    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )


    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )


    print(
        "\nAccuracy :",
        f"{accuracy:.4f}"
    )

    print(
        "Precision:",
        f"{precision:.4f}"
    )

    print(
        "Recall   :",
        f"{recall:.4f}"
    )

    print(
        "F1 Score :",
        f"{f1:.4f}"
    )

    print(
        "ROC-AUC  :",
        f"{roc_auc:.4f}"
    )


    # Confusion Matrix

    cm = confusion_matrix(

        y_test,

        predictions

    )


    print(
        "\nConfusion Matrix:"
    )

    print(cm)


    # Classification Report

    print(
        "\nClassification Report:"
    )

    print(

        classification_report(

            y_test,

            predictions,

            target_names=[

                "Normal",

                "Network Issue"

            ],

            zero_division=0

        )

    )


    return {

        "model_name":
            model_name,

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "roc_auc":
            roc_auc,

        "predictions":
            predictions,

        "probabilities":
            probabilities,

        "confusion_matrix":
            cm

    }


# ============================================================
# 12. EVALUATE LOGISTIC REGRESSION
# ============================================================

logistic_results = evaluate_model(

    "Logistic Regression",

    logistic_model,

    X_test,

    y_test

)


# ============================================================
# 13. EVALUATE RANDOM FOREST
# ============================================================

random_forest_results = evaluate_model(

    "Random Forest",

    random_forest_model,

    X_test,

    y_test

)


# ============================================================
# 14. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("10. MODEL COMPARISON")
print("=" * 70)


comparison = pd.DataFrame(

    [

        {

            "Model":
                logistic_results["model_name"],

            "Accuracy":
                logistic_results["accuracy"],

            "Precision":
                logistic_results["precision"],

            "Recall":
                logistic_results["recall"],

            "F1 Score":
                logistic_results["f1"],

            "ROC-AUC":
                logistic_results["roc_auc"]

        },

        {

            "Model":
                random_forest_results["model_name"],

            "Accuracy":
                random_forest_results["accuracy"],

            "Precision":
                random_forest_results["precision"],

            "Recall":
                random_forest_results["recall"],

            "F1 Score":
                random_forest_results["f1"],

            "ROC-AUC":
                random_forest_results["roc_auc"]

        }

    ]

)


print(

    comparison
    .round(4)
    .to_string(index=False)

)


# ============================================================
# 15. SELECT FINAL MODEL
# ============================================================

if (

    random_forest_results["f1"]

    >=

    logistic_results["f1"]

):

    final_model = random_forest_model

    final_results = random_forest_results

else:

    final_model = logistic_model

    final_results = logistic_results


print("\n" + "=" * 70)

print(
    "11. FINAL MODEL SELECTION"
)

print("=" * 70)


print(

    "Selected Model:",

    final_results["model_name"]

)


# ============================================================
# 16. THRESHOLD TUNING
# ============================================================

print("\n" + "=" * 70)
print("12. THRESHOLD TUNING")
print("=" * 70)


rf_probabilities = (
    random_forest_results["probabilities"]
)


threshold_results = []


for threshold in [

    0.30,

    0.35,

    0.40,

    0.45,

    0.50,

    0.55,

    0.60,

    0.65,

    0.70

]:

    predictions = (

        rf_probabilities >= threshold

    ).astype(int)


    precision = precision_score(

        y_test,

        predictions,

        zero_division=0

    )


    recall = recall_score(

        y_test,

        predictions,

        zero_division=0

    )


    f1 = f1_score(

        y_test,

        predictions,

        zero_division=0

    )


    threshold_results.append(

        {

            "Threshold":
                threshold,

            "Precision":
                precision,

            "Recall":
                recall,

            "F1":
                f1

        }

    )


threshold_df = pd.DataFrame(
    threshold_results
)


print(

    threshold_df
    .round(3)
    .to_string(index=False)

)


best_threshold_row = threshold_df.loc[

    threshold_df["F1"].idxmax()

]


best_threshold = float(

    best_threshold_row["Threshold"]

)


print(

    "\nRecommended threshold:",

    best_threshold

)


# ============================================================
# 17. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("13. FEATURE IMPORTANCE")
print("=" * 70)


rf_preprocessor = (

    random_forest_model
    .named_steps["preprocessor"]

)


rf_estimator = (

    random_forest_model
    .named_steps["model"]

)


feature_names = (

    rf_preprocessor
    .get_feature_names_out()

)


feature_importance = (

    rf_estimator
    .feature_importances_

)


importance_df = pd.DataFrame(

    {

        "Feature":
            feature_names,

        "Importance":
            feature_importance

    }

)


importance_df = (

    importance_df
    .sort_values(
        "Importance",
        ascending=False
    )

)


print(

    importance_df
    .head(15)
    .round(4)
    .to_string(index=False)

)


# Save feature importance

importance_path = (

    f"{MODEL_DIR}/feature_importance.csv"

)


importance_df.to_csv(

    importance_path,

    index=False

)


# ============================================================
# 18. FEATURE IMPORTANCE GRAPH
# ============================================================

top_features = (

    importance_df
    .head(10)
    .sort_values(
        "Importance"
    )

)


plt.figure(

    figsize=(9, 6)

)


plt.barh(

    top_features["Feature"],

    top_features["Importance"]

)


plt.xlabel(
    "Importance"
)


plt.ylabel(
    "Feature"
)


plt.title(
    "Top 10 Network Issue Prediction Features"
)


plt.tight_layout()


feature_plot_path = (

    f"{MODEL_DIR}/feature_importance.png"

)


plt.savefig(

    feature_plot_path,

    dpi=150

)


plt.close()


# ============================================================
# 19. CONFUSION MATRIX GRAPH
# ============================================================

cm = final_results[
    "confusion_matrix"
]


plt.figure(

    figsize=(6, 5)

)


plt.imshow(cm)


plt.title(

    f"Confusion Matrix - "
    f"{final_results['model_name']}"

)


plt.xlabel(
    "Predicted"
)


plt.ylabel(
    "Actual"
)


plt.xticks(

    [0, 1],

    [
        "Normal",
        "Network Issue"
    ]

)


plt.yticks(

    [0, 1],

    [
        "Normal",
        "Network Issue"
    ]

)


for i in range(2):

    for j in range(2):

        plt.text(

            j,

            i,

            cm[i, j],

            ha="center",

            va="center"

        )


plt.tight_layout()


confusion_matrix_path = (

    f"{MODEL_DIR}/confusion_matrix.png"

)


plt.savefig(

    confusion_matrix_path,

    dpi=150

)


plt.close()


# ============================================================
# 20. SAVE FINAL MODEL
# ============================================================

model_path = (

    f"{MODEL_DIR}/van_network_issue_model.joblib"

)


joblib.dump(

    final_model,

    model_path

)


# ============================================================
# 21. SAVE METADATA
# ============================================================

metadata = {

    "model_name":
        final_results["model_name"],

    "target":
        TARGET_COLUMN,

    "target_mapping":
        {
            0: "Normal",
            1: "Network Issue"
        },

    "recommended_threshold":
        best_threshold,

    "features":
        X.columns.tolist(),

    "numeric_features":
        numeric_features,

    "categorical_features":
        [
            "Network_Type",
            "Topology"
        ],

    "metrics":
        {

            "accuracy":
                final_results["accuracy"],

            "precision":
                final_results["precision"],

            "recall":
                final_results["recall"],

            "f1":
                final_results["f1"],

            "roc_auc":
                final_results["roc_auc"]

        }

}


metadata_path = (

    f"{MODEL_DIR}/model_metadata.joblib"

)


joblib.dump(

    metadata,

    metadata_path

)


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ML PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 70)


print(

    "\nFinal Model:",

    final_results["model_name"]

)


print(

    "Accuracy:",

    f"{final_results['accuracy']:.2%}"

)


print(

    "Precision:",

    f"{final_results['precision']:.2%}"

)


print(

    "Recall:",

    f"{final_results['recall']:.2%}"

)


print(

    "F1 Score:",

    f"{final_results['f1']:.2%}"

)


print(

    "ROC-AUC:",

    f"{final_results['roc_auc']:.2%}"

)


print(

    "\nRecommended Threshold:",

    best_threshold

)


print("\nGenerated files:")


print(
    model_path
)


print(
    metadata_path
)


print(
    importance_path
)


print(
    feature_plot_path
)


print(
    confusion_matrix_path
)


print("\nDone! 🚀")