# ==================================================
# Predictive Modeling Web Backend
# Flask application serving ML models and API endpoints
# ==================================================

import os
import io
import base64
import pandas as pd
import numpy as np

# Set matplotlib backend to non-interactive 'Agg' BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, jsonify, request, render_template

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

app = Flask(__name__)

# ==============================
# ML Model Pipeline Setup
# ==============================

# 1. Dataset Generation
data = {
    "Age": [25, 35, 45, 23, 40, 30, 50, 28, 38, 55, 22, 33],
    "Income": [30000, 60000, 80000, 25000, 70000, 40000, 90000, 35000, 65000, 100000, 20000, 50000],
    "CreditScore": [650, 750, 800, 600, 720, 680, 820, 640, 760, 850, 580, 700],
    "LoanAmount": [5000, 10000, 20000, 7000, 15000, 8000, 25000, 6000, 12000, 30000, 5000, 9000],
    "Approved": [0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

# 2. Data Splitting & Scaling
X = df.drop("Approved", axis=1)
y = df["Approved"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Initialize Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Train models on startup
trained_models = {}
model_accuracies = {}
model_reports = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    
    trained_models[name] = model
    model_accuracies[name] = float(acc)
    model_reports[name] = classification_report(y_test, preds, output_dict=True)

# Select Best Model
best_model_name = max(model_accuracies, key=model_accuracies.get)
best_model = trained_models[best_model_name]

# Helper function to convert plots to base64 images
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"

# ==============================
# API Routes
# ==============================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    # Return basic training data and performance
    dataset_records = df.to_dict(orient='records')
    return jsonify({
        "best_model": best_model_name,
        "accuracies": model_accuracies,
        "classification_reports": model_reports,
        "dataset": dataset_records
    })

@app.route('/api/plots', methods=['GET'])
def get_plots():
    plots = {}
    
    # 1. Confusion Matrix
    y_pred = best_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 4.5))
    # Style styling to match dark theme
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(f"Confusion Matrix ({best_model_name})", fontsize=14, pad=15)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plots['confusion_matrix'] = fig_to_base64(fig)
    
    # 2. ROC Curve
    fig, ax = plt.subplots(figsize=(6, 4.5))
    if hasattr(best_model, "predict_proba"):
        prob = best_model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        score = auc(fpr, tpr)
        ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"AUC = {score:.2f}")
    else:
        # Fallback if best model doesn't support probabilities
        ax.text(0.5, 0.5, "ROC Curve Not Supported", ha='center', va='center')
    
    ax.plot([0, 1], [0, 1], color="#9ca3af", linestyle="--")
    ax.set_title(f"ROC Curve ({best_model_name})", fontsize=14, pad=15)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)
    plots['roc_curve'] = fig_to_base64(fig)
    
    # 3. Model Accuracy Comparison
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = list(model_accuracies.keys())
    accs = [val * 100 for val in model_accuracies.values()]
    colors = ['#3b82f6' if n == best_model_name else '#94a3b8' for n in names]
    
    bars = ax.bar(names, accs, color=colors, width=0.5)
    ax.set_title("Model Accuracy Comparison (%)", fontsize=14, pad=15)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 110)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, weight='bold')
    
    plots['accuracy_comparison'] = fig_to_base64(fig)
    
    return jsonify(plots)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        req_data = request.get_json()
        age = float(req_data['age'])
        income = float(req_data['income'])
        credit_score = float(req_data['credit_score'])
        loan_amount = float(req_data['loan_amount'])
        
        # Prepare inputs and apply the scaler
        raw_inputs = np.array([[age, income, credit_score, loan_amount]])
        scaled_inputs = scaler.transform(raw_inputs)
        
        # Run prediction
        pred_class = int(best_model.predict(scaled_inputs)[0])
        
        # Probability (if supported)
        probability = None
        if hasattr(best_model, "predict_proba"):
            probability = float(best_model.predict_proba(scaled_inputs)[0][1])
            
        return jsonify({
            "status": "success",
            "prediction": "Approved" if pred_class == 1 else "Rejected",
            "approved": pred_class == 1,
            "probability": probability,
            "best_model_used": best_model_name
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

if __name__ == '__main__':
    print("Starting ML Predictive Modeling Flask App on http://localhost:5000...")
    app.run(host='0.0.0.0', port=5000, debug=True)
