# ==================================================
# Predictive Modeling & EDA Backend
# Flask application serving ML models, API endpoints, and EDA
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

# ==========================================
# 1. Rich Synthetic Dataset Generation (200 rows)
# ==========================================
np.random.seed(42)
n_samples = 200

# Independent numerical variables
age = np.random.normal(38, 11, n_samples).astype(int)
age = np.clip(age, 18, 70)

income = np.random.normal(68000, 22000, n_samples).astype(int)
income = np.clip(income, 20000, 150000)

credit_score = np.random.normal(690, 85, n_samples).astype(int)
credit_score = np.clip(credit_score, 300, 850)

dti = np.random.uniform(0.1, 0.6, n_samples)  # Debt-to-income ratio (10% to 60%)

# Loan Amount: positively correlated with income, with some noise
loan_amount = (income * np.random.uniform(0.12, 0.38, n_samples)).astype(int)
loan_amount = np.clip(loan_amount, 5000, 50000)

# Determine approval based on credit score, income, and DTI
norm_cs = (credit_score - 300) / 550.0  # scale 0 to 1
norm_inc = (income - 20000) / 130000.0  # scale 0 to 1
norm_dti = (0.6 - dti) / 0.5  # scale 0 to 1 (lower DTI is better)

# Combined credit scoring formula
score = 0.5 * norm_cs + 0.3 * norm_inc + 0.2 * norm_dti
# Add small Gaussian noise to make it realistic
noise = np.random.normal(0, 0.07, n_samples)
final_score = score + noise

# 1 = Approved, 0 = Rejected
approved = (final_score > 0.46).astype(int)

# Create DataFrame
data = {
    "Age": age.tolist(),
    "Income": income.tolist(),
    "CreditScore": credit_score.tolist(),
    "DebtToIncome": np.round(dti, 3).tolist(),
    "LoanAmount": loan_amount.tolist(),
    "Approved": approved.tolist()
}

df = pd.DataFrame(data)

# ==========================================
# 2. Data Splitting, Scaling, & Model Pipeline
# ==========================================
X = df.drop("Approved", axis=1)
y = df["Approved"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scaling
scaler = StandardScaler()
# Fit on training data with feature names
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), 
    columns=X_train.columns
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), 
    columns=X_test.columns
)

# Initialize Models
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
}

# Train models
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
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=140)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"

# ==============================
# Web Server Routes
# ==============================

@app.route('/')
def home():
    return render_template('index.html')

# ML API Endpoints
@app.route('/api/metrics', methods=['GET'])
def get_metrics():
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
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(f"Confusion Matrix ({best_model_name})", fontsize=13, pad=15)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plots['confusion_matrix'] = fig_to_base64(fig)
    
    # 2. ROC Curve
    fig, ax = plt.subplots(figsize=(6, 4.5))
    if hasattr(best_model, "predict_proba"):
        prob = best_model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        score = auc(fpr, tpr)
        ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"AUC = {score:.2f}")
    else:
        ax.text(0.5, 0.5, "ROC Curve Not Supported", ha='center', va='center')
    
    ax.plot([0, 1], [0, 1], color="#9ca3af", linestyle="--")
    ax.set_title(f"ROC Curve ({best_model_name})", fontsize=13, pad=15)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)
    plots['roc_curve'] = fig_to_base64(fig)
    
    # 3. Model Accuracy Comparison
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = list(model_accuracies.keys())
    accs = [val * 100 for val in model_accuracies.values()]
    colors = ['#3b82f6' if n == best_model_name else '#94a3b8' for n in names]
    
    bars = ax.bar(names, accs, color=colors, width=0.5)
    ax.set_title("Model Accuracy Comparison (%)", fontsize=13, pad=15)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 110)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
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
        dti_val = float(req_data['dti'])
        loan_amount = float(req_data['loan_amount'])
        
        # Prepare inputs as a pandas DataFrame matching scaled training format
        raw_inputs = pd.DataFrame([[age, income, credit_score, dti_val, loan_amount]], 
                                  columns=["Age", "Income", "CreditScore", "DebtToIncome", "LoanAmount"])
        scaled_inputs = scaler.transform(raw_inputs)
        
        # Run prediction
        pred_class = int(best_model.predict(scaled_inputs)[0])
        
        # Probability
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

# ==============================
# EDA API Endpoints
# ==============================

@app.route('/api/eda/summary', methods=['GET'])
def get_eda_summary():
    # 1. Descriptive stats
    desc = df.describe().to_dict()
    
    # 2. Correlation matrix
    corr = df.corr().to_dict()
    
    # 3. Key influencing factors (features sorted by correlation with "Approved")
    approved_corr = df.corr()["Approved"].drop("Approved").to_dict()
    influencing_factors = sorted(
        [{"feature": k, "correlation": round(v, 3)} for k, v in approved_corr.items()],
        key=lambda x: abs(x["correlation"]),
        reverse=True
    )
    
    # 4. Summary counts
    total_approved = int(df["Approved"].sum())
    total_rejected = len(df) - total_approved
    approval_rate = round((total_approved / len(df)) * 100, 1)
    
    return jsonify({
        "summary_statistics": desc,
        "correlation_matrix": corr,
        "influencing_factors": influencing_factors,
        "counts": {
            "total_records": len(df),
            "approved": total_approved,
            "rejected": total_rejected,
            "approval_rate": approval_rate
        }
    })

@app.route('/api/eda/plots', methods=['GET'])
def get_eda_plots():
    plots = {}
    
    # Plot 1: Correlation Heatmap
    fig, ax = plt.subplots(figsize=(6.5, 5))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", 
                linewidths=0.5, cbar=True, ax=ax, annot_kws={"size": 10})
    ax.set_title("Correlation Heatmap Matrix", fontsize=13, pad=15)
    plots['correlation_heatmap'] = fig_to_base64(fig)
    
    # Plot 2: Credit Score Distribution by Approval Status
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.histplot(data=df, x="CreditScore", hue="Approved", multiple="stack", 
                 kde=True, palette={0: "#ef4444", 1: "#10b981"}, bins=15, ax=ax)
    ax.set_title("Credit Score Distribution by Approval", fontsize=13, pad=15)
    ax.set_xlabel("Credit Score", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    plots['credit_distribution'] = fig_to_base64(fig)
    
    # Plot 3: Income vs Credit Score Scatter colored by Approval Status
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.scatterplot(data=df, x="CreditScore", y="Income", hue="Approved",
                    palette={0: "#ef4444", 1: "#10b981"}, alpha=0.8, s=60, ax=ax)
    ax.set_title("Income vs Credit Score Analysis", fontsize=13, pad=15)
    ax.set_xlabel("Credit Score", fontsize=11)
    ax.set_ylabel("Annual Income ($)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    plots['income_vs_credit'] = fig_to_base64(fig)
    
    # Plot 4: DTI Ratio distribution by Approval
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.boxplot(data=df, x="Approved", y="DebtToIncome", 
                palette={0: "#ef4444", 1: "#10b981"}, ax=ax)
    ax.set_title("Debt-to-Income (DTI) Boxplot", fontsize=13, pad=15)
    ax.set_xticklabels(["Rejected (0)", "Approved (1)"])
    ax.set_ylabel("Debt-to-Income Ratio", fontsize=11)
    plots['dti_boxplot'] = fig_to_base64(fig)
    
    return jsonify(plots)

if __name__ == '__main__':
    print("Starting ML Predictive Modeling & EDA Studio on http://localhost:5000...")
    app.run(host='0.0.0.0', port=5000, debug=True)
