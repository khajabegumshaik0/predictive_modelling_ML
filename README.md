# 📊 ML Predictive Modeling Studio

A modern, high-end web application that trains machine learning models to classify loan application approvals in real-time, built with Python (Flask, Scikit-Learn) and a responsive, glassmorphic frontend interface.

## 🚀 Features

- **Automated Pipeline**: Automatically trains three machine learning algorithms on startup:
  - Logistic Regression
  - Decision Tree
  - Random Forest
- **Real-Time Classification**: Input customer parameters (Age, Annual Income, Credit Score, and Loan Amount) to evaluate loan approval state.
- **Dynamic Visualizations**: View and switch between interactive evaluation charts without page reloads:
  - **Confusion Matrix** (best performing model)
  - **ROC Curve** (model sensitivity)
  - **Model Accuracies** (accuracy comparison metrics)
- **Glassmorphic UI**: Beautiful dark mode dashboard with fluid transitions, real-time feedback, validation checks, and responsive design.
- **Dataset Inspector**: Live table displaying the underlying training dataset.

---

## 🛠️ Tech Stack

**Backend:**
- Python 3.x
- Flask (Web Server & REST APIs)
- Scikit-Learn (Data Scaling, Classification Models & Metrics)
- Matplotlib & Seaborn (Chart Generation)
- Pandas & NumPy (Data Structuring & Arrays)

**Frontend:**
- HTML5 (Semantic Structure)
- CSS3 (Vanilla design utilizing custom properties, glassmorphism, flexbox/grid layout, animations)
- JavaScript (Fetch API, DOM manipulation, dynamic charts rendering)

---

## ⚙️ Getting Started

### 1. Prerequisites
Ensure you have Python 3.x and `pip` installed.

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/predictive-modeling-ml.git
cd predictive-modeling-ml
```

### 3. Install Dependencies
Install all the required machine learning and web server libraries:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn flask
```

### 4. Run the Application
Start the Flask development server:
```bash
python app.py
```

### 5. Access the Dashboard
Once the server starts, open your web browser and navigate to:
👉 **[http://localhost:5000](http://localhost:5000)**

---

## 📂 Project Structure

```
├── app.py                  # Flask server, ML models training & API endpoints
├── templates/
│   └── index.html          # Frontend dashboard UI (HTML, CSS, JS)
├── .gitignore              # Files to ignore in Git version control
└── README.md               # Project documentation
```

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
