# ❤️ Heart Disease Detection

This project predicts the likelihood of heart disease in patients using machine learning techniques.  
It leverages **pandas** for data manipulation, **scikit-learn** for building models, and visualization libraries for insights.

---

## 🚀 Features
- Clean and preprocess medical datasets (CSV, Excel)
- Perform exploratory data analysis (EDA) with pandas
- Train ML models (Logistic Regression, Random Forest, etc.)
- Evaluate performance with accuracy, precision, recall, and ROC-AUC
- Predict heart disease risk for new patient data

---

## 📂 Dataset
The dataset typically includes:
- **Age**
- **Sex**
- **Chest Pain Type (cp)**
- **Resting Blood Pressure (trestbps)**
- **Cholesterol (chol)**
- **Fasting Blood Sugar (fbs)**
- **Resting ECG (restecg)**
- **Maximum Heart Rate Achieved (thalach)**
- **Exercise Induced Angina (exang)**
- **Oldpeak (ST depression)**
- **Slope of the ST segment (slope)**
- **Number of major vessels (ca)**
- **Thalassemia (thal)**
- **Target (1 = heart disease, 0 = no heart disease)**

*(Dataset source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease))*

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/heart-disease-detection.git
cd heart-disease-detection
Create a virtual environment:

bash
Copy code
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
Install dependencies:

bash
Copy code
pip install -r requirements.txt
🛠 Usage
Run the project:

bash
Copy code
python main.py
Example usage inside Python:

python
Copy code
import pandas as pd
from model import HeartDiseaseModel

# Load dataset
df = pd.read_csv("data/heart.csv")

# Train model
model = HeartDiseaseModel()
model.train(df)

# Predict on new patient data
sample_patient = [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]
prediction = model.predict(sample_patient)

print("Heart Disease Risk:", "Yes" if prediction[0] == 1 else "No")
📂 Project Structure
bash
Copy code
heart-disease-detection/
├── data/                # Dataset (heart.csv)
├── notebooks/           # Jupyter notebooks for EDA
├── src/                 # Source code
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── model.py
│   └── utils.py
├── tests/               # Unit tests
├── requirements.txt     # Python dependencies
├── main.py              # Entry point for training & prediction
└── README.md            # Project documentation
📊 Model Performance
Accuracy: ~85% (Logistic Regression baseline)

ROC-AUC: ~0.90 (Random Forest)

(Performance may vary depending on dataset and preprocessing steps.)

🤝 Contributing
Contributions are welcome!

Fork the repo

Create a new branch (git checkout -b feature-name)

Commit your changes

Push the branch (git push origin feature-name)

Open a Pull Request

📜 License
This project is licensed under the MIT License.

yaml
Copy code

---

Would you like me to also create a **`requirements.txt`** file for this project with all the necessary libraries (pandas, numpy, scikit-learn, matplotlib, etc.) so you can run it right away?
