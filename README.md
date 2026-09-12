# 🔥 Forest Fire Risk Prediction

A Machine Learning application for predicting **forest fire risk** using environmental, temporal, and geographic features.

The project compares multiple classifiers, explains feature importance with SHAP, selects a compact Top-10 feature set, tunes the probability threshold, and deploys the final model through an interactive Streamlit interface.

## 🎯 Project Objective

Build a reliable binary classifier that predicts:

- `0` → No Fire
- `1` → Fire

The application also displays:

- Fire probability
- Risk level
- Final Fire / No Fire decision
- Interactive Plotly gauge

## 🧠 Models Evaluated

Three classification models were evaluated:

1. **Decision Tree**
2. **SVM (RBF)**
3. **XGBoost**

XGBoost was selected as the strongest model among the evaluated classifiers.

## 🔍 Data Processing

The dataset contained:

- Environmental / remote-sensing features
- Temporal features
- Geographic features

Main preprocessing steps:

- Removed duplicate rows
- Converted `-99999` placeholder values to missing values
- Handled missing values using median imputation
- Used stratified 80/20 train-test splitting
- Used SMOTE only on the training data to address class imbalance
- Extracted latitude and longitude from `.geo`
- Engineered `year` and `day_of_year` from the original date

## 🧩 SHAP Feature Selection

SHAP (SHapley Additive exPlanations) was used to rank features according to their contribution to the XGBoost predictions.

The final model uses these **Top-10 features**:

```text
precmax
day_of_year
year
precsum
precmin
ndvimedian
ndwimin
days
latitude
slope
```

Using the Top-10 features reduces the input complexity while maintaining strong predictive performance.

## 📊 Final Model

The final application uses **Top-10 XGBoost**.

Selected decision threshold:

```text
0.30
```

The threshold was selected using validation data with **F2-score**, giving more importance to Recall because missing a real fire is more critical in fire detection.

### Final test performance at threshold = 0.30

| Metric | Score |
|---|---:|
| Accuracy | 93.84% |
| Precision | 76.96% |
| Recall | 90.32% |
| F1-Score | 83.11% |
| Top-10 ROC-AUC | 0.9764 |

## 🚦 Risk Levels in the UI

The UI uses the Fire probability to communicate risk:

| Fire Probability | Risk Level |
|---|---|
| `< 30%` | 🟢 Low Risk |
| `30% – < 75%` | 🟡 Medium Risk |
| `≥ 75%` | 🔴 High Risk |

Important: the **risk bands** are for communication in the UI, while the **binary Fire / No Fire decision** uses the validated threshold of `0.30`.

## 🖥️ Streamlit Application

The application allows the user to enter:

- Maximum precipitation
- Total precipitation
- Minimum precipitation
- NDVI median
- NDWI minimum
- Days
- Latitude
- Slope
- Date

The date is used to derive:

- `year`
- `day_of_year`

The model then calculates the Fire probability and produces the final prediction and risk level.

## 📁 Project Structure

```text
Forest-Fire-Risk-Prediction/
│
├── app.py
├── forest_fire_final_model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Run Locally

Create and activate a virtual environment if needed, then install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## ☁️ Deployment

This project can be deployed using **Streamlit Community Cloud** directly from GitHub.

1. Push the project files to a GitHub repository.
2. Make sure `app.py`, `requirements.txt`, and `forest_fire_final_model.pkl` are included.
3. Open Streamlit Community Cloud.
4. Connect your GitHub account.
5. Create a new app and select:
   - Repository
   - Branch
   - `app.py` as the entrypoint
6. Deploy.

## ⚠️ Model File Note

The application requires:

```text
forest_fire_final_model.pkl
```

This file contains:

- The trained Top-10 XGBoost pipeline
- The Top-10 feature list
- The final decision threshold

If the model file is too large for normal GitHub storage, use **Git LFS** for the `.pkl` file.

## 🛠️ Technologies

- Python
- Pandas
- Scikit-learn
- XGBoost
- Imbalanced-learn / SMOTE
- SHAP
- Joblib
- Plotly
- Streamlit

## 👨‍💻 Project Type

**NTI Final Project — Machine Learning + Explainability + Deployment**

---

### Main takeaway

The system is designed to prioritize the detection of real fire cases while keeping the model practical and explainable through SHAP and an interactive Streamlit interface.
