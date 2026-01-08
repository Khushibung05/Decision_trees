import streamlit as slt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------- PAGE CONFIG ---------------- #
slt.set_page_config("Bank Deposit Prediction App", layout="centered")

# ---------------- LOAD CSS ---------------- #
def load_css(file):
    with open(file) as f:
        slt.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles.css")

# ---------------- TITLE ---------------- #
slt.markdown("""
<div class="card">
        <h1>🏦 Bank Deposit Prediction</h1>
        <p>Predict whether a customer will <b>Subscribe to a Term Deposit</b> using <b>Decision Tree</b></p>
</div>  
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ---------------- #
@slt.cache_data
def load_data():
    return pd.read_csv("bank_marketing_dataset.csv")

df = load_data()

# ---------------- DATASET PREVIEW ---------------- #
slt.markdown('<div class="card">', unsafe_allow_html=True)
slt.subheader("Dataset Preview")
slt.dataframe(df.head())
slt.markdown('</div>', unsafe_allow_html=True)

# ---------------- OUTLIER HANDLING ---------------- #
num_cols = df.select_dtypes(include=["int64", "float64"]).columns

def cap_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower, upper)
    return df

for col in num_cols:
    df = cap_outliers_iqr(df, col)

# ---------------- PREPARE DATA ---------------- #
X = df.drop("deposit", axis=1)
y = df["deposit"]

le = LabelEncoder()
encoders = {}

for col in X.columns:
    if X[col].dtype == "object":
        encoders[col] = LabelEncoder()
        X[col] = encoders[col].fit_transform(X[col])

y = le.fit_transform(y)  # yes=1, no=0

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- TRAIN MODEL ---------------- #
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=6,
    random_state=42
)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# ---------------- METRICS ---------------- #
accuracy = accuracy_score(y_test, y_pred)

slt.markdown('<div class="card">', unsafe_allow_html=True)
slt.subheader("Model Performance")

c1, c2 = slt.columns(2)
c1.metric("Accuracy", f"{accuracy:.2f}")
c2.metric("Tree Depth", model.get_depth())

slt.markdown('</div>', unsafe_allow_html=True)

# ---------------- CONFUSION MATRIX ---------------- #
slt.markdown('<div class="card">', unsafe_allow_html=True)
slt.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="g", cmap="magma", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
slt.pyplot(fig)

slt.markdown('</div>', unsafe_allow_html=True)

# ---------------- DECISION TREE VISUAL ---------------- #
slt.markdown('<div class="card">', unsafe_allow_html=True)
slt.subheader("Decision Tree Visualization")

fig, ax = plt.subplots(figsize=(14, 6))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No Deposit", "Deposit"],
    filled=True,
    ax=ax
)
slt.pyplot(fig)

slt.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREDICTION INPUT ---------------- #
slt.markdown('<div class="card">', unsafe_allow_html=True)
slt.subheader("Predict for a New Customer")

user_input = {}

for col in X.columns:
    if col in encoders:   # categorical
        user_input[col] = slt.selectbox(col, encoders[col].classes_)
    else:                 # numerical
        user_input[col] = slt.slider(
            col,
            float(X[col].min()),
            float(X[col].max()),
            float(X[col].mean())
        )

input_df = pd.DataFrame([user_input])

for col in encoders:
    input_df[col] = encoders[col].transform(input_df[col])

prediction = model.predict(input_df)[0]

if prediction == 1:
    slt.markdown(
        '<div class="prediction-box">✅ Customer is LIKELY to SUBSCRIBE</div>',
        unsafe_allow_html=True
    )
else:
    slt.markdown(
        '<div class="prediction-box">❌ Customer is NOT likely to subscribe</div>',
        unsafe_allow_html=True
    )

slt.markdown('</div>', unsafe_allow_html=True)
