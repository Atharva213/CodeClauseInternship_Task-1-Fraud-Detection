import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('creditcard.csv')

data = load_data()

# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Split data into features and target
X = data.drop(columns="Class", axis=1)
y = data["Class"]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=2)

# Model Selection
model_name = st.sidebar.selectbox("Select Model", ("Random Forest", "Gradient Boosting"))

# Train the selected model
if model_name == "Random Forest":
    model = RandomForestClassifier(n_estimators=100, random_state=2)
    model.fit(X_train, y_train)
else:
    model = GradientBoostingClassifier(n_estimators=100, random_state=2)
    model.fit(X_train, y_train)

# Evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)
test_report = classification_report(y_test, model.predict(X_test), target_names=['Legitimate', 'Fraudulent'])

# Display model performance metrics
st.title("Credit Card Fraud Detection Model")
st.subheader(f"{model_name} Model Performance Metrics")
st.write(f"Training Accuracy: {train_acc:.4f}")
st.write(f"Test Accuracy: {test_acc:.4f}")
st.write("Test Classification Report:")
st.text_area("Classification Report:", value=test_report, height=200)

# Visualization of data distribution
st.subheader("Data Distribution")
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x='Class', data=data, ax=ax, hue='Class', palette='Set2', legend=False)
ax.set_xlabel('Class', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Distribution of Classes', fontsize=15)
st.pyplot(fig)

# Additional visualizations
st.subheader("Additional Visualizations")
st.write("Distribution of Transaction Amount by Class")
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(data=data, x='Amount', hue='Class', kde=True, ax=ax, palette='Set2', legend=False)
ax.set_xlabel('Transaction Amount', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Distribution of Transaction Amount by Class', fontsize=15)
st.pyplot(fig)

st.write("Correlation Heatmap")
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
st.pyplot(plt)

st.write("Scatter Plot of Time vs. Amount by Class")
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x='Time', y='Amount', data=data, hue='Class', ax=ax, palette='Set2')
ax.set_xlabel('Time (seconds)', fontsize=12)
ax.set_ylabel('Amount', fontsize=12)
ax.set_title('Time vs. Amount by Class', fontsize=15)
st.pyplot(fig)
