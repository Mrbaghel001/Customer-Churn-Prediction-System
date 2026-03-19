from data_preprocessing import load_and_preprocess
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load data
df = load_and_preprocess("data/churn.csv")

X = df.drop('Churn', axis=1)
y = df['Churn']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model
with open("model/churn_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save column names
with open("model/columns.pkl", "wb") as f:
    pickle.dump(X.columns, f)

print("Model trained and saved!")