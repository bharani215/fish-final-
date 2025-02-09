from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load dataset
df = pd.read_csv(r"D:\final year project\archive\realfishdataset.csv")  # ✅ Correct

features = ["ph", "temperature", "turbidity"]
target = "fish"

X = df[features]
y = df[target]

# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    input_data = np.array([[data["ph"], data["temperature"], data["turbidity"]]])
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    fish_type = label_encoder.inverse_transform(prediction)
    return jsonify({"fish": fish_type[0]})

if __name__ == "__main__":
    app.run(debug=True)
