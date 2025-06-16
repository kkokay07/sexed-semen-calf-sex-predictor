import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
import joblib

# Load data
data = pd.read_csv("data_cs.txt", sep="\t")

# Features and target
features = ['Breed', 'Parity', 'Management_Systems']
target = 'Calf_sex'

# Label encoding for all categorical variables
encoders = {col: LabelEncoder().fit(data[col]) for col in features}
X = pd.DataFrame({col: encoders[col].transform(data[col]) for col in features})
y_encoder = LabelEncoder().fit(data[target])
y = y_encoder.transform(data[target])

# Train best model (you can swap with ExtraTrees or DecisionTree if desired)
model = MLPClassifier(max_iter=1000, random_state=42)
model.fit(X, y)

# Save model and encoders for future use
joblib.dump(model, "best_model.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(y_encoder, "target_encoder.pkl")

# GUI Application
def predict_calf_sex():
    try:
        breed_val = breed_var.get()
        parity_val = parity_var.get()
        mgmt_val = mgmt_var.get()

        input_df = pd.DataFrame({
            'Breed': [breed_val],
            'Parity': [parity_val],
            'Management_Systems': [mgmt_val]
        })

        # Encode using previously fitted encoders
        input_encoded = pd.DataFrame({
            col: [encoders[col].transform(input_df[col])[0]]
            for col in features
        })

        pred = model.predict(input_encoded)[0]
        pred_label = y_encoder.inverse_transform([pred])[0]
        messagebox.showinfo("Prediction", f"Predicted Calf Sex: {pred_label}")
    except Exception as e:
        messagebox.showerror("Error", f"Prediction failed: {str(e)}")

# GUI Setup
root = tk.Tk()
root.title("Calf Sex Predictor")
root.geometry("400x300")
root.configure(bg="#f0f0f0")

tk.Label(root, text="Select Breed:", bg="#f0f0f0").pack(pady=5)
breed_var = tk.StringVar()
breed_menu = ttk.Combobox(root, textvariable=breed_var, state="readonly")
breed_menu['values'] = sorted(data['Breed'].unique())
breed_menu.pack(pady=5)

tk.Label(root, text="Select Parity:", bg="#f0f0f0").pack(pady=5)
parity_var = tk.StringVar()
parity_menu = ttk.Combobox(root, textvariable=parity_var, state="readonly")
parity_menu['values'] = sorted(data['Parity'].unique())
parity_menu.pack(pady=5)

tk.Label(root, text="Select Management System:", bg="#f0f0f0").pack(pady=5)
mgmt_var = tk.StringVar()
mgmt_menu = ttk.Combobox(root, textvariable=mgmt_var, state="readonly")
mgmt_menu['values'] = sorted(data['Management_Systems'].unique())
mgmt_menu.pack(pady=5)

tk.Button(root, text="Predict Calf Sex", command=predict_calf_sex,
          bg="#4CAF50", fg="white", padx=10, pady=5).pack(pady=20)

root.mainloop()

