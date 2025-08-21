import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd

# Load the trained model and pipeline
model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

# --- Prediction Function ---
def predict_price():
    try:
        # Collect inputs
        longitude = float(entry_longitude.get())
        latitude = float(entry_latitude.get())
        housing_median_age = float(entry_age.get())
        total_rooms = float(entry_rooms.get())
        total_bedrooms = float(entry_bedrooms.get())
        population = float(entry_population.get())
        households = float(entry_households.get())
        median_income = float(entry_income.get())
        ocean_proximity = combo_proximity.get()

        # Prepare input data
        input_data = pd.DataFrame([{
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "median_income": median_income,
            "ocean_proximity": ocean_proximity
        }])

        # Transform and predict
        transformed_data = pipeline.transform(input_data)
        prediction = model.predict(transformed_data)[0]

        # Show result
        result_label.config(
            text=f"🏡 Predicted House Price: ${prediction:,.2f}",
            foreground="green",
            font=("Helvetica", 14, "bold")
        )
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input! {e}")


# --- GUI Setup ---
root = tk.Tk()
root.title("California House Price Predictor")
root.geometry("600x650")
root.configure(bg="#f0f4f7")

title_label = tk.Label(
    root,
    text="🏠 California House Price Predictor",
    font=("Helvetica", 18, "bold"),
    bg="#4a90e2",
    fg="white",
    pady=10
)
title_label.pack(fill="x")

frame = tk.Frame(root, bg="#f0f4f7", padx=20, pady=20)
frame.pack(pady=10, fill="both", expand=True)

# Input fields
fields = [
    ("Longitude", -122.23),
    ("Latitude", 37.88),
    ("Housing Median Age", 30),
    ("Total Rooms", 1500),
    ("Total Bedrooms", 300),
    ("Population", 500),
    ("Households", 200),
    ("Median Income (10k USD)", 5.0)
]

entries = {}
for i, (label, default) in enumerate(fields):
    tk.Label(frame, text=label, bg="#f0f4f7", font=("Helvetica", 11)).grid(row=i, column=0, sticky="w", pady=5)
    entry = ttk.Entry(frame)
    entry.grid(row=i, column=1, pady=5, padx=10)
    entry.insert(0, str(default))
    entries[label] = entry

entry_longitude = entries["Longitude"]
entry_latitude = entries["Latitude"]
entry_age = entries["Housing Median Age"]
entry_rooms = entries["Total Rooms"]
entry_bedrooms = entries["Total Bedrooms"]
entry_population = entries["Population"]
entry_households = entries["Households"]
entry_income = entries["Median Income (10k USD)"]

# Ocean Proximity Dropdown
tk.Label(frame, text="Ocean Proximity 🌊", bg="#f0f4f7", font=("Helvetica", 11)).grid(row=len(fields), column=0, sticky="w", pady=5)
combo_proximity = ttk.Combobox(frame, values=["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"])
combo_proximity.grid(row=len(fields), column=1, pady=5, padx=10)
combo_proximity.current(0)

# Predict Button
predict_btn = tk.Button(
    root,
    text="🚀 Predict Price",
    command=predict_price,
    bg="#4a90e2",
    fg="white",
    font=("Helvetica", 12, "bold"),
    relief="raised",
    padx=10,
    pady=5
)
predict_btn.pack(pady=20)

# Result Label
result_label = tk.Label(root, text="", bg="#f0f4f7", font=("Helvetica", 12))
result_label.pack(pady=10)

root.mainloop()
