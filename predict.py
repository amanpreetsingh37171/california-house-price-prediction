import joblib
import pandas as pd

# 1. Load the saved pipeline + model (you must have saved them in File 1)
model = joblib.load("model.pkl")  # change name if different
pipeline = joblib.load("pipeline.pkl")  # preprocessing pipeline

# 2. Example: Take input from user (manual)
def predict_from_input():
    print("Enter the house features:")

    longitude = float(input("Longitude: "))
    latitude = float(input("Latitude: "))
    housing_median_age = float(input("Median Age of Houses: "))
    total_rooms = float(input("Total Rooms: "))
    total_bedrooms = float(input("Total Bedrooms: "))
    population = float(input("Population: "))
    households = float(input("Households: "))
    median_income = float(input("Median Income: "))
    ocean_proximity = input("Ocean Proximity (e.g. <1H OCEAN, INLAND, NEAR OCEAN, ISLAND): ")

    # Put inputs into a DataFrame
    new_data = pd.DataFrame([{
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

    # 3. Preprocess + predict
    prepared_data = pipeline.transform(new_data)
    prediction = model.predict(prepared_data)

    print(f"\n💰 Predicted House Price: ${prediction[0]:,.2f}")

# 4. Run
if __name__ == "__main__":
    predict_from_input()
