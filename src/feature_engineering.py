import pandas as pd

REFERENCE_YEAR = 2026

def add_features(df):
    df = df.copy()
    df["car_age"] = (REFERENCE_YEAR - df["year"]).clip(lower=0)
    df["mileage_per_year"] = df["mileage_km"] / df["car_age"].clip(lower=1)
    df["engine_volume_liters"] = df["volume_cm3"] / 1000
    df["brand_model"] = df["make"].fillna("unknown") + "_" + df["model"].fillna("unknown")
    df["is_newer_car"] = (df["car_age"] <= 5).astype(int)
    df["is_high_mileage"] = (df["mileage_km"] >= 250_000).astype(int)
    return df

if __name__ == "__main__":
    from data_cleaning import clean_data
    data = clean_data()
    add_features(data).to_csv("data/cars_features.csv", index=False)
