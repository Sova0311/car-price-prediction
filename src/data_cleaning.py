import pandas as pd

RENAME_MAP = {
    "mileage(kilometers)": "mileage_km",
    "volume(cm3)": "volume_cm3",
}

def clean_data(input_path="data/cars.csv", output_path="data/cars_clean.csv"):
    df = pd.read_csv(input_path)
    df = df.rename(columns=RENAME_MAP)
    df.columns = [c.strip().lower() for c in df.columns]

    # Normalize text categories and treat blank strings as missing.
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip().str.lower().replace("", pd.NA)

    # Remove rows that cannot be used for supervised price prediction.
    df = df.dropna(subset=["priceusd", "year", "mileage_km"])
    df = df[(df["priceusd"] > 0) & (df["priceusd"] <= 500_000)]
    df = df[df["year"].between(1950, 2026)]
    df = df[df["mileage_km"].between(0, 1_500_000)]

    # Implausible engine sizes are treated as missing rather than deleting the car.
    df.loc[~df["volume_cm3"].between(500, 10_000), "volume_cm3"] = pd.NA
    df.to_csv(output_path, index=False)
    print(f"Sačuvano {len(df):,} redova u {output_path}")
    return df

if __name__ == "__main__":
    clean_data()
