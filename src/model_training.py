import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from data_cleaning import clean_data
from feature_engineering import add_features
from data_preprocessing import TARGET, build_preprocessor

def train_model():
    df = add_features(clean_data())
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("regressor", RandomForestRegressor(
            n_estimators=60, min_samples_leaf=3, max_depth=24,
            max_features=0.8, random_state=42, n_jobs=-1
        )),
    ])
    model.fit(X_train, y_train)
    joblib.dump(model, "models/car_price_model.joblib")
    pd.DataFrame({"actual_price": y_test, "predicted_price": model.predict(X_test)}).to_csv(
        "reports/test_predictions.csv", index=False
    )
    print("Model sačuvan u models/car_price_model.joblib")
    return model, X_test, y_test

if __name__ == "__main__":
    train_model()
