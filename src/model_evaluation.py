import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model, X_test, y_test, output_path="reports/evaluation.txt"):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    report = (
        f"MAE:  ${mae:,.2f}\n"
        f"MSE:  {mse:,.2f}\n"
        f"RMSE: ${rmse:,.2f}\n"
        f"R2:   {r2:.4f}\n"
    )
    print(report)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

if __name__ == "__main__":
    import joblib
    from sklearn.model_selection import train_test_split
    from data_cleaning import clean_data
    from feature_engineering import add_features
    from data_preprocessing import TARGET
    df = add_features(clean_data())
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    model = joblib.load("models/car_price_model.joblib")
    evaluate_model(model, X_test, y_test)
