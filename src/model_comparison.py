import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_cleaning import clean_data
from feature_engineering import add_features
from data_preprocessing import TARGET, build_preprocessor

def compare_models():
    df = add_features(clean_data())
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, random_state=42
    )
    models = {
        "Ridge Regression": Ridge(alpha=10.0),
        "Decision Tree": DecisionTreeRegressor(
            min_samples_leaf=5, max_depth=24, random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=40, min_samples_leaf=3, max_depth=24, max_features=.8,
            random_state=42, n_jobs=-1
        ),
    }
    rows = []
    for name, estimator in models.items():
        pipe = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("regressor", estimator),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append({
            "model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "MSE": mean_squared_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R2": r2_score(y_test, pred),
        })
        print(f"{name}: MAE=${rows[-1]['MAE']:,.2f}, RMSE=${rows[-1]['RMSE']:,.2f}, R²={rows[-1]['R2']:.4f}")
    result = pd.DataFrame(rows).sort_values("MAE")
    result.to_csv("reports/model_comparison.csv", index=False)
    return result

if __name__ == "__main__":
    compare_models()
