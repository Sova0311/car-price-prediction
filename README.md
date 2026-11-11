# Predikcija cene polovnih automobila

Regresioni ML projekat koji procenjuje cenu polovnog automobila u USD na osnovu njegovih karakteristika. Ciljna promenljiva je `priceUSD`.

## Struktura projekta

```text
car-price-prediction/
├── data/
│   ├── cars.csv
│   └── cars_clean.csv              # nastaje pokretanjem čišćenja
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── model_comparison.py
├── models/
│   └── car_price_model.joblib
├── reports/
│   ├── evaluation.txt
│   ├── model_comparison.csv
│   └── test_predictions.csv
└── requirements.txt
```

## Instalacija

Preporučuje se Python 3.10 ili noviji.

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Pokretanje

Iz glavnog direktorijuma projekta:

```bash
python src/data_cleaning.py
python src/feature_engineering.py
python src/model_training.py
python src/model_evaluation.py
python src/model_comparison.py
```

Ako se skripte pokreću kao moduli, može i:

```bash
python -m src.model_training
```

## Obrada podataka i karakteristike

- Nazivi kolona `mileage(kilometers)` i `volume(cm3)` preimenovani su u `mileage_km` i `volume_cm3`.
- Uklanjaju se redovi bez ciljne cene, sa nepozitivnom cenom, nevalidnom godinom ili ekstremnom kilometražom.
- Nerealna zapremina motora tretira se kao nedostajuća vrednost.
- Nedostajuće numeričke vrednosti popunjavaju se medijanom, a kategorijske najčešćom vrednošću.
- Kategorijske promenljive kodiraju se pomoću `OneHotEncoder`; numeričke se skaliraju.
- Dodate su karakteristike `car_age`, `mileage_per_year`, `engine_volume_liters`, `brand_model`, `is_newer_car` i `is_high_mileage`.

Godina za računanje starosti vozila je 2026. (konstanta `REFERENCE_YEAR` u `src/feature_engineering.py`).

## Modeli i evaluacija

Skripta `model_comparison.py` poredi Random Forest, Extra Trees i Hist Gradient Boosting na istom trening/test skupu (80/20, `random_state=42`). Prikazuje MAE, RMSE i R². Skripta `model_training.py` trenira i čuva Random Forest pipeline.

- **MAE**: prosečna apsolutna greška u dolarima; najlakša je za praktično tumačenje.
- **RMSE**: više kažnjava velike greške.
- **R²**: deo varijanse ciljne promenljive koji model objašnjava.
- **MSE**: prosečna kvadratna greška.

Rezultati se generišu u direktorijumu `reports/`. Ne treba unapred pretpostaviti da je Random Forest najbolji: konačan izbor treba potvrditi stvarnim rezultatima poređenja.

## Primer upotrebe sačuvanog modela

```python
import joblib
import pandas as pd
from src.feature_engineering import add_features

model = joblib.load("models/car_price_model.joblib")
car = pd.DataFrame([{
    "make": "volkswagen",
    "model": "golf",
    "year": 2014,
    "condition": "with mileage",
    "mileage_km": 180000,
    "fuel_type": "diesel",
    "volume_cm3": 1600,
    "color": "black",
    "transmission": "mechanics",
    "drive_unit": "front-wheel drive",
    "segment": "C"
}])
prediction = model.predict(add_features(car))[0]
print(f"Procena cene: ${prediction:,.0f}")
```


## Rezultati na dostavljenom skupu

Test skup: 20% podataka, `random_state=42`. Rezultati:

| model            |     MAE |         MSE |    RMSE |       R2 |
|:-----------------|--------:|------------:|--------:|---------:|
| Random Forest    | 1158.66 | 8.83685e+06 | 2972.68 | 0.874863 |
| Decision Tree    | 1365.39 | 1.25765e+07 | 3546.33 | 0.821907 |
| Ridge Regression | 2492.37 | 2.66317e+07 | 5160.59 | 0.622873 |

Metrike sačuvanog Random Forest modela:

```text
MAE: 1,158.6626
MSE: 8,836,849.0076
RMSE: 2,972.6838
R2: 0.8749
```
## Napomena

Model daje statističku procenu na osnovu istorijskih oglasa iz dostavljenog skupa, a ne garantovanu tržišnu cenu. Rezultat zavisi od kvaliteta podataka i zastupljenosti sličnih vozila u skupu.
