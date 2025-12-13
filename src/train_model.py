import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
print("script is started")

def main():
    BASE_DIR = Path(__file__).resolve().parents[1]
  
    DATA_PATH = BASE_DIR / "data" / "us_lng_export_prices.csv"

    df = pd.read_csv(DATA_PATH)
    df.columns = [c.lower() for c in df.columns]

    X = df[["year"]]
    y = df[df.columns[-1]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    print("✅ Model trained successfully")
    print(f"📉 Mean Absolute Error: {mae:.2f}")

if __name__ == "__main__":
    main()