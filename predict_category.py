"""Interactive product category prediction script.

Usage:
    python src/predict_category.py

Type a product title and the trained model will return the predicted category.
Type 'quit' or 'exit' to stop the script.
"""

from pathlib import Path
import joblib

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "product_classifier.pkl"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python src/train_model.py` first."
        )

    model = joblib.load(MODEL_PATH)

    print("Product Category Predictor")
    print("Type a product title. Type 'quit' or 'exit' to stop.")

    while True:
        title = input("\nEnter product title: ").strip()

        if title.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        if not title:
            print("Please enter a non-empty product title.")
            continue

        prediction = model.predict([title])[0]
        print(f"Predicted category: {prediction}")


if __name__ == "__main__":
    main()
