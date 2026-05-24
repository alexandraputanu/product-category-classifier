"""Train and save a product category classifier.

Usage:
    python src/train_model.py

The script loads data/products.csv, cleans required columns, creates simple
text-based feature-engineering augmentation, compares two models, saves the
best model as models/product_classifier.pkl, and writes evaluation results to
reports/model_report.txt.
"""

from pathlib import Path
import re
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import SGDClassifier

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "products.csv"
MODEL_PATH = ROOT_DIR / "models" / "product_classifier.pkl"
REPORT_PATH = ROOT_DIR / "reports" / "model_report.txt"


def load_and_clean_data(path: Path) -> pd.DataFrame:
    """Load CSV, normalize column names and keep valid title/category rows."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    required_cols = ["Product Title", "Category Label"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.dropna(subset=required_cols).copy()
    df["Product Title"] = df["Product Title"].astype(str).str.lower().str.strip()
    df["Category Label"] = df["Category Label"].astype(str).str.strip()
    return df


def category_tokens(category: str) -> set[str]:
    """Extract words from category labels and add simple singular variants."""
    tokens = set(re.findall(r"[a-z]+", category.lower()))
    tokens.update({token[:-1] for token in tokens if token.endswith("s") and len(token) > 3})
    return tokens


def augment_titles(df: pd.DataFrame) -> pd.DataFrame:
    """Create extra training examples by removing category words from titles.

    Example: 'smeg sbs8004po fridge freezer' -> 'smeg sbs8004po'.
    This helps the model classify short product-code queries entered by users.
    """
    rows = []

    for _, row in df.iterrows():
        title = row["Product Title"]
        category = row["Category Label"]
        tokens = category_tokens(category)

        if not tokens:
            continue

        pattern = r"\b(" + "|".join(map(re.escape, sorted(tokens, key=len, reverse=True))) + r")\b"
        cleaned_title = re.sub(pattern, " ", title)
        cleaned_title = re.sub(r"\s+", " ", cleaned_title).strip()

        if cleaned_title and cleaned_title != title and len(cleaned_title) >= 5:
            rows.append({"Product Title": cleaned_title, "Category Label": category})

    if not rows:
        return df.copy()

    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


def build_models() -> dict[str, Pipeline]:
    """Return candidate text-classification models."""
    text_features = FeatureUnion([
        ("word_tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=12000, min_df=2)),
        ("char_tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=12000, min_df=2)),
    ])

    return {
        "Linear SVM": Pipeline([
            ("features", text_features),
            ("classifier", SGDClassifier(loss="hinge", alpha=1e-5, max_iter=20, tol=1e-3, random_state=42)),
        ]),
        "Multinomial Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=12000, min_df=2)),
            ("classifier", MultinomialNB()),
        ]),
    }


def main() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_and_clean_data(DATA_PATH)
    X = df["Product Title"]
    y = df["Category Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    train_df = pd.DataFrame({"Product Title": X_train, "Category Label": y_train})
    train_augmented = augment_titles(train_df)

    results = []
    trained_models = {}

    for model_name, model in build_models().items():
        model.fit(train_augmented["Product Title"], train_augmented["Category Label"])
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results.append((model_name, accuracy, predictions))
        trained_models[model_name] = model
        print(f"{model_name} accuracy: {accuracy:.4f}")

    best_model_name, best_accuracy, best_predictions = max(results, key=lambda item: item[1])

    # Retrain the selected model on all available clean data plus augmentation.
    full_augmented = augment_titles(df)
    final_model = build_models()[best_model_name]
    final_model.fit(full_augmented["Product Title"], full_augmented["Category Label"])
    joblib.dump(final_model, MODEL_PATH)

    manual_examples = [
        "iphone 7 32gb gold",
        "olympus e m10 mark iii geh use silber",
        "kenwood k20mss15 solo",
        "bosch wap28390gb 8kg 1400 spin",
        "bosch serie 4 kgv39vl31g",
        "smeg sbs8004po",
    ]

    report = [
        "Product Category Classification Report",
        "=" * 45,
        f"Rows used after cleaning: {len(df)}",
        f"Rows after title augmentation for final training: {len(full_augmented)}",
        f"Number of categories: {y.nunique()}",
        "",
        "Model comparison on held-out test set:",
    ]
    for model_name, accuracy, _ in results:
        report.append(f"- {model_name}: accuracy={accuracy:.4f}")

    report.extend([
        "",
        f"Best model: {best_model_name}",
        f"Best accuracy: {best_accuracy:.4f}",
        "",
        "Classification report for best model:",
        classification_report(y_test, best_predictions, zero_division=0),
        "",
        "Manual test examples using final saved model:",
    ])

    for title in manual_examples:
        report.append(f"- {title} -> {final_model.predict([title])[0]}")

    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    print(f"\nBest model: {best_model_name}")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
