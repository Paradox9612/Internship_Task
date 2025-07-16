# main.py
from src.load_data import load_iris_dataset
from src.preprocess import preprocess_and_split
from src.model import train_model
from src.evaluate import evaluate_model
import pickle
import os

def main():
    # Step 1: Load Data
    X, y = load_iris_dataset()

    # Step 2: Preprocess
    X_train, X_test, y_train, y_test, le = preprocess_and_split(X, y)

    # Step 3: Train and Evaluate all models
    for model_type in ['logistic', 'tree', 'svm']:
        print(f"\n🔧 Training Model: {model_type.upper()}")
        model = train_model(X_train, y_train, model_type=model_type)
        acc = evaluate_model(model, X_test, y_test, le, model_type)

        # Save model
        os.makedirs("outputs", exist_ok=True)
        with open(f"outputs/{model_type}_model.pkl", "wb") as f:
            pickle.dump(model, f)

if __name__ == "__main__":
    main()
