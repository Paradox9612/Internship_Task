
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_model(model, X_test, y_test, label_encoder, model_name):
    y_pred = model.predict(X_test)
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    acc = accuracy_score(y_test_labels, y_pred_labels)
    print(f"\n🎯 Accuracy: {acc * 100:.2f}%\n")
    print("📊 Classification Report:")
    print(classification_report(y_test_labels, y_pred_labels))

    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=label_encoder.classes_)

    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_name} - Confusion Matrix")
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(f"outputs/{model_name}_confusion_matrix.png")
    plt.close()

    return acc
