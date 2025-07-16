from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_and_split(X, y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, le
