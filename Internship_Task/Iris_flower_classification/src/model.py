
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def train_model(X_train, y_train, model_type="svm"):
    if model_type == "svm":
        model = SVC(kernel='linear')
    elif model_type == "logistic":
        model = LogisticRegression()
    elif model_type == "tree":
        model = DecisionTreeClassifier()
    else:
        raise ValueError("Invalid model type")

    model.fit(X_train, y_train)
    return model
