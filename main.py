import os
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from joblib import dump
import pandas as pd


def get_data(base_dir):
    data_file_names = [x for x in os.listdir(base_dir) if x.endswith('.csv')]
    data = {}
    for name in data_file_names:
        path_file = os.path.join(base_dir, name)
        data[name] = pd.read_csv("Iris.csv")
    return data


def split_data(data, test_size=0.2, random_state=42):
    df = data['iris.csv']

    # Drop the 'Id' column
    df = df.drop('Id', axis=1, errors='ignore')

    X = df.drop('Species', axis=1)
    y = df['Species']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}

def train_model(X_train, y_train):
    log_reg = LogisticRegression()
    model = log_reg.fit(X_train, y_train)
    return model


def save_model(model):
    dump(model, "iris_prediction.joblib")
    print("Model saved")
