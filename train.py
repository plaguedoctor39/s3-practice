from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from nyoka import xgboost_to_pmml
import pandas as pd

import joblib
import os


def train(data_conf, model_conf, **kwargs):
    print(model_conf['hyperParameters'])
    hyperparams = model_conf["hyperParameters"]
    train_df = pd.read_csv(data_conf)

    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    target_name = 'Outcome'

    # split data into X and y
    X_train = train_df.drop(target_name, 1)
    y_train = train_df[target_name]

    print("Starting training...")

    # fit model to training data
    model = Pipeline([('scaler', MinMaxScaler()),
                     ('xgb', XGBClassifier(eta=float(hyperparams["eta"]),
                                           max_depth=int(hyperparams["max_depth"])))])
    # xgboost saves feature names but lets store on pipeline for easy access later
    model.feature_names = feature_names
    model.target_name = target_name

    model.fit(X_train, y_train)

    print("Finished training")

    # export model artefacts
    joblib.dump(model, "artifacts/output/model.joblib")
    xgboost_to_pmml(pipeline=model, col_names=feature_names, target_name=target_name, pmml_f_name="artifacts/output/model.pmml")

    print("Saved trained model")