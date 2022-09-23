import numpy as np
import pandas as pd
import xgboost as xgb
import multiprocessing

from src.data_utils.constants import *


class TrainMLModel:
    def __init__(self):
        self.df: pd.DataFrame = None

    def load_data_for_model(self, data):
        self.df = data

    def train_xgb_model(self):
        X_train = self.df.drop(columns=["target", FECHA_DATO])
        y_train = self.df["target"]
        xgb_model = xgb.XGBClassifier(n_jobs=multiprocessing.cpu_count())
        xgb_model.fit(X_train, y_train)
        self.df["predict_proba"] = xgb_model.predict_proba(X_train)[:, 1]
        return self.df
