import numpy as np
import pandas as pd
import xgboost as xgb
import multiprocessing
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    precision_recall_curve,
    auc,
    recall_score,
    precision_score,
)
from sklearn.metrics.cluster import contingency_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from src.data_utils.constants import *

space = {
    "max_depth": hp.choice("max_depth", np.arange(1, 14, dtype=int)),
    "gamma": hp.uniform("gamma", 1, 9),
    "reg_alpha": hp.quniform("reg_alpha", 40, 180, 1),
    "reg_lambda": hp.uniform("reg_lambda", 0, 1),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
    "min_child_weight": hp.quniform("min_child_weight", 0, 10, 1),
    "n_estimators": 180,
    "seed": 0,
}

params = {
    "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
    "colsample_bytree": [0.3, 0.4, 0.5, 0.7],
}


class TrainMLModel:
    def __init__(self):
        self.df: pd.DataFrame = None
        self.xgb_model = None
        self.cut_off: float = None
        self.X_train: pd.DataFrame = None
        self.y_train: pd.DataFrame = None
        self.X_test: pd.DataFrame = None
        self.y_test: pd.DataFrame = None

    def load_data_for_model(self, data):
        self.df = data

    def fit(
        self,
        bayesian_optimisation: bool = False,
        random_search: bool = False,
        apply_smote: bool = False,
    ):
        X_train, X_test, y_train, y_test = train_test_split(
            self.df.drop(columns=["target", NCODPERS]),
            self.df["target"],
            test_size=0.3,
            train_size=0.7,
            random_state=42,
            stratify=self.df["target"],
        )
        print(f"X_train size: {X_train.shape}")
        print(f"y_train distribution: {y_train.value_counts()}")
        if apply_smote:
            sm = SMOTE(random_state=42)
            X_train, y_train = sm.fit_resample(X_train, y_train)
            print(f"X_train size after SMOTE: {X_train.shape}")
            print(f"y_train distribution after SMOTE: {y_train.value_counts()}")
        if bayesian_optimisation:
            best_hyperparameters = self._train_bayesian_optimisation(
                X_train, y_train, X_test, y_test
            )
            self.xgb_model = xgb.XGBClassifier(
                random_state=42,
                n_jobs=multiprocessing.cpu_count(),
                **best_hyperparameters,
            )
        elif random_search:
            best_hyperparameters = self._train_random_search(X_train, y_train)
            self.xgb_model = xgb.XGBClassifier(
                random_state=42,
                n_jobs=multiprocessing.cpu_count(),
                **best_hyperparameters,
            )
        else:
            self.xgb_model = xgb.XGBClassifier(
                random_state=42, n_jobs=multiprocessing.cpu_count()
            )
        self.xgb_model = self.xgb_model.fit(X_train, y_train)
        xgb_fea_imp = pd.DataFrame(list(self.xgb_model.get_booster().get_fscore().items()),
                                   columns=['feature', 'importance']).sort_values('importance', ascending=False)
        xgb_fea_imp['importance_percent'] = xgb_fea_imp['importance'] / sum(xgb_fea_imp['importance'])
        xgb_fea_imp['_cumulated_importance_percent'] = xgb_fea_imp['importance_percent'].cumsum()
        xgb_fea_imp['_cumulated_importance_percent'] <= 0.95
        chosen_variables = list(xgb_fea_imp[xgb_fea_imp['_cumulated_importance_percent'] <= 0.95]['feature']) + ["target", NCODPERS]
        X_train["predict_proba"] = self.xgb_model.predict_proba(X_train)[:, 1]
        X_train[TARGET] = y_train
        print("results for train")
        results_train = self.calculate_hit_rate_and_lift(X_train)
        self.calculate_predictive_power(X_train)
        X_test["predict_proba"] = self.xgb_model.predict_proba(X_test)[:, 1]
        X_test[TARGET] = y_test
        print("results for test")
        results_test = self.calculate_hit_rate_and_lift(X_test)
        self.calculate_predictive_power(X_test)
        return self.df, chosen_variables

    def predict(self):
        print("results for out-of-sample")
        X_train = self.df.drop(columns=["target", NCODPERS])
        y_train = self.df["target"]
        self.df["predict_proba"] = self.xgb_model.predict_proba(X_train)[:, 1]
        results = self.calculate_hit_rate_and_lift(self.df)
        self.calculate_predictive_power(self.df)

    def objective(self, space):
        clf = xgb.XGBClassifier(
            n_estimators=space["n_estimators"],
            max_depth=int(space["max_depth"]),
            gamma=space["gamma"],
            reg_alpha=int(space["reg_alpha"]),
            min_child_weight=int(space["min_child_weight"]),
            colsample_bytree=int(space["colsample_bytree"]),
        )

        evaluation = [(self.X_train, self.y_train), (self.X_test, self.y_test)]

        clf.fit(
            self.X_train,
            self.y_train,
            eval_set=evaluation,
            eval_metric="auc",
            early_stopping_rounds=10,
            verbose=False,
        )

        pred = clf.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, pred > 0.01)
        print("SCORE:", accuracy)
        return {"loss": -accuracy, "status": STATUS_OK}

    def _train_bayesian_optimisation(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        trials = Trials()

        best_hyperparams = fmin(
            fn=self.objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials,
            rstate=np.random.default_rng(42),
        )
        print(best_hyperparams)
        return best_hyperparams

    def _train_random_search(self, X, y):
        self.X_train = X
        self.y_train = y

        classifier = xgb.XGBClassifier(random_state=42)
        rs_model = RandomizedSearchCV(
            classifier,
            param_distributions=params,
            n_iter=5,
            scoring="roc_auc",
            n_jobs=-1,
            cv=5,
            verbose=3,
            random_state=42,
        )
        rs_model.fit(self.X_train, self.y_train)
        print(rs_model.best_params_)
        return rs_model.best_params_

    def calculate_predictive_power(self, df):
        df["predict"] = np.where(df["predict_proba"] >= self.cut_off, 1, 0)
        results = pd.DataFrame(columns=["Measure", "Value"])
        results.loc[0, "Measure"] = "ROC AUC"
        results.loc[0, "Value"] = roc_auc_score(df[TARGET], df["predict_proba"])
        results.loc[1, "Measure"] = "Gini"
        results.loc[1, "Value"] = 2 * results.loc[0, "Value"] - 1
        precision, recall, thresholds = precision_recall_curve(
            df[TARGET], df["predict_proba"]
        )
        results.loc[2, "Measure"] = "Precision-Recall AUC"
        results.loc[2, "Value"] = auc(recall, precision)
        results.loc[3, "Measure"] = "Accuracy"
        results.loc[3, "Value"] = accuracy_score(df[TARGET], df["predict"])
        results.loc[4, "Measure"] = "F1 Score"
        results.loc[4, "Value"] = f1_score(df[TARGET], df["predict"])
        results.loc[5, "Measure"] = "Balanced accuracy"
        results.loc[5, "Value"] = balanced_accuracy_score(df[TARGET], df["predict"])
        results.loc[6, "Measure"] = "Precision"
        results.loc[6, "Value"] = precision_score(df[TARGET], df["predict"])
        results.loc[7, "Measure"] = "Recall"
        results.loc[7, "Value"] = recall_score(df[TARGET], df["predict"])
        print(results)
        print(contingency_matrix(df[TARGET], df["predict"]))
        return results

    def calculate_hit_rate_and_lift(self, df):
        results = pd.DataFrame(columns=["Measure", "Population", "HR", "Lift"])
        results.loc[0, "Measure"] = "Hit rate entire population"
        results.loc[0, "Population"] = len(df)
        results.loc[0, "HR"] = df[TARGET].mean()
        df_temp = df.sort_values("predict_proba", ascending=False)
        results.loc[1, "Measure"] = "Hit rate top 2.5%"
        df_temp_25_perc = df_temp[0 : int(0.025 * len(df_temp))]
        results.loc[1, "Population"] = int(0.025 * len(df_temp))
        results.loc[1, "HR"] = df_temp_25_perc[TARGET].mean()
        results.loc[1, "Lift"] = results.loc[1, "HR"] / results.loc[0, "HR"]
        results.loc[2, "Measure"] = "Hit rate top 5%"
        df_temp_5_perc = df_temp[0 : int(0.05 * len(df_temp))]
        results.loc[2, "Population"] = int(0.05 * len(df_temp))
        results.loc[2, "HR"] = df_temp_5_perc[TARGET].mean()
        results.loc[2, "Lift"] = results.loc[2, "HR"] / results.loc[0, "HR"]
        results.loc[3, "Measure"] = "Hit rate top 10%"
        df_temp_10_perc = df_temp[0 : int(0.1 * len(df_temp))]
        results.loc[3, "Population"] = int(0.1 * len(df_temp))
        results.loc[3, "HR"] = df_temp_10_perc[TARGET].mean()
        results.loc[3, "Lift"] = results.loc[3, "HR"] / results.loc[0, "HR"]
        print(results)
        if self.cut_off is None:
            self.cut_off = df_temp_5_perc["predict_proba"].min()
        return results

    def release_memory(self):
        self.df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
