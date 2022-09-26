import numpy as np
import pandas as pd
import xgboost as xgb
import multiprocessing
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
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

current_hyperparameters = {
    "objective": "binary:logistic",
    "base_score": 0.5,
    "booster": "gbtree",
    "colsample_bylevel": 1,
    "colsample_bynode": 1,
    "colsample_bytree": 1,
    "gamma": 0,
    "gpu_id": -1,
    "interaction_constraints": "",
    "learning_rate": 0.300000012,
    "max_delta_step": 0,
    "max_depth": 6,
    "min_child_weight": 1,
    "monotone_constraints": "()",
    "n_jobs": 8,
    "num_parallel_tree": 1,
    "predictor": "auto",
    "reg_alpha": 0,
    "reg_lambda": 1,
    "scale_pos_weight": 1,
    "subsample": 1,
    "tree_method": "exact",
    "validate_parameters": 1,
    "verbosity": None,
}

CUMULATED_IMPORTANCE_PERCENT = "cumulated_importance_percent"
IMPORTANCE = "importance"
IMPORTANCE_PERCENT = "importance_percent"
FEATURE = "feature"
HR = "HR"
LIFT = "Lift"
MEASURE = "Measure"
POPULATION = "Population"
VALUE = "Value"


class TrainMLModel:
    def __init__(self):
        self.xgb_model = None
        self.cut_off: float = None
        self.X_train: pd.DataFrame = None
        self.y_train: pd.DataFrame = None
        self.X_test: pd.DataFrame = None
        self.y_test: pd.DataFrame = None

    def fit(
        self,
        config,
        df,
        bayesian_optimisation: bool = False,
        random_search: bool = False,
        apply_smote: bool = False,
    ):
        valid_importance_percent = float(
            config[VALUES_SECTION][VALID_IMPORTANCE_PERCENT_VALUE]
        )
        set_seed = int(config[VALUES_SECTION][SET_SEED_VALUE])
        X_train, X_test, y_train, y_test = train_test_split(
            df.drop(columns=[TARGET, NCODPERS]),
            df[TARGET],
            test_size=0.3,
            train_size=0.7,
            random_state=set_seed,
            stratify=df[TARGET],
        )
        print(f"X_train size: {X_train.shape}")
        print(f"y_train distribution: {y_train.value_counts()}")
        if apply_smote:
            sm = SMOTE(random_state=set_seed)
            X_train_temp = X_train
            y_train_temp = y_train
            X_train, y_train = sm.fit_resample(X_train, y_train)
            print(f"X_train size after SMOTE: {X_train.shape}")
            print(f"y_train distribution after SMOTE: {y_train.value_counts()}")
        if bayesian_optimisation:
            best_hyperparameters = self._train_bayesian_optimisation(
                X_train, y_train, X_test, y_test, set_seed
            )
            self.xgb_model = xgb.XGBClassifier(
                random_state=set_seed,
                n_jobs=multiprocessing.cpu_count(),
                **best_hyperparameters,
            )
        elif random_search:
            best_hyperparameters = self._train_random_search(X_train, y_train, set_seed)
            self.xgb_model = xgb.XGBClassifier(
                random_state=set_seed,
                n_jobs=multiprocessing.cpu_count(),
                **best_hyperparameters,
            )
        else:
            self.xgb_model = xgb.XGBClassifier(
                random_state=set_seed, **current_hyperparameters
            )
        self.xgb_model = self.xgb_model.fit(X_train, y_train)
        xgb_fea_imp = pd.DataFrame(
            list(self.xgb_model.get_booster().get_fscore().items()),
            columns=[FEATURE, IMPORTANCE],
        ).sort_values(IMPORTANCE, ascending=False)
        xgb_fea_imp[IMPORTANCE_PERCENT] = xgb_fea_imp[IMPORTANCE] / sum(
            xgb_fea_imp[IMPORTANCE]
        )
        xgb_fea_imp[CUMULATED_IMPORTANCE_PERCENT] = xgb_fea_imp[
            IMPORTANCE_PERCENT
        ].cumsum()
        chosen_variables = list(
            xgb_fea_imp[
                xgb_fea_imp[CUMULATED_IMPORTANCE_PERCENT] <= valid_importance_percent
            ][FEATURE]
        ) + [TARGET, NCODPERS]
        if apply_smote:
            X_train = X_train_temp
            y_train = y_train_temp
        X_train[PREDICT_PROBA] = self.xgb_model.predict_proba(X_train)[:, 1]
        X_train[TARGET] = y_train
        print("Results for train")
        self._calculate_hit_rate_and_lift(X_train)
        self._calculate_predictive_power(X_train)
        X_test[PREDICT_PROBA] = self.xgb_model.predict_proba(X_test)[:, 1]
        X_test[TARGET] = y_test
        print("Results for test")
        self._calculate_hit_rate_and_lift(X_test)
        self._calculate_predictive_power(X_test)
        return chosen_variables

    def predict(self, df):
        print("Results for out-of-sample")
        X_train = df.drop(columns=[TARGET, NCODPERS])
        df[PREDICT_PROBA] = self.xgb_model.predict_proba(X_train)[:, 1]
        self._calculate_hit_rate_and_lift(df)
        self._calculate_predictive_power(df)

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
            eval_metric="aucpr",
            early_stopping_rounds=10,
            verbose=False,
        )

        pred = clf.predict(self.X_test)
        precision, recall, thresholds = precision_recall_curve(
            self.y_test, pred
        )
        pr_auc = auc(recall, precision)
        print("SCORE:", pr_auc)
        return {"loss": pr_auc, "status": STATUS_OK}

    def _train_bayesian_optimisation(self, X_train, y_train, X_test, y_test, set_seed):
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
            rstate=np.random.default_rng(set_seed),
        )
        print(best_hyperparams)
        self._release_memory()
        return best_hyperparams

    def _train_random_search(self, X, y, set_seed):
        self.X_train = X
        self.y_train = y

        classifier = xgb.XGBClassifier(random_state=set_seed)
        rs_model = RandomizedSearchCV(
            classifier,
            param_distributions=params,
            n_iter=5,
            scoring="roc_auc",
            n_jobs=-1,
            cv=5,
            verbose=3,
            random_state=set_seed,
        )
        rs_model.fit(self.X_train, self.y_train)
        print(rs_model.best_params_)
        self._release_memory()
        return rs_model.best_params_

    def _calculate_predictive_power(self, df):
        df[PREDICT] = np.where(df[PREDICT_PROBA] >= self.cut_off, 1, 0)
        results = pd.DataFrame(columns=[MEASURE, VALUE])
        results.loc[0, MEASURE] = "ROC AUC"
        results.loc[0, VALUE] = roc_auc_score(df[TARGET], df[PREDICT_PROBA])
        results.loc[1, MEASURE] = "Gini"
        results.loc[1, VALUE] = 2 * results.loc[0, VALUE] - 1
        precision, recall, thresholds = precision_recall_curve(
            df[TARGET], df[PREDICT_PROBA]
        )
        results.loc[2, MEASURE] = "Precision-Recall AUC"
        results.loc[2, VALUE] = auc(recall, precision)
        results.loc[3, MEASURE] = "Accuracy"
        results.loc[3, VALUE] = accuracy_score(df[TARGET], df[PREDICT])
        results.loc[4, MEASURE] = "F1 Score"
        results.loc[4, VALUE] = f1_score(df[TARGET], df[PREDICT])
        results.loc[5, MEASURE] = "Balanced accuracy"
        results.loc[5, VALUE] = balanced_accuracy_score(df[TARGET], df[PREDICT])
        results.loc[6, MEASURE] = "Precision"
        results.loc[6, VALUE] = precision_score(df[TARGET], df[PREDICT])
        results.loc[7, MEASURE] = "Recall"
        results.loc[7, VALUE] = recall_score(df[TARGET], df[PREDICT])
        print(results)
        print(contingency_matrix(df[TARGET], df[PREDICT]))
        return results

    def _calculate_hit_rate_and_lift(self, df):
        results = pd.DataFrame(columns=[MEASURE, POPULATION, HR, LIFT])
        results.loc[0, MEASURE] = "Hit rate entire population"
        results.loc[0, POPULATION] = len(df)
        results.loc[0, HR] = df[TARGET].mean()
        df_temp = df.sort_values(PREDICT_PROBA, ascending=False)
        results.loc[1, MEASURE] = "Hit rate top 2.5%"
        df_temp_25_perc = df_temp[0 : int(0.025 * len(df_temp))]
        results.loc[1, POPULATION] = int(0.025 * len(df_temp))
        results.loc[1, HR] = df_temp_25_perc[TARGET].mean()
        results.loc[1, LIFT] = results.loc[1, HR] / results.loc[0, HR]
        results.loc[2, MEASURE] = "Hit rate top 5%"
        df_temp_5_perc = df_temp[0 : int(0.05 * len(df_temp))]
        results.loc[2, POPULATION] = int(0.05 * len(df_temp))
        results.loc[2, HR] = df_temp_5_perc[TARGET].mean()
        results.loc[2, LIFT] = results.loc[2, HR] / results.loc[0, HR]
        results.loc[3, MEASURE] = "Hit rate top 10%"
        df_temp_10_perc = df_temp[0 : int(0.1 * len(df_temp))]
        results.loc[3, POPULATION] = int(0.1 * len(df_temp))
        results.loc[3, HR] = df_temp_10_perc[TARGET].mean()
        results.loc[3, LIFT] = results.loc[3, HR] / results.loc[0, HR]
        print(results)
        if self.cut_off is None:
            self.cut_off = df_temp_5_perc[PREDICT_PROBA].min()

    def _release_memory(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
