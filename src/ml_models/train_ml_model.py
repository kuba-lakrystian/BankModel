import numpy as np
import pandas as pd
import xgboost as xgb
import multiprocessing
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score, precision_recall_curve, auc
from sklearn.metrics.cluster import contingency_matrix

from src.data_utils.constants import *

space={'max_depth': hp.choice('max_depth', np.arange(1, 14, dtype=int)),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': 180,
        'seed': 0
    }

params = {
    'learning_rate': [0.05,0.10,0.15,0.20,0.25,0.30],
    'max_depth': [ 3, 4, 5, 6, 8, 10, 12, 15],
    'min_child_weight': [ 1, 3, 5, 7 ],
    'gamma': [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
    'colsample_bytree': [ 0.3, 0.4, 0.5 , 0.7 ]
}


class TrainMLModel:
    def __init__(self):
        self.df: pd.DataFrame = None
        self.xgb_model = None
        self.X_train: pd.DataFrame = None
        self.y_train: pd.DataFrame = None

    def load_data_for_model(self, data):
        self.df = data

    def fit(self, bayesian_optimisation: bool = False, random_search: bool = False):
        X_train = self.df.drop(columns=["target", FECHA_DATO, NCODPERS])
        y_train = self.df["target"]
        if bayesian_optimisation:
            best_hyperparameters = self._train_bayesian_optimisation(X_train, y_train)
            self.xgb_model = xgb.XGBClassifier(n_jobs=multiprocessing.cpu_count(), **best_hyperparameters)
        elif random_search:
            best_hyperparameters = self._train_random_search(X_train, y_train)
            self.xgb_model = xgb.XGBClassifier(n_jobs=multiprocessing.cpu_count(), **best_hyperparameters)
        else:
            self.xgb_model = xgb.XGBClassifier(n_jobs=multiprocessing.cpu_count())
        self.xgb_model.fit(X_train, y_train)
        self.df['predict_proba'] = self.predict(X_train)
        results, cut_off = self.calculate_hit_rate_and_lift()
        self.calculate_predictive_power(cut_off)
        return self.df

    def predict(self, X):
        return self.xgb_model.predict_proba(X)[:, 1]

    def objective(self, space):
        clf = xgb.XGBClassifier(
            n_estimators=space['n_estimators'], max_depth=int(space['max_depth']), gamma=space['gamma'],
            reg_alpha=int(space['reg_alpha']), min_child_weight=int(space['min_child_weight']),
            colsample_bytree=int(space['colsample_bytree']))

        evaluation = [(self.X_train, self.y_train), (self.X_train, self.y_train)]

        clf.fit(self.X_train, self.y_train,
                eval_set=evaluation, eval_metric="auc",
                early_stopping_rounds=10, verbose=False)

        pred = clf.predict(self.X_train)
        accuracy = accuracy_score(self.y_train, pred > 0.01)
        print("SCORE:", accuracy)
        return {'loss': -accuracy, 'status': STATUS_OK}

    def _train_bayesian_optimisation(self, X, y):
        self.X_train = X
        self.y_train = y
        trials = Trials()

        best_hyperparams = fmin(fn=self.objective,
                                space=space,
                                algo=tpe.suggest,
                                max_evals=100,
                                trials=trials)
        print(best_hyperparams)
        return best_hyperparams

    def _train_random_search(self, X, y):
        self.X_train = X
        self.y_train = y

        classifier = xgb.XGBClassifier()
        rs_model = RandomizedSearchCV(classifier, param_distributions=params, n_iter=5, scoring='roc_auc', n_jobs=-1,
                                      cv=5, verbose=3)
        rs_model.fit(self.X_train, self.y_train)
        print(rs_model.best_params_)
        return rs_model.best_params_

    def calculate_predictive_power(self, cut_off: float = 0.5):
        self.df['predict'] = np.where(self.df['predict_proba'] >= cut_off, 1, 0)
        results = pd.DataFrame(columns=['Measure', 'Value'])
        results.loc[0, 'Measure'] = 'ROC AUC'
        results.loc[0, 'Value'] = roc_auc_score(self.df[TARGET], self.df['predict_proba'])
        results.loc[1, 'Measure'] = 'Gini'
        results.loc[1, 'Value'] = 2*results.loc[0, 'Value'] - 1
        precision, recall, thresholds = precision_recall_curve(self.df[TARGET], self.df['predict_proba'])
        results.loc[2, 'Measure'] = 'Precision-Recall AUC'
        results.loc[2, 'Value'] = auc(recall, precision)
        results.loc[3, 'Measure'] = 'Accuracy'
        results.loc[3, 'Value'] = accuracy_score(self.df[TARGET], self.df['predict'])
        results.loc[4, 'Measure'] = 'F1 Score'
        results.loc[4, 'Value'] = f1_score(self.df[TARGET], self.df['predict'])
        results.loc[5, 'Measure'] = 'Balanced accuracy'
        results.loc[5, 'Value'] = balanced_accuracy_score(self.df[TARGET], self.df['predict'])
        print(results)
        print(contingency_matrix(self.df[TARGET], self.df['predict']))
        return results

    def calculate_hit_rate_and_lift(self):
        results = pd.DataFrame(columns = ['Measure', 'Population', 'HR', 'Lift'])
        results.loc[0, 'Measure'] = 'Hit rate entire population'
        results.loc[0, 'Population'] = len(self.df)
        results.loc[0, 'HR'] = self.df[TARGET].mean()
        df_temp = self.df.sort_values('predict_proba', ascending=False)
        results.loc[1, 'Measure'] = 'Hit rate top 2.5%'
        df_temp_25_perc = df_temp[0:int(0.025*len(df_temp))]
        results.loc[1, 'Population'] = int(0.025*len(df_temp))
        results.loc[1, 'HR'] = df_temp_25_perc[TARGET].mean()
        results.loc[1, 'Lift'] = results.loc[1, 'HR']/results.loc[0, 'HR']
        results.loc[2, 'Measure'] = 'Hit rate top 5%'
        df_temp_5_perc = df_temp[0:int(0.05*len(df_temp))]
        results.loc[2, 'Population'] = int(0.05 * len(df_temp))
        results.loc[2, 'HR'] = df_temp_5_perc[TARGET].mean()
        results.loc[2, 'Lift'] = results.loc[2, 'HR'] / results.loc[0, 'HR']
        results.loc[3, 'Measure'] = 'Hit rate top 10%'
        df_temp_10_perc = df_temp[0:int(0.1 * len(df_temp))]
        results.loc[3, 'Population'] = int(0.1 * len(df_temp))
        results.loc[3, 'HR'] = df_temp_10_perc[TARGET].mean()
        results.loc[3, 'Lift'] = results.loc[3, 'HR'] / results.loc[0, 'HR']
        print(results)
        return results, df_temp_25_perc['predict_proba'].min()
