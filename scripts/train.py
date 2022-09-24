import os.path
import pandas as pd

from src.data_loader.data_loader import DataLoader
from src.data_loader.data_preprocess import DataPreprocess
from src.data_loader.feature_selection import FeatureSelection
from src.data_utils.constants import *
from src.ml_models.explainerdashboard import ExplainerDashboardCustom
from src.data_utils.helpers import Serialization
from src.ml_models.train_ml_model import TrainMLModel
from sklearn.model_selection import RandomizedSearchCV

pd.set_option("mode.chained_assignment", None)


def train():
    if os.path.isfile("data/df_target_final_train.pickle") and os.path.isfile("data/df_final_final_train.pickle") and os.path.isfile("data/df_target_final_test.pickle") and os.path.isfile("data/df_final_final_test.pickle"):
        data_target_train = Serialization.load_state("df_target_final_train", "data")
        data_training_train = Serialization.load_state("df_final_final_train", "data")
        data_target_test = Serialization.load_state("df_target_final_test", "data")
        data_training_test = Serialization.load_state("df_final_final_test", "data")
    else:
        dl = DataLoader()
        data_train = dl.train_test_load()
        dp = DataPreprocess()
        dp.load_data(data_train)
        data_preprocessed = dp.prepare_target()
        data_target_train, data_training_train = dp.extract_data_range("2015-01-01", "2015-06-30", "2015-07-28", 'df_final_final_train', 'df_target_final_train')
        data_target_test, data_training_test = dp.extract_data_range("2015-08-01", "2016-01-30", "2016-02-28",
                                                                       'df_final_final_test', 'df_target_final_test')
        dp.release_memory()

    print('Data saved')
    fs = FeatureSelection()
    fs.load_data_to_selection(data_target_train, data_training_train)
    a = fs.convert_to_dummy()
    b = fs.salary_preprocessing()
    fs.release_memory()
    fs.load_data_to_selection(data_target_test, data_training_test)
    a2 = fs.convert_to_dummy()
    b2 = fs.salary_preprocessing()
    # Drop correlated variables
    merged_train = a.merge(b, how="inner", on=[NCODPERS]).merge(
        data_target_train, how="inner", on=[NCODPERS]
    )

    tmm = TrainMLModel()
    tmm.load_data_for_model(merged_train)
    a = tmm.fit(bayesian_optimisation=False, random_search=False)
    Serialization.save_state(tmm, "tmm_xgb_model", "data/trained_instances")
    if not os.path.isfile("dashboard.yaml") and os.path.isfile("explainer.joblib"):
        print("ExplainerDashboard calculated")
        edc = ExplainerDashboardCustom()
        edc.load_objects(merged_train, tmm.xgb_model)
        edc.train_explainer_dashboard()


if __name__ == "__main__":
    train()
