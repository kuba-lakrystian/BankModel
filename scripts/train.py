import os.path
import pandas as pd

from src.data_loader.data_loader import DataLoader
from src.data_loader.data_preprocess import DataPreprocess
from src.data_loader.feature_selection import FeatureSelection
from src.data_utils.constants import *
from src.data_utils.helpers import Serialization
from src.ml_models.train_ml_model import TrainMLModel
from sklearn.model_selection import RandomizedSearchCV

pd.set_option("mode.chained_assignment", None)


def train():
    if os.path.isfile("data/df_target_final.pickle") and os.path.isfile(
        "data/df_final_final.pickle"
    ):
        data_target = Serialization.load_state("df_target_final", "data")
        data_training = Serialization.load_state("df_final_final", "data")
    else:
        dl = DataLoader()
        data_train = dl.train_test_load()
        dp = DataPreprocess()
        dp.load_data(data_train)
        data_preprocessed = dp.prepare_target()
        data_target, data_training = dp.extract_data_range()

    print('Data saved')
    fs = FeatureSelection()
    fs.load_data_to_selection(data_target, data_training)
    a = fs.convert_to_dummy()
    b = fs.salary_preprocessing()
    # Drop correlated variables
    merged = a.merge(b, how="inner", on=[NCODPERS]).merge(
        data_target, how="inner", on=[NCODPERS]
    )
    print("to save")
    Serialization.save_state(merged, "merged_data", "data")
    print('Saved')
    tmm = TrainMLModel()
    tmm.load_data_for_model(merged)
    a = tmm.fit(bayesian_optimisation=False, random_search=True)
    Serialization.save_state(tmm, "tmm_xgb_model", "data/trained_instances")


if __name__ == "__main__":
    train()
