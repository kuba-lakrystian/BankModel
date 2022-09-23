import pandas as pd

from src.data_utils.constants import *

pd.set_option("mode.chained_assignment", None)


class DataLoader:
    def __init__(self):
        self.data_train_raw = None
        self.data_test_raw = None
        self.data_path = DATA_PATH

    @staticmethod
    def _load_files(data_path):
        data_loaded = pd.read_csv(data_path)
        return data_loaded

    def train_test_load(self):
        self.data_train_raw = self._load_files(
            EMPTY_STR.join([DATA_PATH, "/data_recommendation_engine/train_ver2.csv"])
        )
        self.data_test_raw = self._load_files(
            EMPTY_STR.join([DATA_PATH, "/data_recommendation_engine/test_ver2.csv"])
        )
        return self.data_train_raw
