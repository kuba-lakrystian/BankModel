import pandas as pd

from src.data_utils.constants import *


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
        self.data_train_raw = self._load_files(EMPTY_STR.join([DATA_PATH, "train.csv"]))
        self.data_test_raw = self._load_files(EMPTY_STR.join([DATA_PATH, "test.csv"]))
        return self.data_train_raw


data_test_raw = pd.read_csv(EMPTY_STR.join([DATA_PATH, "test.csv"]))