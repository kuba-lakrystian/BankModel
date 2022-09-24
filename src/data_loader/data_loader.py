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

        self.data_train_raw[AGE] = pd.to_numeric(
            self.data_train_raw[AGE], errors="coerce"
        )
        self.data_test_raw[AGE] = pd.to_numeric(
            self.data_test_raw[AGE], errors="coerce"
        )
        self.data_train_raw[ANTIGUAEDAD] = pd.to_numeric(
            self.data_train_raw[ANTIGUAEDAD], errors="coerce"
        )
        self.data_test_raw[ANTIGUAEDAD] = pd.to_numeric(
            self.data_test_raw[ANTIGUAEDAD], errors="coerce"
        )
        self.data_train_raw[INDREL_1MES] = pd.to_numeric(
            self.data_train_raw[INDREL_1MES], errors="coerce"
        )
        self.data_test_raw[INDREL_1MES] = pd.to_numeric(
            self.data_test_raw[INDREL_1MES], errors="coerce"
        )
        constant_variables = self.data_train_raw.columns[
            self.data_train_raw.nunique() <= 1
        ]
        self.data_train_raw = self.data_train_raw.drop(columns=list(constant_variables))
        return self.data_train_raw
