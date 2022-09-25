import pandas as pd

from src.data_utils.constants import *

pd.set_option("mode.chained_assignment", None)


class DataLoader:
    def __init__(self):
        self.data_train_raw = None

    @staticmethod
    def _load_files(data_path):
        data_loaded = pd.read_csv(data_path)
        return data_loaded

    def train_load(self, config):
        data_path = config[INPUT_SECTION][DATA_PATH]
        raw_data_path = config[INPUT_SECTION][RAW_DATA_FILE]
        self.data_train_raw = self._load_files(
            EMPTY_STR.join([data_path, raw_data_path])
        )

        for i in [AGE, ANTIGUAEDAD, INDREL_1MES]:
            self.data_train_raw[i] = pd.to_numeric(
                self.data_train_raw[i], errors=COERCE
            )

        constant_variables = self.data_train_raw.columns[
            self.data_train_raw.nunique() <= 1
        ]
        self.data_train_raw = self.data_train_raw.drop(columns=list(constant_variables))

        return self.data_train_raw
