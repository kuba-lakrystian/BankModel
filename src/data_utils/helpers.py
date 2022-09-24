import dill
from sklearn import preprocessing
import numpy as np
from scipy.stats import chi2_contingency

from src.data_utils.constants import *


class Serialization:
    @staticmethod
    def save_state(element, file_name, file_path):
        with open(
            SLASH_STR.join([file_path, DOT_STR.join([file_name, "pickle"])]), "wb"
        ) as dill_file:
            dill.dump(element, dill_file)

    @staticmethod
    def load_state(file_name, file_path):
        with open(
            SLASH_STR.join([file_path, DOT_STR.join([file_name, "pickle"])]), "rb"
        ) as dill_file:
            element = dill.load(dill_file)
            return element


class CramerV:
    def __init__(self):
        self.df: pd.DataFrame = None

    def load_data(self, df):
        self.df = df

    @staticmethod
    def cramersV(var1, var2):
        crosstab = np.array(pd.crosstab(var1, var2, rownames=None, colnames=None))
        stat = chi2_contingency(crosstab)[0]
        obs = np.sum(crosstab)
        mini = min(crosstab.shape) - 1
        return stat / (obs * mini)

    def initialize(self):
        label = preprocessing.LabelEncoder()
        data_encoded = pd.DataFrame()

        for i in self.df.columns:
            data_encoded[i] = label.fit_transform(self.df[i])

        rows = []

        for var1 in data_encoded:
            col = []
            for var2 in data_encoded:
                cramers = self.cramersV(data_encoded[var1], data_encoded[var2])
                col.append(round(cramers, 2))
            rows.append(col)

        cramers_results = np.array(rows)
        df = pd.DataFrame(
            cramers_results, columns=data_encoded.columns, index=data_encoded.columns
        )
        return df
