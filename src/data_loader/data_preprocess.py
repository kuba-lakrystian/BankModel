import pandas as pd

from src.data_utils.constants import *


class DataPreprocess:

    def __init__(self):
        self.df: pd.DataFrame = None

    def load_data(self, data):
        self.df = data

    def choose_variables(self):
        chosen_data = self.df[['ID_code', 'var_0']]
        return chosen_data
