import numpy as np
import pandas as pd

from src.data_utils.constants import *
from src.data_utils.helpers import Serialization

pd.set_option("mode.chained_assignment", None)

AFTER = "after"
BEFORE = "before"
MO = "mo"
ONE = "one"
TWO = "two"
THREE = "three"


class DataPreprocess:
    def __init__(self):
        self.df: pd.DataFrame = None
        self.df_target: pd.DataFrame = None
        self.constant_variables = None
        self.config = None

    def load_data(self, config, data):
        self.df = data
        self.config = config

    def prepare_target(self):
        df = self.df[[FECHA_DATO, NCODPERS, IND_TJCR_FIN_ULT1]]
        df[
            UNDERSCORE_STR.join([IND_TJCR_FIN_ULT1, ONE, MO, BEFORE])
        ] = self.custom_shift(df, 1)
        df[
            UNDERSCORE_STR.join([IND_TJCR_FIN_ULT1, TWO, MO, BEFORE])
        ] = self.custom_shift(df, 2)
        df[
            UNDERSCORE_STR.join([IND_TJCR_FIN_ULT1, THREE, MO, BEFORE])
        ] = self.custom_shift(df, 3)
        df[
            UNDERSCORE_STR.join([IND_TJCR_FIN_ULT1, ONE, MO, AFTER])
        ] = self.custom_shift(df, -1)

        df[TARGET] = np.where(
            (df[UNDERSCORE_STR.join([IND_TJCR_FIN_ULT1, ONE, MO, BEFORE])] == 0)
            & (df[UNDERSCORE_STR.join([IND_TJCR_FIN_ULT1, TWO, MO, BEFORE])] == 0)
            & (df[UNDERSCORE_STR.join([IND_TJCR_FIN_ULT1, THREE, MO, BEFORE])] == 0)
            & (df[IND_TJCR_FIN_ULT1] == 1)
            & (df[UNDERSCORE_STR.join([IND_TJCR_FIN_ULT1, ONE, MO, AFTER])] == 1),
            1,
            0,
        )
        self.df_target = df[[FECHA_DATO, NCODPERS, TARGET]]

    @staticmethod
    def custom_shift(data, months):
        return data.groupby(NCODPERS)[IND_TJCR_FIN_ULT1].shift(months)

    def extract_data_range(self, dates, file_data, file_labels):
        data_path = self.config[INPUT_SECTION][DATA_PATH]
        date_start = dates[0]
        date_end = dates[1]
        date_target = dates[2]
        file_name_X = file_data
        file_name_y = file_labels
        df_target_final = self.df_target[self.df_target[FECHA_DATO] == date_target]
        df_final = self.df
        df_final[DATE] = pd.to_datetime(df_final[FECHA_DATO])
        mask = (df_final[DATE] > date_start) & (df_final[DATE] <= date_end)
        df_final_final = self.df[mask]
        df_final_final = df_final_final[
            df_final_final[NCODPERS].isin(df_target_final[NCODPERS].unique())
        ]
        months_with_engagements = df_final_final[NCODPERS].value_counts().reset_index()
        only_full_customers = list(
            months_with_engagements[months_with_engagements.ncodpers == 6][INDEX]
        )
        df_final_final = df_final_final[
            df_final_final[NCODPERS].isin(only_full_customers)
        ]
        df_target_final = df_target_final[
            df_target_final[NCODPERS].isin(df_final_final[NCODPERS].unique())
        ]
        if self.constant_variables is None:
            self.constant_variables = df_final_final.columns[
                df_final_final.nunique() <= 1
            ]
        df_final_final = df_final_final.drop(columns=list(self.constant_variables))
        Serialization.save_state(df_target_final, file_name_y, data_path)
        Serialization.save_state(df_final_final, file_name_X, data_path)
        return df_target_final, df_final_final

    def release_memory(self):
        self.df = None
        self.df_target = None
