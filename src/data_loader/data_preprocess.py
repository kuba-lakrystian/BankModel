import numpy as np
import pandas as pd

from src.data_utils.constants import *
from src.data_utils.helpers import Serialization

pd.set_option("mode.chained_assignment", None)


class DataPreprocess:
    def __init__(self):
        self.df: pd.DataFrame = None
        self.df_target: pd.DataFrame = None

    def load_data(self, data):
        self.df = data

    def prepare_target(self):
        df = self.df[[FECHA_DATO, NCODPERS, IND_TJCR_FIN_ULT1]]
        df["ind_tjcr_fin_ult1_1_mo_before"] = self.custom_shift(df, 1)
        df["ind_tjcr_fin_ult1_2_mo_before"] = self.custom_shift(df, 2)
        df["ind_tjcr_fin_ult1_3_mo_before"] = self.custom_shift(df, 3)
        df["ind_tjcr_fin_ult1_1_mo_after"] = self.custom_shift(df, -1)

        df[TARGET] = np.where(
            (df["ind_tjcr_fin_ult1_1_mo_before"] == 0)
            & (df["ind_tjcr_fin_ult1_2_mo_before"] == 0)
            & (df["ind_tjcr_fin_ult1_3_mo_before"] == 0)
            & (df["ind_tjcr_fin_ult1"] == 1)
            & (df["ind_tjcr_fin_ult1_1_mo_after"] == 1),
            1,
            0,
        )
        self.df_target = df[[FECHA_DATO, NCODPERS, TARGET]]
        return self.df_target

    @staticmethod
    def custom_shift(data, months):
        return data.groupby(NCODPERS)[IND_TJCR_FIN_ULT1].shift(months)

    def extract_data_range(self):
        df_target_final = self.df_target[self.df_target[FECHA_DATO] == "2015-07-28"]
        df_final = self.df
        df_final["date"] = pd.to_datetime(df_final[FECHA_DATO])
        mask = (df_final["date"] > "2015-01-01") & (df_final["date"] <= "2015-06-30")
        df_final_final = self.df[mask]
        df_final_final = df_final_final[
            df_final_final[NCODPERS].isin(df_target_final[NCODPERS].unique())
        ]
        months_with_engagements = df_final_final[NCODPERS].value_counts().reset_index()
        only_full_customers = list(
            months_with_engagements[months_with_engagements.ncodpers == 6]["index"]
        )
        df_final_final = df_final_final[
            df_final_final[NCODPERS].isin(only_full_customers)
        ]
        df_target_final = df_target_final[
            df_target_final[NCODPERS].isin(df_final_final[NCODPERS].unique())
        ]
        constant_variables = df_final_final.columns[df_final_final.nunique() <= 1]
        df_final_final = df_final_final.drop(columns=list(constant_variables))
        self._release_memory()
        Serialization.save_state(df_target_final, "df_target_final", "data")
        Serialization.save_state(df_final_final, "df_final_final", "data")
        return df_target_final, df_final_final

    def _release_memory(self):
        self.df = None
        self.df_target = None
