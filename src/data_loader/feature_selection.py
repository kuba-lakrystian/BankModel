import numpy as np
import pandas as pd

from src.data_utils.constants import *

from sklearn.preprocessing import OneHotEncoder

pd.set_option("mode.chained_assignment", None)


class FeatureSelection:
    def __init__(self):
        self.df: pd.DataFrame = None
        self.df_target: pd.DataFrame = None
        self.encoder = None

    def load_data_to_selection(self, data_target, data_training):
        self.df = data_training
        self.df.index = list(self.df[NCODPERS])
        self.df_target = data_target

    def convert_to_dummy(self):
        df_individual = self.df[self.df[FECHA_DATO] == self.df[FECHA_DATO].max()]
        df_individual = df_individual[[IND_EMPLEADO, SEXO]]
        if self.encoder is None:
            self.encoder = OneHotEncoder(handle_unknown="ignore")
            self.encoder.fit(df_individual)
        df_transformed = pd.DataFrame(
            self.encoder.transform(df_individual).toarray(),
            columns=self.encoder.get_feature_names_out(),
        )
        df_transformed[NCODPERS] = df_individual.index
        return df_transformed

    def salary_preprocessing(self):
        df = self.df[[FECHA_DATO, NCODPERS, RENTA]]
        no_of_nan = (
            df[RENTA]
            .isnull()
            .groupby(df[NCODPERS])
            .sum()
            .astype(int)
            .reset_index(name="count")
        )
        # The same values for every month
        df = df[df[NCODPERS].isin(list(no_of_nan[no_of_nan["count"] <= 3][NCODPERS]))]
        df["period_id"] = (
            12 * pd.to_datetime(df[FECHA_DATO]).dt.year
            + pd.to_datetime(df[FECHA_DATO]).dt.month
        )
        df_3m = df[
            df["period_id"].isin(
                list(range(df["period_id"].max() - 2, df["period_id"].max()))
            )
        ]
        df_3m_agg = (
            df_3m.groupby(NCODPERS)[RENTA]
            .agg(["min", "max", "sum", "mean"])
            .reset_index()
        )
        df_6m = df[
            df["period_id"].isin(
                list(range(df["period_id"].max() - 5, df["period_id"].max()))
            )
        ]
        df_6m_agg = (
            df_6m.groupby(NCODPERS)[RENTA]
            .agg(["min", "max", "sum", "mean"])
            .reset_index()
        )
        df_3m_agg.columns = [NCODPERS] + [
            str(col) + "_3m" for col in df_3m_agg.columns if col != NCODPERS
        ]
        df_6m_agg.columns = [NCODPERS] + [
            str(col) + "_6m" for col in df_6m_agg.columns if col != NCODPERS
        ]
        df_fin = df_3m_agg.merge(df_6m_agg, how="inner", on=[NCODPERS])
        return df_fin

    def prepare_general_variables(self):
        df_individual = self.df[self.df[FECHA_DATO] == self.df[FECHA_DATO].max()]

    def release_memory(self):
        self.df = None
        self.df_target = None