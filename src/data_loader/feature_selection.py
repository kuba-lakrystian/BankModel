import numpy as np
import pandas as pd

from src.data_utils.constants import *
from src.data_utils.iv import IV

from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from collections import defaultdict
from functools import reduce

pd.set_option("mode.chained_assignment", None)


class FeatureSelection:
    def __init__(self):
        self.df: pd.DataFrame = None
        self.df_target: pd.DataFrame = None
        self.encoder = None
        self.to_remove = None
        self.update_categorical = None

    def load_data_to_selection(self, data_target, data_training):
        self.df = data_training
        self.df.index = list(self.df[NCODPERS])
        self.df_target = data_target

    def convert_to_dummy(self, df):
        df_categorical = df.select_dtypes(include=["object"])
        if self.encoder is None:
            self.encoder = OneHotEncoder(handle_unknown="ignore")
            self.encoder.fit(df_categorical)
        df_transformed = pd.DataFrame(
            self.encoder.transform(df_categorical).toarray(),
            columns=self.encoder.get_feature_names_out(),
        )
        df = df.drop(columns=list(df_categorical.columns))
        df_final = pd.concat([df, df_transformed], axis=1)
        return df_final

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
        df_fin = df_3m_agg.merge(df_6m_agg, how=INNER, on=[NCODPERS])
        return df_fin

    def prepare_general_variables(self):
        df_individual = self.df[self.df[FECHA_DATO] == self.df[FECHA_DATO].max()]

    def release_memory(self):
        self.df = None
        self.df_target = None

    def find_best_variables(self, df):
        df = df.drop(columns=[FECHA_DATO, NCODPERS])
        d = defaultdict(preprocessing.LabelEncoder)
        fit = df.select_dtypes(include=["object"]).apply(
            lambda x: d[x.name].fit_transform(x)
        )
        for i in list(d.keys()):
            df[i] = d[i].transform(df[i])

        features = df[df.columns.difference(["target"])]
        labels = df["target"]

        iv_class = IV()
        iv_wyn = iv_class.data_vars(df, df.target)
        print(iv_wyn)

        # vi_wyn = iv_class.vi(features, labels)
        # print(vi_wyn)

        rfe_wyn = iv_class.rfe(features, labels)
        print(rfe_wyn)

        vi_extratrees_wyn = iv_class.vi_extratrees(features, labels)
        print(vi_extratrees_wyn)

        chi_sq_wyn = iv_class.chi_sq(features, labels)
        print(chi_sq_wyn)

        l1_wyn = iv_class.l1(features, labels)
        print(l1_wyn)

        dfs = [iv_wyn, rfe_wyn, vi_extratrees_wyn, chi_sq_wyn, l1_wyn]
        final_results = reduce(
            lambda left, right: pd.merge(left, right, on="index"), dfs
        )
        print(final_results.head())

        columns = ["IV", "Extratrees", "Chi_Square"]

        score_table = pd.DataFrame({}, [])
        score_table["index"] = final_results["index"]

        for i in columns:
            score_table[i] = (
                final_results["index"]
                .isin(list(final_results.nlargest(5, i)["index"]))
                .astype(int)
            )

        score_table["RFE"] = final_results["RFE"].astype(int)
        score_table["L1"] = final_results["L1"].astype(int)

        score_table["final_score"] = score_table.sum(axis=1)

        print("Importance: Final score:")
        print(score_table.sort_values("final_score", ascending=0))

        vif = iv_class.calculate_vif(features)
        print("VIF:")
        print(vif)

    def last_month_variables(self, df, train: bool = False):
        df_individual = df[df[FECHA_DATO] == df[FECHA_DATO].max()]
        df_individual = df_individual[
            [
                NCODPERS,
                IND_EMPLEADO,
                SEXO,
                PAIS_RESIDENCIA,
                AGE,
                IND_NUEVO,
                CANAL_ENTRADA,
                NOMPROV,
                SEGMENTO,
            ]
        ]
        if train:
            self.to_remove = list()
            self.update_categorical = {}
            df_categorical = df_individual.select_dtypes(include=["object"])
            for i in df_categorical.columns:
                count = df[i].value_counts(normalize=True).reset_index()
                count["cumsum"] = count[i].cumsum()
                if count.loc[0, "cumsum"] > 0.99:
                    self.to_remove = self.to_remove + [i]
                    continue
                if len(count) > 8:
                    mask = count["cumsum"] <= 0.85
                    to_take = count[mask]
                    self.update_categorical[i] = list(to_take["index"])
        if self.to_remove:
            df_individual = df_individual.drop(columns=self.to_remove)
        if self.update_categorical:
            for key in self.update_categorical:
                df_individual[key] = [
                    x if x in self.update_categorical[key] else "Other"
                    for x in df_individual[key]
                ]
        return df_individual
