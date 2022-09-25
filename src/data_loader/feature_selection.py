import pandas as pd

from src.data_utils.constants import *
from src.data_utils.iv import IV

from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from collections import defaultdict
from functools import reduce

pd.set_option("mode.chained_assignment", None)

CUMSUM_VAR = "cumsum_var"
OTHER = "Other"


class FeatureSelection:
    def __init__(self):
        self.encoder = None
        self.to_remove = None
        self.update_categorical = None

    def convert_to_dummy(self, df, columns_to_drop):
        df = df.drop(columns=columns_to_drop)
        df_categorical = df.select_dtypes(include=[OBJECT])
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

    @staticmethod
    def prepare_aggregates_for_salary(df, months):
        df_xm = df[
            df[PERIOD_ID].isin(
                list(range(df[PERIOD_ID].max() - months + 1, df[PERIOD_ID].max() + 1))
            )
        ]
        df_xm_agg = (
            df_xm.groupby(NCODPERS)[RENTA].agg([MIN, MAX, SUM, MEAN]).reset_index()
        )
        df_xm_agg.columns = [NCODPERS] + [
            f"{str(col)}_{months}m" for col in df_xm_agg.columns if col != NCODPERS
        ]
        return df_xm_agg

    @staticmethod
    def prepare_product_activity_variables(df):
        activity_columns = [
            col for col in df.columns if "ult1" in col and col != IND_TJCR_FIN_ULT1
        ]
        df = df[[FECHA_DATO, NCODPERS] + activity_columns]
        df[PERIOD_ID] = (
            12 * pd.to_datetime(df[FECHA_DATO]).dt.year
            + pd.to_datetime(df[FECHA_DATO]).dt.month
        )
        df_xm = df[
            df[PERIOD_ID].isin(
                list(range(df[PERIOD_ID].max() - 3 + 1, df[PERIOD_ID].max() + 1))
            )
        ]
        df_xm = df_xm.drop(columns=[FECHA_DATO, PERIOD_ID])
        df_xm_agg = df_xm.groupby(NCODPERS).agg(MAX).add_suffix("_max3m")
        df_xm_agg = df_xm_agg.reset_index()
        return df_xm_agg

    def salary_preprocessing(self, data):
        df = data[[FECHA_DATO, NCODPERS, RENTA]]
        no_of_nan = (
            df[RENTA]
            .isnull()
            .groupby(df[NCODPERS])
            .sum()
            .astype(int)
            .reset_index(name=COUNT)
        )

        df = df[df[NCODPERS].isin(list(no_of_nan[no_of_nan[COUNT] <= 3][NCODPERS]))]
        df[PERIOD_ID] = (
            12 * pd.to_datetime(df[FECHA_DATO]).dt.year
            + pd.to_datetime(df[FECHA_DATO]).dt.month
        )

        df_3m_agg = self.prepare_aggregates_for_salary(df, 3)
        df_6m_agg = self.prepare_aggregates_for_salary(df, 6)
        df_fin = df_3m_agg.merge(df_6m_agg, how=INNER, on=[NCODPERS])
        return df_fin

    @staticmethod
    def find_best_variables(df):
        df = df.drop(columns=[FECHA_DATO, NCODPERS])
        d = defaultdict(preprocessing.LabelEncoder)
        fit = df.select_dtypes(include=[OBJECT]).apply(
            lambda x: d[x.name].fit_transform(x)
        )
        for i in list(d.keys()):
            df[i] = d[i].transform(df[i])

        features = df[df.columns.difference([TARGET])]
        labels = df[TARGET]

        iv_class = IV()
        iv_wyn = iv_class.data_vars(df, df.target)
        print(iv_wyn)

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
            df_categorical = df_individual.select_dtypes(include=[OBJECT])
            for i in df_categorical.columns:
                count = df[i].value_counts(normalize=True).reset_index()
                count[CUMSUM_VAR] = count[i].cumsum()
                if count.loc[0, CUMSUM_VAR] > PERCENT_FOR_CONSTANT_VARIABLE:
                    self.to_remove = self.to_remove + [i]
                    continue
                if len(count) > NUMBER_OF_SIGNIFICANT_CATEGORIES:
                    mask = count[CUMSUM_VAR] <= PERCENT_OF_SIGNIFICANT_CATEGORIES
                    to_take = count[mask]
                    self.update_categorical[i] = list(to_take[INDEX])
        if self.to_remove:
            df_individual = df_individual.drop(columns=self.to_remove)
        if self.update_categorical:
            for key in self.update_categorical:
                df_individual[key] = [
                    x if x in self.update_categorical[key] else OTHER
                    for x in df_individual[key]
                ]
        return df_individual

    def prepare_methed_dataset(self, data_training, data_target, train: bool = False):
        general_variables = self.last_month_variables(data_training, train=train)
        salary_variables = self.salary_preprocessing(data_training)
        activity_variables = self.prepare_product_activity_variables(data_training)
        merged = (
            general_variables.merge(salary_variables, how=INNER, on=[NCODPERS])
            .merge(activity_variables, how=INNER, on=[NCODPERS])
            .merge(data_target, how=INNER, on=[NCODPERS])
        )
        return merged
