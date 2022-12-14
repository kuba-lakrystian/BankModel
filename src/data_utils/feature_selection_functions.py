import numpy as np
import pandas as pd
import re
import scipy.stats.stats as stats
import traceback
from pandas import Series
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import chi2, RFE, SelectFromModel, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.data_utils.constants import *

max_bin = 20
force_bin = 3

BUCKET = "Bucket"
COUNT = "COUNT"
DIST_EVENT = "DIST_EVENT"
DIST_NON_EVENT = "DIST_NON_EVENT"
EVENT = "EVENT"
EVENT_RATE = "EVENT_RATE"
FEATURES = "Features"
MAX_VALUE = "MAX_VALUE"
MIN_VALUE = "MIN_VALUE"
NON_EVENT_RATE = "NON_EVENT_RATE"
NONEVENT = "NONEVENT"
X_VAR = "X"
VAR_NAME = "VAR_NAME"
VAR = "VAR"
WOE = "WOE"
Y_VAR = "Y"


class FeatureSelectionFunctions:
    @staticmethod
    def mono_bin(Y, X, n=max_bin):
        df1 = pd.DataFrame({X_VAR: X, Y_VAR: Y})
        justmiss = df1[[X_VAR, Y_VAR]][df1[X_VAR].isnull()]
        notmiss = df1[[X_VAR, Y_VAR]][df1[X_VAR].notnull()]
        r = 0
        while np.abs(r) < 1:
            try:
                d1 = pd.DataFrame(
                    {X_VAR: notmiss.X, Y_VAR: notmiss.Y, BUCKET: pd.qcut(notmiss.X, n)}
                )
                d2 = d1.groupby(BUCKET, as_index=True)
                r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
                n = n - 1
            except Exception as e:
                n = n - 1

        if len(d2) == 1:
            n = force_bin
            bins = notmiss.X.quantile(np.linspace(0, 1, n))
            if len(bins) == 2:
                bins = np.insert(bins, 0, 1)
                bins[1] = bins[1] - (bins[1] / 2)
            d1 = pd.DataFrame(
                {
                    X_VAR: notmiss.X,
                    Y_VAR: notmiss.Y,
                    BUCKET: pd.cut(notmiss.X, np.unique(bins), include_lowest=True),
                }
            )
            d2 = d1.groupby(BUCKET, as_index=True)

        d3 = pd.DataFrame({}, index=[])
        d3[MIN_VALUE] = d2.min().X
        d3[MAX_VALUE] = d2.max().X
        d3[COUNT] = d2.count().Y
        d3[EVENT] = d2.sum().Y
        d3[NONEVENT] = d2.count().Y - d2.sum().Y
        d3 = d3.reset_index(drop=True)

        if len(justmiss.index) > 0:
            d4 = pd.DataFrame({MIN_VALUE: np.nan}, index=[0])
            d4[MAX_VALUE] = np.nan
            d4[COUNT] = justmiss.count().Y
            d4[EVENT] = justmiss.sum().Y
            d4[NONEVENT] = justmiss.count().Y - justmiss.sum().Y
            d3 = d3.append(d4, ignore_index=True)

        d3[EVENT_RATE] = d3[EVENT] / d3[COUNT]
        d3[NON_EVENT_RATE] = d3[NONEVENT] / d3[COUNT]
        d3[DIST_EVENT] = d3[EVENT] / d3.sum().EVENT
        d3[DIST_NON_EVENT] = d3[NONEVENT] / d3.sum().NONEVENT
        d3[WOE] = np.log(d3[DIST_EVENT] / d3[DIST_NON_EVENT])
        d3[IV] = (d3[DIST_EVENT] - d3[DIST_NON_EVENT]) * np.log(
            d3[DIST_EVENT] / d3[DIST_NON_EVENT]
        )
        d3[VAR_NAME] = VAR
        d3 = d3[
            [
                VAR_NAME,
                MIN_VALUE,
                MAX_VALUE,
                COUNT,
                EVENT,
                EVENT_RATE,
                NONEVENT,
                NON_EVENT_RATE,
                DIST_EVENT,
                DIST_NON_EVENT,
                WOE,
                IV,
            ]
        ]
        d3 = d3.replace([np.inf, -np.inf], 0)
        d3.IV = d3.IV.sum()

        return d3

    @staticmethod
    def char_bin(Y, X):
        df1 = pd.DataFrame({X_VAR: X, Y_VAR: Y})
        justmiss = df1[[X_VAR, Y_VAR]][df1.X.isnull()]
        notmiss = df1[[X_VAR, Y_VAR]][df1.X.notnull()]
        df2 = notmiss.groupby(X_VAR, as_index=True)

        d3 = pd.DataFrame({}, index=[])
        d3[COUNT] = df2.count().Y
        d3[MIN_VALUE] = df2.sum().Y.index
        d3[MAX_VALUE] = d3[MIN_VALUE]
        d3[EVENT] = df2.sum().Y
        d3[NONEVENT] = df2.count().Y - df2.sum().Y

        if len(justmiss.index) > 0:
            d4 = pd.DataFrame({MIN_VALUE: np.nan}, index=[0])
            d4[MAX_VALUE] = np.nan
            d4[COUNT] = justmiss.count().Y
            d4[EVENT] = justmiss.sum().Y
            d4[NONEVENT] = justmiss.count().Y - justmiss.sum().Y
            d3 = d3.append(d4, ignore_index=True)

        d3[EVENT_RATE] = d3[EVENT] / d3[COUNT]
        d3[NON_EVENT_RATE] = d3[NONEVENT] / d3[COUNT]
        d3[DIST_EVENT] = d3[EVENT] / d3.sum().EVENT
        d3[DIST_NON_EVENT] = d3[NONEVENT] / d3.sum().NONEVENT
        d3[WOE] = np.log(d3[DIST_EVENT] / d3[DIST_NON_EVENT])
        d3[IV] = (d3[DIST_EVENT] - d3[DIST_NON_EVENT]) * np.log(
            d3[DIST_EVENT] / d3[DIST_NON_EVENT]
        )
        d3[VAR_NAME] = VAR
        d3 = d3[
            [
                VAR_NAME,
                MIN_VALUE,
                MAX_VALUE,
                COUNT,
                EVENT,
                EVENT_RATE,
                NONEVENT,
                NON_EVENT_RATE,
                DIST_EVENT,
                DIST_NON_EVENT,
                WOE,
                IV,
            ]
        ]
        d3 = d3.replace([np.inf, -np.inf], 0)
        d3.IV = d3.IV.sum()
        d3 = d3.reset_index(drop=True)

        return d3

    def data_vars(self, df1, target):
        stack = traceback.extract_stack()
        filename, lineno, function_name, code = stack[-2]
        vars_name = re.compile(r"\((.*?)\).*$").search(code).groups()[0]
        final = (re.findall(r"[\w']+", vars_name))[-1]

        x = df1.dtypes.index
        count = -1

        for i in x:
            if i.upper() not in (final.upper()):
                if np.issubdtype(df1[i], np.number) and len(Series.unique(df1[i])) > 2:
                    conv = self.mono_bin(target, df1[i])
                    conv[VAR_NAME] = i
                    count = count + 1
                else:
                    conv = self.char_bin(target, df1[i])
                    conv[VAR_NAME] = i
                    count = count + 1

                if count == 0:
                    iv_df = conv
                else:
                    iv_df = iv_df.append(conv, ignore_index=True)

        iv = pd.DataFrame({IV: iv_df.groupby(VAR_NAME).IV.max()})
        iv = iv.reset_index()
        iv = iv.rename(columns={VAR_NAME: INDEX})
        iv.sort_values([IV], ascending=0)
        return iv.sort_values([IV], ascending=0)

    @staticmethod
    def rfe(features, labels):
        model = LogisticRegression()
        rfe = RFE(model, n_features_to_select=20)
        rfe.fit(features, labels)
        selected = pd.DataFrame(
            rfe.support_, columns=[RFE_VALUE], index=features.columns
        )
        selected = selected.reset_index()
        return selected

    @staticmethod
    def vi_extratrees(features, labels):
        model = ExtraTreesClassifier()
        model.fit(features, labels)
        fi = pd.DataFrame(
            model.feature_importances_, columns=[EXTRATREES], index=features.columns
        )
        fi = fi.reset_index()
        return fi.sort_values([EXTRATREES], ascending=0)

    @staticmethod
    def chi_sq(features, labels):
        model = SelectKBest(score_func=chi2, k=5)
        fit = model.fit(features, labels)
        chi_sq = pd.DataFrame(fit.scores_, columns=[CHI_SQUARE], index=features.columns)
        chi_sq = chi_sq.reset_index()
        return chi_sq.sort_values(CHI_SQUARE, ascending=0)

    @staticmethod
    def l1(features, labels):
        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(features, labels)
        model = SelectFromModel(lsvc, prefit=True)
        l1 = pd.DataFrame(model.get_support(), columns=[L1], index=features.columns)
        l1 = l1.reset_index()
        return l1

    @staticmethod
    def calculate_vif(features):
        vif = pd.DataFrame()
        vif[FEATURES] = features.columns
        vif[VIF] = [
            variance_inflation_factor(features.values, i)
            for i in range(features.shape[1])
        ]
        return vif
