import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.data_utils.constants import *

data_train_raw = pd.read_csv(
    EMPTY_STR.join([DATA_PATH, "data_recommendation_engine/train_ver2.csv"])
)

# Target: who will buy a credit card
# Definition: 1 - when there is no credit card three months before and is in two consecutive months
data_sample = data_train_raw[[FECHA_DATO, NCODPERS, IND_TJCR_FIN_ULT1]]

# Date range
data_sample[FECHA_DATO].min()  # '2015-01-28'
data_sample[FECHA_DATO].max()  # '2016-05-28'

# Create target
data_sample["ind_tjcr_fin_ult1_1_mo_before"] = data_sample.groupby(NCODPERS)[
    IND_TJCR_FIN_ULT1
].shift(1)
data_sample["ind_tjcr_fin_ult1_2_mo_before"] = data_sample.groupby(NCODPERS)[
    IND_TJCR_FIN_ULT1
].shift(2)
data_sample["ind_tjcr_fin_ult1_3_mo_before"] = data_sample.groupby(NCODPERS)[
    IND_TJCR_FIN_ULT1
].shift(3)
data_sample["ind_tjcr_fin_ult1_1_mo_after"] = data_sample.groupby(NCODPERS)[
    IND_TJCR_FIN_ULT1
].shift(-1)

data_sample["target"] = np.where(
    (data_sample["ind_tjcr_fin_ult1_1_mo_before"] == 0)
    & (data_sample["ind_tjcr_fin_ult1_2_mo_before"] == 0)
    & (data_sample["ind_tjcr_fin_ult1_3_mo_before"] == 0)
    & (data_sample["ind_tjcr_fin_ult1"] == 1)
    & (data_sample["ind_tjcr_fin_ult1_1_mo_after"] == 1),
    1,
    0,
)

# Test
exemplary_data = data_sample[data_sample[NCODPERS] == 544886].sort_values(FECHA_DATO)

# How many purchases
data_sample.value_counts(TARGET)

# Over time
targets = data_sample[data_sample[TARGET] == 1]
table = (
    pd.DataFrame(targets[FECHA_DATO].value_counts()).reset_index().sort_values("index")
)
data_sample[FECHA_DATO].value_counts()


plt.plot(table["index"], table["fecha_dato"])
# 2015-07-28
