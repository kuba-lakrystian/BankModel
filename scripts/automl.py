import h2o
from h2o.automl import H2OAutoML

from src.data_utils.constants import *
from src.data_utils.helpers import Serialization


SET_SEED = 42

h2o.init()

merged = Serialization.load_state("merged_data", "data")

X_train = merged.drop(columns=[FECHA_DATO, NCODPERS])
X_train_h2o = h2o.H2OFrame(X_train)

x = X_train_h2o.columns
y = TARGET
x = x.remove(y)

X_train_h2o[y] = X_train_h2o[y].asfactor()

aml = H2OAutoML(max_models=5, seed=SET_SEED)
aml.train(x=x, y=y, training_frame=X_train_h2o)

lb = aml.leaderboard
lb.head(rows=lb.nrows)
