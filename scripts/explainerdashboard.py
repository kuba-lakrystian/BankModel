import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from src.data_utils.helpers import Serialization
from src.data_utils.constants import *

merged = Serialization.load_state("merged_data", "data")
tmn_xgb_model = Serialization.load_state("tmm_xgb_model", "data/trained_instances")
xgb_model = tmn_xgb_model.xgb_model


X_train = merged.drop(columns=["target", FECHA_DATO, NCODPERS])
y_train = merged["target"]

explainer = ClassifierExplainer(xgb_model, X_train, y_train)
db = ExplainerDashboard(explainer, title="BankModel", whatif=False, shap_interaction=False, decision_trees=False)
explainer.dump("explainer.joblib")
db.to_yaml("dashboard.yaml", explainerfile="explainer.joblib")
#db.run()
#explainerdashboard run dashboard.yaml inside virtualenv(in pycharm)
