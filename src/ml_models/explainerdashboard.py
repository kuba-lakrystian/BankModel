import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from src.data_utils.helpers import Serialization
from src.data_utils.constants import *

# #db.run()
# #explainerdashboard run dashboard.yaml inside virtualenv(in pycharm)


class ExplainerDashboardCustom:
    def __init__(self):
        self.df: pd.DataFrame = None
        self.xgb_model = None

    def load_objects(self, data, model):
        self.df = data
        self.xgb_model = model

    def train_explainer_dashboard(self):
        X_train = self.df.drop(columns=["target", FECHA_DATO, NCODPERS])
        y_train = self.df["target"]

        explainer = ClassifierExplainer(self.xgb_model, X_train, y_train)
        db = ExplainerDashboard(explainer, title="BankModel", whatif=False, shap_interaction=False,
                                decision_trees=False)
        explainer.dump("explainer.joblib")
        db.to_yaml("dashboard.yaml", explainerfile="explainer.joblib")
        print('ExplainerDashboard objects saved')











