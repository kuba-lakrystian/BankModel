from explainerdashboard import ClassifierExplainer, ExplainerDashboard
import pandas as pd
import matplotlib as mpl

from src.data_utils.constants import *

mpl.use("TkAgg")

DASHBOARD_NAME = "BankModel"


class ExplainerDashboardCustom:
    def __init__(self):
        self.df: pd.DataFrame = None
        self.xgb_model = None

    def load_objects(self, data, model):
        self.df = data
        self.xgb_model = model

    def train_explainer_dashboard(self, config):
        dashboard_yml_file = config[MODEL_SECTION][DASHBOARD_YML_NAME]
        dashboard_joblib_file = config[MODEL_SECTION][DASHBOARD_JOBLIB_NAME]

        X_train = self.df.drop(columns=[TARGET, NCODPERS])
        X_train.index = self.df[NCODPERS]
        y_train = self.df[TARGET]

        explainer = ClassifierExplainer(self.xgb_model, X_train, y_train)
        db = ExplainerDashboard(
            explainer,
            title=DASHBOARD_NAME,
            whatif=False,
            shap_interaction=False,
            decision_trees=False,
        )
        explainer.dump(dashboard_joblib_file)
        db.to_yaml(dashboard_yml_file, explainerfile=dashboard_joblib_file)
        print("ExplainerDashboard objects saved")
