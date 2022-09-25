import os.path

from src.data_loader.data_loader import DataLoader
from src.data_loader.data_preprocess import DataPreprocess
from src.data_loader.feature_selection import FeatureSelection
from src.data_utils.constants import *
from src.ml_models.explainerdashboard import ExplainerDashboardCustom
from src.data_utils.helpers import Serialization
from src.ml_models.train_ml_model import TrainMLModel


def train(feature_selection: bool = False, opt_model: bool = False):
    if (
        os.path.isfile(
            SLASH_STR.join([DATA_PATH, DOT_STR.join([TRAIN_SET_FILE_NAMES[1], PICKLE])])
        )
        and os.path.isfile(
            SLASH_STR.join([DATA_PATH, DOT_STR.join([TRAIN_SET_FILE_NAMES[0], PICKLE])])
        )
        and os.path.isfile(
            SLASH_STR.join([DATA_PATH, DOT_STR.join([TEST_SET_FILE_NAMES[1], PICKLE])])
        )
        and os.path.isfile(
            SLASH_STR.join([DATA_PATH, DOT_STR.join([TEST_SET_FILE_NAMES[0], PICKLE])])
        )
    ):
        print("Loading pretrained data files")
        data_target_train = Serialization.load_state(TRAIN_SET_FILE_NAMES[1], DATA_PATH)
        data_training_train = Serialization.load_state(
            TRAIN_SET_FILE_NAMES[0], DATA_PATH
        )
        data_target_test = Serialization.load_state(TEST_SET_FILE_NAMES[1], DATA_PATH)
        data_training_test = Serialization.load_state(TEST_SET_FILE_NAMES[0], DATA_PATH)
        print("Pretrained data files loaded")
    else:
        dl = DataLoader()
        print(f"Raw data loading")
        data_train = dl.train_load()
        print(f"Raw data loaded")
        dp = DataPreprocess()
        dp.load_data(data_train)
        dp.prepare_target()
        data_target_train, data_training_train = dp.extract_data_range(
            DATES_FOR_TRAIN_SET, TRAIN_SET_FILE_NAMES
        )
        data_target_test, data_training_test = dp.extract_data_range(
            DATES_FOR_TEST_SET, TEST_SET_FILE_NAMES
        )
        dp.release_memory()
        print(f"Data train and test serialized and saved in {DATA_PATH}")

    print("Prepare data for modelling")
    fs = FeatureSelection()
    merged_train = fs.prepare_methed_dataset(
        data_training_train, data_target_train, train=True
    )
    if feature_selection is True:
        print("Deep feature selection started")
        fs.find_best_variables(merged_train)
        print(
            "Explore results above and exclude redundant variables by modifying"
            " COLUMNS_TO_DROP before you train final model"
        )
        return True
    merged_train = fs.convert_to_dummy(merged_train, columns_to_drop=COLUMNS_TO_DROP)
    merged_test = fs.prepare_methed_dataset(
        data_training_test, data_target_test, train=False
    )
    merged_test = fs.convert_to_dummy(merged_test, columns_to_drop=COLUMNS_TO_DROP)
    print("Training model")
    tmm = TrainMLModel()
    tmm.load_data_for_model(merged_train)
    a, variables_to_optimise = tmm.fit(
        bayesian_optimisation=False, random_search=False, apply_smote=False
    )
    tmm.release_memory()
    tmm.load_data_for_model(merged_test)
    tmm.predict()
    Serialization.save_state(tmm, "tmm_xgb_model", "data/trained_instances")
    if opt_model:
        merged_train = merged_train[variables_to_optimise]
        merged_test = merged_test[variables_to_optimise]
        tmm_opt = TrainMLModel()
        tmm_opt.load_data_for_model(merged_train)
        a, variables_to_optimise = tmm_opt.fit(
            bayesian_optimisation=False, random_search=False, apply_smote=False
        )
        tmm_opt.release_memory()
        tmm_opt.load_data_for_model(merged_test)
        tmm_opt.predict()
    if not os.path.isfile("dashboard.yaml") and os.path.isfile("explainer.joblib"):
        print("ExplainerDashboard calculated")
        edc = ExplainerDashboardCustom()
        if opt_model:
            edc.load_objects(merged_train, tmm_opt.xgb_model)
        else:
            edc.load_objects(merged_train, tmm.xgb_model)
        edc.train_explainer_dashboard()


if __name__ == "__main__":
    train(feature_selection=True, opt_model=False)
