import os.path
import configparser

from src.data_loader.data_loader import DataLoader
from src.data_loader.data_preprocess import DataPreprocess
from src.data_loader.feature_selection import FeatureSelection
from src.data_utils.constants import *
from src.ml_models.explainerdashboard import ExplainerDashboardCustom
from src.data_utils.helpers import Serialization
from src.ml_models.train_ml_model import TrainMLModel


def train():
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    data_path = config[INPUT_SECTION][DATA_PATH]
    pretrained_train = config[INPUT_SECTION][PRETRAINED_TRAIN]
    pretrained_train_labels = config[INPUT_SECTION][PRETRAINED_TRAIN_LABELS]
    pretrained_test = config[INPUT_SECTION][PRETRAINED_TEST]
    pretrained_test_labels = config[INPUT_SECTION][PRETRAINED_TEST_LABELS]
    model_path = config[MODEL_SECTION][MODEL_PATH]
    model_name = config[MODEL_SECTION][MODEL_NAME]
    dashboard_yml_file = config[MODEL_SECTION][DASHBOARD_YML_NAME]
    dashboard_joblib_file = config[MODEL_SECTION][DASHBOARD_JOBLIB_NAME]
    feature_selection: bool = config[PARAMETERS_SECTION][FEATURE_SELECTION_PARAMETER] == TRUE_STR
    opt_model: bool = config[PARAMETERS_SECTION][OPT_MODEL_PARAMETER] == TRUE_STR
    garbage_model: bool = config[PARAMETERS_SECTION][GARBAGE_MODEL_PARAMETER] == TRUE_STR
    if (
        os.path.isfile(SLASH_STR.join([data_path, pretrained_train_labels]))
        and os.path.isfile(SLASH_STR.join([data_path, pretrained_train]))
        and os.path.isfile(SLASH_STR.join([data_path, pretrained_test_labels]))
        and os.path.isfile(SLASH_STR.join([data_path, pretrained_test]))
    ):
        print("Loading pretrained data files")
        data_target_train = Serialization.load_state(pretrained_train_labels, data_path)
        data_training_train = Serialization.load_state(pretrained_train, data_path)
        data_target_test = Serialization.load_state(pretrained_test_labels, data_path)
        data_training_test = Serialization.load_state(pretrained_test, data_path)
        print("Pretrained data files loaded")
    else:
        dl = DataLoader()
        print(f"Raw data loading")
        data_train = dl.train_load(config)
        print(f"Raw data loaded")
        dp = DataPreprocess()
        dp.load_data(config, data_train)
        dp.prepare_target()
        data_target_train, data_training_train = dp.extract_data_range(
            DATES_FOR_TRAIN_SET, pretrained_train, pretrained_train_labels
        )
        data_target_test, data_training_test = dp.extract_data_range(
            DATES_FOR_TEST_SET, pretrained_test, pretrained_test_labels
        )
        dp.release_memory()
        print(f"Data train and test serialized and saved in {data_path}")

    print("Prepare data for modelling")
    fs = FeatureSelection()
    merged_train = fs.prepare_methed_dataset(
        config, data_training_train, data_target_train, train=True
    )
    if feature_selection is True:
        print("Deep feature selection started")
        fs.find_best_variables(merged_train)
        print(
            "Explore results above and exclude redundant variables by modifying"
            " COLUMNS_TO_DROP before you train final model"
        )
        return True
    merged_train_valid = fs.convert_to_dummy(
        merged_train, columns_to_drop=COLUMNS_TO_DROP
    )
    merged_test = fs.prepare_methed_dataset(
        config, data_training_test, data_target_test, train=False
    )
    merged_test_valid = fs.convert_to_dummy(
        merged_test, columns_to_drop=COLUMNS_TO_DROP
    )
    print("Training model")
    tmm = TrainMLModel()
    variables_to_optimise = tmm.fit(
        config,
        merged_train_valid,
        bayesian_optimisation=True,
        random_search=False,
        apply_smote=False,
    )
    tmm.predict(merged_test_valid)
    Serialization.save_state(tmm, model_name, model_path)
    if opt_model is True:
        merged_train_valid = merged_train_valid[variables_to_optimise]
        merged_test_valid = merged_test_valid[variables_to_optimise]
        tmm_opt = TrainMLModel()
        tmm_opt.fit(
            config,
            merged_train_valid,
            bayesian_optimisation=False,
            random_search=False,
            apply_smote=False,
        )
        tmm_opt.predict(merged_test_valid)
    if not os.path.isfile(dashboard_yml_file) and os.path.isfile(dashboard_joblib_file):
        print("ExplainerDashboard calculated")
        edc = ExplainerDashboardCustom()
        if opt_model:
            edc.load_objects(merged_train_valid, tmm_opt.xgb_model)
        else:
            edc.load_objects(merged_train_valid, tmm.xgb_model)
        edc.train_explainer_dashboard(config)
    if garbage_model is True:
        print("Garbage model")
        fs_garbage = FeatureSelection()
        proper_columns = [
            x
            for x in list(merged_train.columns)
            if x not in COLUMNS_TO_DROP and x != TARGET and x != NCODPERS
        ] + [FECHA_DATO, CONYUEMP]
        merged_train_garbage = fs_garbage.convert_to_dummy(
            merged_train, columns_to_drop=proper_columns
        )
        merged_test_garbage = fs_garbage.convert_to_dummy(
            merged_test, columns_to_drop=proper_columns
        )
        tmm_garbage = TrainMLModel()
        tmm_garbage.fit(
            config,
            merged_train_garbage,
            bayesian_optimisation=False,
            random_search=False,
            apply_smote=False,
        )
        tmm_garbage.predict(merged_test_garbage)


if __name__ == "__main__":
    train()
