VERSION = "1.0.0"

# CONFIG FILE

INPUT_SECTION = "Input"
DATA_PATH = "data_path"
PRETRAINED_TRAIN = "prepared_data_train"
PRETRAINED_TRAIN_LABELS = "prepared_data_train_labels"
PRETRAINED_TEST = "prepared_data_test"
PRETRAINED_TEST_LABELS = "prepared_data_test_labels"
PRETRAINED_OOT = "prepared_data_oot"
PRETRAINED_OOT_LABELS = "prepared_data_oot_labels"
RAW_DATA_FILE = "raw_data_file"
MODEL_SECTION = "Model"
MODEL_NAME = "model_name"
MODEL_PATH = "model_path"
DASHBOARD_YML_NAME = "dashboard_yml_name"
DASHBOARD_JOBLIB_NAME = "dashboard_joblib_name"
PARAMETERS_SECTION = "Parameters"
FEATURE_SELECTION_PARAMETER = "feature_selection"
OPT_MODEL_PARAMETER = "opt_model"
GARBAGE_MODEL_PARAMETER = "garbage_model"
VALUES_SECTION = "Values"
PERCENT_FOR_CONSTANT_VARIABLE_VALUE = "percent_for_constant_variable"
NUMBER_OF_SIGNIFICANT_CATEGORIES_VALUE = "number_of_significant_categories"
PERCENT_OF_SIGNIFICANT_CATEGORIES_VALUE = "percent_of_significant_categories"
VALID_IMPORTANCE_PERCENT_VALUE = "valid_importance_percent"
SET_SEED_VALUE = "set_seed"
CONFIG_FILE = "config.ini"

# GENERAL

COERCE = "coerce"
DOT_STR = "."
EMPTY_STR = ""
INNER = "inner"
SLASH_STR = "/"
UNDERSCORE_STR = "_"
MIN = "min"
MAX = "max"
MEAN = "mean"
SUM = "sum"
TRUE_STR = "True"

# VARIABLES

CHI_SQUARE = "Chi_Square"
IV = "IV"
EXTRATREES = "Extratrees"
L1 = "L1"
RFE_VALUE = "RFE_value"
VIF = "VIF"
COUNT = "count"
DATE = "DATE"
INDEX = "index"
TARGET = "target"
OBJECT = "object"
PERIOD_ID = "period_id"
PREDICT = "predict"
PREDICT_PROBA = "predict_proba"
FECHA_DATO = "fecha_dato"
NCODPERS = "ncodpers"
IND_EMPLEADO = "ind_empleado"
PAIS_RESIDENCIA = "pais_residencia"
SEXO = "sexo"
AGE = "age"
FECHA_ALTA = "fecha_alta"
IND_NUEVO = "ind_nuevo"
ANTIGUAEDAD = "antiguedad"
INDREL = "indrel"
ULT_FEC_CLI_1T = "ult_fec_cli_1t"
INDREL_1MES = "indrel_1mes"
TIPREL_1MES = "tiprel_1mes"
INDRESI = "indresi"  #
INDEXT = "indext"
CONYUEMP = "conyuemp"
CANAL_ENTRADA = "canal_entrada"
INDFALL = "indfall"
TIPODOM = "tipodom"
COD_PROV = "cod_prov"
NOMPROV = "nomprov"
IND_ACTIVIDAD_CLIENTE = "ind_actividad_cliente"
RENTA = "renta"
SEGMENTO = "segmento"
IND_TJCR_FIN_ULT1 = "ind_tjcr_fin_ult1"
SUM_6M = "sum_6m"
SUM_3M = "sum_3m"
MIN_6M = "min_6m"
MIN_3M = "min_3m"
MEAN_6M = "mean_6m"
MEAN_3M = "mean_3m"

# DATES FOR TRAIN AND TEST SETS

DATES_FOR_TRAIN_SET = ["2015-01-01", "2015-06-30", "2015-07-28"]
DATES_FOR_TEST_SET = ["2015-08-01", "2016-01-30", "2016-02-28"]
DATES_FOR_OOT_SET = ["2015-10-01", "2016-03-30", "2016-04-28"]

# REDUNDANT VARIABLES

COLUMNS_TO_DROP = [
    FECHA_DATO,
    IND_NUEVO,
    "sum_6m",
    "sum_3m",
    "min_6m",
    "min_3m",
    "mean_6m",
    "mean_3m",
    "ind_ctma_fin_ult1_max3m",
    CONYUEMP,
    "ind_aval_fin_ult1_max3m",
    "ind_ahor_fin_ult1_max3m",
    "ind_cder_fin_ult1_max3m",
    "ind_deme_fin_ult1_max3m",
    "ind_ctju_fin_ult1_max3m",
    "ind_deco_fin_ult1_max3m",
    "ind_viv_fin_ult1_max3m",
    "ind_hip_fin_ult1_max3m",
    "ind_plan_fin_ult1_max3m",
    "ind_pres_fin_ult1_max3m",
    "ult_fec_cli_1t",
    NOMPROV,
]
