VERSION = "1.0.0"

DATA_PATH = "data"
PICKLE = "pickle"

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

CHI_SQUARE = "Chi_Square"
IV = "IV"
EXTRATREES = "Extratrees"
L1 = "L1"
RFE = "RFE"
VIF = "VIF"
COUNT = "count"
DATE = "DATE"
INDEX = "index"
TARGET = "target"
OBJECT = "object"
PERIOD_ID = "period_id"
PREDICT = "predict"
PREDICT_PROBA = "predict_proba"
FECHA_DATO = "fecha_dato"  # Period ID
NCODPERS = "ncodpers"  # Customer ID
IND_EMPLEADO = "ind_empleado"  # Is it an employee: A active, B ex employed, F filial, N not employee, P passive
PAIS_RESIDENCIA = "pais_residencia"  # Citizenship
SEXO = "sexo"  # Gender
AGE = "age"  # Age
FECHA_ALTA = "fecha_alta"  # First day of a customer in the bank
IND_NUEVO = "ind_nuevo"  # New customer
ANTIGUAEDAD = (
    "antiguedad"  # How many months it has been since joining the bank (if < 6, then 6)
)
INDREL = "indrel"
ULT_FEC_CLI_1T = "ult_fec_cli_1t"
INDREL_1MES = "indrel_1mes"  # (First/Primary customer), 2 (co-owner ),P (Potential),3 (former primary), 4(former co-owner). At the beginning of a month
TIPREL_1MES = "tiprel_1mes"  # Customer relation type at the beginning of the month, A (active), I (inactive), P (former customer),R (Potential)
INDRESI = "indresi"  #
INDEXT = "indext"
CONYUEMP = "conyuemp"  # Spouse status
CANAL_ENTRADA = "canal_entrada"  # Channel
INDFALL = "indfall"  #
TIPODOM = "tipodom"  # Address type
COD_PROV = "cod_prov"  # Province code
NOMPROV = "nomprov"  # Province name
IND_ACTIVIDAD_CLIENTE = "ind_actividad_cliente"
RENTA = "renta"  # Gross income
SEGMENTO = "segmento"

IND_TJCR_FIN_ULT1 = "ind_tjcr_fin_ult1"  # Credit card

DATES_FOR_TRAIN_SET = ["2015-01-01", "2015-06-30", "2015-07-28"]
DATES_FOR_TEST_SET = ["2015-08-01", "2016-01-30", "2016-02-28"]
TRAIN_SET_FILE_NAMES = ["df_final_final_train", "df_target_final_train"]
TEST_SET_FILE_NAMES = ["df_final_final_test", "df_target_final_test"]

COLUMNS_TO_DROP = [
    FECHA_DATO,
    IND_NUEVO,
    "sum_6m",
    "sum_3m",
    "min_6m",
    "min_3m",
    "mean_6m",
    "mean_3m",
    "ind_ctop_fin_ult1_max3m",
    "ind_nom_pens_ult1_max3m",
    "ind_ecue_fin_ult1_max3m",
    "ind_cco_fin_ult1_max3m",
    "ind_cco_fin_ult1_max3m",
    "ind_ctpp_fin_ult1_max3m",
    INDEXT,
    "ind_ctma_fin_ult1_max3m",
    CONYUEMP,
    "ind_aval_fin_ult1_max3m",
    "ind_ahor_fin_ult1_max3m",
    "ind_cder_fin_ult1_max3m",
    "ind_deme_fin_ult1_max3m",
    "ind_dela_fin_ult1_max3m",
    "ind_ctju_fin_ult1_max3m",
    "ind_deco_fin_ult1_max3m",
    "ind_viv_fin_ult1_max3m",
    "ind_hip_fin_ult1_max3m",
    "ind_plan_fin_ult1_max3m",
    "ind_fond_fin_ult1_max3m",
    "ind_valo_fin_ult1_max3m",
    "ind_pres_fin_ult1_max3m",
]

PERCENT_FOR_CONSTANT_VARIABLE = 0.99
NUMBER_OF_SIGNIFICANT_CATEGORIES = 8
PERCENT_OF_SIGNIFICANT_CATEGORIES = 0.85
VALID_IMPORTANCE_PERCENT = 0.95
SET_SEED = 42
