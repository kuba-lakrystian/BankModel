VERSION = '1.0.0'

DATA_PATH = "data"

COERCE = 'coerce'
DOT_STR = "."
EMPTY_STR = ""
INNER = 'inner'
SLASH_STR = "/"
UNDERSCORE_STR = "_"

DATE = "DATE"
INDEX = 'index'
TARGET = "target"
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

INDREL_1MES = "indrel_1mes"  # (First/Primary customer), 2 (co-owner ),P (Potential),3 (former primary), 4(former co-owner). At the beginning of a month
TIPREL_1MES = "tiprel_1mes"  # Customer relation type at the beginning of the month, A (active), I (inactive), P (former customer),R (Potential)
INDRESI = "indresi"  #

CONYUEMP = "conyuemp"  # Spouse status
CANAL_ENTRADA = "canal_entrada"  # Channel
INDFALL = "indfall"  #
TIPODOM = "tipodom"  # Address type
COD_PROV = "cod_prov"  # Province code
NOMPROV = "nomprov"  # Province name
RENTA = "renta"  # Gross income
SEGMENTO = "segmento"

IND_TJCR_FIN_ULT1 = "ind_tjcr_fin_ult1"  # Credit card

DATES_FOR_TRAIN_SET = ["2015-01-01", "2015-06-30", "2015-07-28"]
DATES_FOR_TEST_SET = ["2015-08-01", "2016-01-30", "2016-02-28"]
TRAIN_SET_FILE_NAMES = ['df_final_final_train', 'df_target_final_train']
TEST_SET_FILE_NAMES = ['df_final_final_test', 'df_target_final_test']

COLUMNS_TO_DROP = [
    FECHA_DATO,
    IND_NUEVO,
    "sum_6m",
    "sum_3m",
    "min_6m",
    "min_3m",
    "mean_6m",
    "mean_3m",
]
