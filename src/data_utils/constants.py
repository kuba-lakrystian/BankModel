import pandas as pd

DATA_PATH = "data"

DOT_STR = "."
EMPTY_STR = ""
SLASH_STR = "/"

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
