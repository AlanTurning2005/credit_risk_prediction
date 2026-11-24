from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "config"

BATCH_SIZE = 1024
SHUFFLE_BUFFER_SIZE = 5000

ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"
DAYS_EMPLOYED_ANOMALY_VALUE = 365243

SPLIT_CONFIG = {
    "test_size": 0.1,
    "valid_size": 0.1,
    "random_state": 23102026,
}

# 1. Raw input columns selected from application_train.csv (11 numerical + 10 categorical)
RAW_NUMERICAL_COLS = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
]

RAW_CATEGORICAL_COLS = [
    "NAME_CONTRACT_TYPE",
    "NAME_INCOME_TYPE",
    "CODE_GENDER",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
    "NAME_HOUSING_TYPE",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
]

ALL_RAW_SELECTED_COLS = [ID_COL, TARGET_COL] + RAW_NUMERICAL_COLS + RAW_CATEGORICAL_COLS

# 2. Danh sách 16 biến số sau Feature Engineering (giữ tương thích với data/processed & tf.data)
NUMERICAL_COLS = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "AGE",
    "YEARS_EMPLOYED",
    "DAYS_EMPLOYED_ANOM",
    "EXT_SOURCES_MEAN",
    "PAYMENT_RATE",
    "INCOME_CREDIT_PERC",
    "ANNUITY_INCOME_PERC",
]

# 3. Danh sách 10 biến danh mục
CATEGORICAL_COLS = list(RAW_CATEGORICAL_COLS)

# Domain validation constraints
STRICTLY_POSITIVE_COLS = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
]
NON_NEGATIVE_COLS = ["CNT_CHILDREN", "YEARS_EMPLOYED"]
MIN_ONE_COLS = ["CNT_FAM_MEMBERS"]
BOUNDED_01_COLS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

# 4. Thứ tự toàn bộ 27 cột trong file CSV lưu ra
CSV_COLUMNS = NUMERICAL_COLS + CATEGORICAL_COLS + [TARGET_COL]

# 5. Record defaults cho tf.data pipeline hiện có
RECORD_DEFAULTS = [
    [0.0] if col in NUMERICAL_COLS else ([0] if col == TARGET_COL else ["Missing"])
    for col in CSV_COLUMNS
]
