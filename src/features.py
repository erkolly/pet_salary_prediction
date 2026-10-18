"""
features.py — подготовка признаков.

Идея из EDA: главное — опыт, должность (но сгруппированная по смыслу)
и где человек живёт/работает. Поэтому:
  * experience_level, company_size  → порядковые числа (EN < MI < SE < EX)
  * job_title (50 уникальных, много единичных) → job_family + is_leadership
  * страны (50+ уникальных) → регион с похожим уровнем зарплат
  * salary и salary_currency выкидываем: это утечка таргета
    (salary — та же зарплата, только в другой валюте)
"""
import numpy as np
import pandas as pd

EXPERIENCE_ORDER = {"EN": 0, "MI": 1, "SE": 2, "EX": 3}
COMPANY_SIZE_ORDER = {"S": 0, "M": 1, "L": 2}

EXPERIENCE_LABELS = {"EN": "Junior (EN)", "MI": "Middle (MI)", "SE": "Senior (SE)", "EX": "Executive (EX)"}
COMPANY_SIZE_LABELS = {"S": "Маленькая (<50)", "M": "Средняя (50–250)", "L": "Крупная (>250)"}
EMPLOYMENT_LABELS = {"FT": "Полная занятость", "PT": "Частичная", "CT": "Контракт", "FL": "Фриланс"}

# Страны → регион. Группировка по уровню зарплат в IT, а не по географии:
# в датасете на большинство стран меньше 5 записей, по отдельности их не выучить.
US = {"US", "PR"}  # PR — Пуэрто-Рико, территория США
HIGH_INCOME = {"CA", "AU", "NZ", "CH", "DK", "NL", "DE", "AT", "BE", "LU", "IE", "IL",
               "GB", "SE", "NO", "FI", "JP", "SG", "AE", "FR"}
MID_INCOME = {"ES", "IT", "PT", "GR", "PL", "CZ", "SI", "EE", "HR", "MT", "CY", "HU",
              "RO", "BG", "RS", "RU", "TR", "CL", "MX", "BR", "AR", "CN", "HK", "MY", "JE"}
# всё остальное → "OTHER" (Индия, Пакистан, Нигерия, Вьетнам и т.д.)

REGION_LABELS = {
    "US": "США",
    "HIGH": "Богатые страны (Канада, Зап. Европа, UK, Япония, Австралия…)",
    "MID": "Средний уровень (Юж./Вост. Европа, Латам, Китай…)",
    "OTHER": "Остальные (Индия, Пакистан, Африка, Юго-Вост. Азия…)",
}


def country_region(code: str) -> str:
    if code in US:
        return "US"
    if code in HIGH_INCOME:
        return "HIGH"
    if code in MID_INCOME:
        return "MID"
    return "OTHER"


# Должность → семейство. Порядок правил важен: сначала более специфичные.
JOB_FAMILY_RULES = [
    ("Analyst",     ["analyst", "analytics"]),
    ("ML/AI",       ["machine learning", "ml ", "ml engineer", "ai ", "ai scientist",
                     "computer vision", "nlp", "3d computer vision"]),
    ("Research",    ["research"]),
    ("Data Eng",    ["data engineer", "big data", "etl", "cloud data", "data architect",
                     "big data architect", "engineering manager", "director of data engineering"]),
    ("Data Science", ["data scien", "data specialist", "head of data"]),
]
LEADERSHIP_WORDS = ["lead", "head", "director", "principal", "manager", "staff"]

JOB_FAMILY_LABELS = {
    "Analyst": "Аналитик данных / BI",
    "Data Eng": "Data Engineer / архитектор",
    "Data Science": "Data Scientist",
    "ML/AI": "ML / CV / NLP инженер",
    "Research": "Исследователь (Research)",
}


def job_family(title: str) -> str:
    t = title.lower() + " "
    # «Analytics Engineer» и «Data Analytics Engineer» — это скорее инженеры данных
    if "analytics engineer" in t:
        return "Data Eng"
    for family, words in JOB_FAMILY_RULES:
        if any(w in t for w in words):
            return family
    return "Data Science"


def is_leadership(title: str) -> int:
    t = title.lower()
    return int(any(w in t for w in LEADERSHIP_WORDS))


FEATURES_NUM = ["experience", "company_size_n", "remote_ratio", "work_year",
                "is_leadership", "is_full_time", "same_country"]
FEATURES_CAT = ["job_family", "residence_region", "company_region"]
FEATURES = FEATURES_NUM + FEATURES_CAT
TARGET = "salary_in_usd"


def load_raw(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляем полные дубли: при кросс-валидации одна и та же строка
    могла бы попасть и в обучение, и в проверку — это завышает качество."""
    return df.drop_duplicates().reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Из «сырых» колонок датасета (или из формы в приложении) делает признаки модели."""
    out = pd.DataFrame(index=df.index)
    out["experience"] = df["experience_level"].map(EXPERIENCE_ORDER)
    out["company_size_n"] = df["company_size"].map(COMPANY_SIZE_ORDER)
    out["remote_ratio"] = df["remote_ratio"].astype(int)
    out["work_year"] = df["work_year"].astype(int)
    out["is_leadership"] = df["job_title"].map(is_leadership)
    out["is_full_time"] = (df["employment_type"] == "FT").astype(int)
    out["same_country"] = (df["employee_residence"] == df["company_location"]).astype(int)
    out["job_family"] = df["job_title"].map(job_family)
    out["residence_region"] = df["employee_residence"].map(country_region)
    out["company_region"] = df["company_location"].map(country_region)
    return out[FEATURES]


def target(df: pd.DataFrame) -> np.ndarray:
    """Учим на логарифме зарплаты: распределение с длинным хвостом,
    а ошибка «в разах» для зарплат честнее, чем ошибка в долларах."""
    return np.log(df[TARGET].values)
