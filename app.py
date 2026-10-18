"""
app.py — веб-приложение: вводишь параметры, получаешь прогноз зарплаты.

Запуск из корня проекта:
    streamlit run app.py
Откроется браузер на http://localhost:8501
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import features as F  # noqa: E402

MODEL_PATH = ROOT / "models" / "salary_model.joblib"
META_PATH = ROOT / "models" / "metadata.json"
DATA_PATH = ROOT / "data" / "ds_salaries.csv"

st.set_page_config(page_title="Прогноз зарплаты в Data/ML", page_icon="💼", layout="centered")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        import train
        train.main()
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return model, meta


@st.cache_data
def load_data():
    return F.clean(F.load_raw(DATA_PATH))


model, meta = load_model()
df = load_data()

# Список стран для выпадающих списков: код + сколько записей в датасете
country_counts = pd.concat([df.employee_residence, df.company_location]).value_counts()
countries = list(country_counts.index)
country_fmt = lambda c: f"{c} — {F.REGION_LABELS[F.country_region(c)].split(' (')[0]} ({country_counts[c]} зап.)"

st.title("💼 Прогноз зарплаты в Data / ML")
st.caption(f"Модель: {meta['model']}, обучена на {meta['n_rows']} анкетах 2020–2022 гг. "
           f"Средняя ошибка на тесте ≈ ${meta['test']['MAE, $']:,.0f}.")

with st.form("params"):
    c1, c2 = st.columns(2)
    with c1:
        job_title = st.selectbox("Должность", sorted(df.job_title.unique()),
                                 index=sorted(df.job_title.unique()).index("Data Scientist"))
        experience = st.selectbox("Уровень опыта", list(F.EXPERIENCE_LABELS),
                                  format_func=F.EXPERIENCE_LABELS.get, index=1)
        employment = st.selectbox("Тип занятости", list(F.EMPLOYMENT_LABELS),
                                  format_func=F.EMPLOYMENT_LABELS.get)
        company_size = st.selectbox("Размер компании", list(F.COMPANY_SIZE_LABELS),
                                    format_func=F.COMPANY_SIZE_LABELS.get, index=1)
    with c2:
        residence = st.selectbox("Страна проживания", countries, format_func=country_fmt,
                                 index=countries.index("US"))
        location = st.selectbox("Страна компании", countries, format_func=country_fmt,
                                index=countries.index("US"))
        remote = st.select_slider("Удалёнка, %", options=[0, 50, 100], value=100)
        year = st.selectbox("Год", [2022, 2021, 2020])
    submitted = st.form_submit_button("Рассчитать", type="primary", width="stretch")

row = pd.DataFrame([{
    "work_year": year, "experience_level": experience, "employment_type": employment,
    "job_title": job_title, "employee_residence": residence, "remote_ratio": remote,
    "company_location": location, "company_size": company_size,
}])
X = F.build_features(row)
pred = float(np.exp(model.predict(X))[0])
lo, hi = pred * meta["interval_80"]["low_mult"], pred * meta["interval_80"]["high_mult"]

st.divider()
m1, m2, m3 = st.columns(3)
m1.metric("Прогноз, $ в год", f"{pred:,.0f}")
m2.metric("В месяц, $", f"{pred / 12:,.0f}")
m3.metric("80%-интервал, $", f"{lo / 1000:,.0f}k – {hi / 1000:,.0f}k")
st.caption("80%-интервал: в 8 случаях из 10 реальная зарплата людей с такими параметрами попадает в этот диапазон.")

with st.expander("Как модель видит ваши параметры"):
    st.write({
        "семейство должности": F.JOB_FAMILY_LABELS[X.job_family.iloc[0]],
        "лидерская роль": "да" if X.is_leadership.iloc[0] else "нет",
        "регион проживания": F.REGION_LABELS[X.residence_region.iloc[0]],
        "регион компании": F.REGION_LABELS[X.company_region.iloc[0]],
    })

# Похожие люди из датасета: тот же регион, опыт и семейство должностей
feats = F.build_features(df)
mask = ((feats.residence_region == X.residence_region.iloc[0])
        & (feats.experience == X.experience.iloc[0])
        & (feats.job_family == X.job_family.iloc[0]))
similar = df[mask]
st.subheader(f"Похожие анкеты из датасета: {len(similar)}")
if len(similar):
    st.write(f"Медиана у них: **${similar.salary_in_usd.median():,.0f}**")
    show = (similar[["work_year", "job_title", "employee_residence", "company_size", "remote_ratio", "salary_in_usd"]]
            .sort_values("salary_in_usd", ascending=False).head(15)
            .rename(columns={"work_year": "год", "job_title": "должность", "employee_residence": "страна",
                             "company_size": "компания", "remote_ratio": "удалёнка %", "salary_in_usd": "зп, $"}))
    st.dataframe(show,
                 hide_index=True, width="stretch")
else:
    st.info("Таких анкет в датасете нет — прогноз опирается на общие закономерности, относитесь к нему осторожно.")

if X.residence_region.iloc[0] in ("MID", "OTHER"):
    st.warning("По этому региону в датасете мало данных: ошибка модели здесь выше (медианно ~35–40%).")
