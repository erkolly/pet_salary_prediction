"""
train.py — сравнивает модели, обучает лучшую и сохраняет её в models/.

Запуск из корня проекта:  python src/train.py

Схема честной оценки:
  1. Делим данные 80/20. Тестовые 20% НЕ трогаем до самого конца.
  2. На 80% сравниваем модели повторной кросс-валидацией (5 фолдов × 3 повтора):
     датасет маленький (565 строк), одно разбиение даёт слишком «шумную» оценку.
  3. Лучшую модель обучаем на всех 80% и один раз меряем на тесте.
  4. Финальную модель для приложения обучаем на всех 100% данных.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_val_predict, train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import features as F  # noqa: E402

SEED = 42
DATA = ROOT / "data" / "ds_salaries.csv"
MODELS_DIR = ROOT / "models"


def make_preprocessor(scale: bool) -> ColumnTransformer:
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("num", num, F.FEATURES_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), F.FEATURES_CAT),
    ])


def make_models() -> dict:
    return {
        "Baseline (медиана)": Pipeline([("prep", make_preprocessor(False)),
                                        ("model", DummyRegressor(strategy="median"))]),
        "Ridge (линейная)": Pipeline([("prep", make_preprocessor(True)),
                                      ("model", Ridge(alpha=1.0))]),
        "Random Forest": Pipeline([("prep", make_preprocessor(False)),
                                   ("model", RandomForestRegressor(n_estimators=400, min_samples_leaf=3,
                                                                   max_features=0.6, random_state=SEED,
                                                                   n_jobs=-1))]),
        "Gradient Boosting": Pipeline([("prep", make_preprocessor(False)),
                                       ("model", GradientBoostingRegressor(n_estimators=300, learning_rate=0.03,
                                                                           max_depth=3, subsample=0.8,
                                                                           min_samples_leaf=5,
                                                                           random_state=SEED))]),
    }


def dollar_metrics(y_log_true, y_log_pred) -> dict:
    """Метрики в долларах: модель учится на логарифме, обратно переводим через exp."""
    yt, yp = np.exp(y_log_true), np.exp(y_log_pred)
    return {
        "MAE, $": mean_absolute_error(yt, yp),
        "MedAE, $": median_absolute_error(yt, yp),
        "R² (лог)": r2_score(y_log_true, y_log_pred),
        "R² ($)": r2_score(yt, yp),
        "в пределах ±30%": float(np.mean(np.abs(yp / yt - 1) <= 0.30)),
    }


def compare_models(X, y, n_repeats=3) -> pd.DataFrame:
    rkf = RepeatedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    rows = []
    for name, pipe in make_models().items():
        scores = []
        for tr, va in rkf.split(X):
            pipe.fit(X.iloc[tr], y[tr])
            scores.append(dollar_metrics(y[va], pipe.predict(X.iloc[va])))
        s = pd.DataFrame(scores)
        row = {"модель": name}
        for col in s.columns:
            row[col] = s[col].mean()
        row["MAE std, $"] = s["MAE, $"].std()
        rows.append(row)
    return pd.DataFrame(rows).set_index("модель").sort_values("MAE, $")


def main():
    df = F.clean(F.load_raw(DATA))
    X, y = F.build_features(df), F.target(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    print(f"Строк после удаления дублей: {len(df)} (train {len(X_tr)}, test {len(X_te)})\n")

    print("Кросс-валидация на train (5 фолдов × 3 повтора):")
    cv = compare_models(X_tr, y_tr)
    with pd.option_context("display.float_format", "{:,.3f}".format, "display.width", 140):
        print(cv, "\n")

    best_name = cv.index[0]
    best = make_models()[best_name]
    best.fit(X_tr, y_tr)
    test = dollar_metrics(y_te, best.predict(X_te))
    print(f"Лучшая модель: {best_name}. Результат на отложенном тесте:")
    for k, v in test.items():
        print(f"   {k:<16} {v:,.3f}")

    # Интервал прогноза: берём ошибки модели на out-of-fold предсказаниях
    # (каждая строка предсказана моделью, которая её не видела) и смотрим их разброс.
    oof = cross_val_predict(make_models()[best_name], X, y,
                            cv=KFold(5, shuffle=True, random_state=SEED))
    resid = y - oof
    q_lo, q_hi = np.quantile(resid, [0.10, 0.90])
    coverage = float(np.mean((resid >= q_lo) & (resid <= q_hi)))
    print(f"\nИнтервал 80%: прогноз × [{np.exp(q_lo):.2f}; {np.exp(q_hi):.2f}]")

    final = make_models()[best_name]
    final.fit(X, y)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(final, MODELS_DIR / "salary_model.joblib")
    meta = {
        "model": best_name,
        "n_rows": len(df),
        "cv": cv.reset_index().to_dict(orient="records"),
        "test": test,
        "interval_80": {"low_mult": float(np.exp(q_lo)), "high_mult": float(np.exp(q_hi)),
                        "coverage_oof": coverage},
        "work_year_max": int(df["work_year"].max()),
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Модель сохранена: {MODELS_DIR / 'salary_model.joblib'}")


if __name__ == "__main__":
    main()
