from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, KFold
import joblib


def prepare_features(x, y, extra_vars=None):
    x = x.copy()
    if extra_vars:
        for var in extra_vars:
            if var in y.columns:
                x[var] = y[var]
            else:
                raise ValueError(f"Variável '{var}' não encontrada em y.")

    return x


def get_preprocessor(x):
    categorical_cols = x.select_dtypes(include=["object"]).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols)
        ],
        remainder="passthrough"
    )

    return preprocessor


def test_prediction(x, y, model, prediction_key="G3", test_size=0.2, random_state=42, model_name="Linear Regression"):
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y[prediction_key],
        test_size=test_size,
        random_state=random_state
    )

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"[{model_name}] R²: {r2:.2f}")
    print(f"[{model_name}] RMSE: {rmse:.2f}")


def get_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_model(model, x, y, metric_name="RMSE"):
    rkf = RepeatedKFold(n_splits=5, n_repeats=30, random_state=42)

    rmse_scorer = make_scorer(get_rmse, greater_is_better=False)
    rmse_scores_neg = cross_val_score(model, x, y, scoring=rmse_scorer, cv=rkf, n_jobs=-1)
    rmse_scores = -rmse_scores_neg
    r2_scores = cross_val_score(model, x, y, scoring="r2", cv=rkf, n_jobs=-1)

    print(f"{metric_name} médio: {np.mean(rmse_scores):.2f} ± {np.std(rmse_scores):.2f}")
    print(f"R² médio: {np.mean(r2_scores):.2f} ± {np.std(r2_scores):.2f}")

    return rmse_scores, r2_scores


def repeated_hyperparam_search(
        x, y,
        n_repeats=30, n_iter=20,
        model=None, param_dist=None,
        model_name="Linear Regression"
):
    scores = []
    best_score_overall = -float('inf')
    best_params_overall = None
    best_iteration = None
    best_model_overall = None
    all_results = []

    os.makedirs("results", exist_ok=True)

    print(f"\n{model_name}")
    for i in range(n_repeats):
        print(f"Repetição {i + 1} de {n_repeats}")
        cv = KFold(n_splits=5, shuffle=True, random_state=i)
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring="r2",
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )
        search.fit(x, y)
        score = search.best_score_
        params = search.best_params_

        scores.append(score)
        all_results.append({
            "Repetição": i + 1,
            "R²": round(score, 4),
            **params
        })

        if score > best_score_overall:
            best_score_overall = score
            best_params_overall = params
            best_iteration = i + 1
            best_model_overall = search.best_estimator_

    r2_mean = np.mean(scores)
    r2_std = np.std(scores)

    print(f"\nR² médio nas {n_repeats} repetições: {r2_mean:.4f}")
    print(f"Desvio padrão: {r2_std:.4f}")
    print(f"Melhor R² encontrado: {best_score_overall:.4f} (na repetição {best_iteration})")
    print(f"Melhores hiperparâmetros: {best_params_overall}")

    df_all = pd.DataFrame(all_results)

    df_all.loc[len(df_all.index)] = {
        "Repetição": "Média",
        "R²": round(r2_mean, 4),
        **{k: best_params_overall.get(k, "") for k in best_params_overall}
    }

    df_all.loc[len(df_all.index)] = {
        "Repetição": "Desvio Padrão",
        "R²": round(r2_std, 4),
        **{k: "" for k in best_params_overall}
    }

    r2_best = round(best_score_overall, 4)

    best_section = pd.DataFrame([{
        "Repetição": f"Melhor ({best_iteration})",
        "R²": r2_best,
        **best_params_overall
    }])

    with pd.ExcelWriter(f"results/{model_name}.xlsx", engine="openpyxl", mode="w") as writer:
        df_all.to_excel(writer, sheet_name="Resultados", index=False, startrow=0)
        best_section.to_excel(writer, sheet_name="Resultados", index=False, startrow=len(df_all) + 3)

    joblib.dump(best_model_overall, f"models/{model_name}_best_model.joblib")
    print(f"Melhor modelo salvo em 'models/{model_name}_best_model.joblib'")

    return r2_mean, r2_std, r2_best


def run_polynomial_search(x, y, degrees=None, n_repeats=5):
    from models import polynomial_regression_model

    if degrees is None:
        degrees = [2, 3, 4]

    rkf = RepeatedKFold(n_splits=5, n_repeats=n_repeats, random_state=42)
    best_score = float("inf")
    best_model = None

    for degree in degrees:
        model = polynomial_regression_model(x, degree=degree)
        scores = cross_val_score(
            model, x, y,
            scoring=make_scorer(mean_squared_error, greater_is_better=False),
            cv=rkf, n_jobs=-1
        )

        scores_reshaped = scores.reshape(n_repeats, 5)
        mean_per_repeat = np.mean(scores_reshaped, axis=1)
        rmse_per_repeat = np.sqrt(-mean_per_repeat)

        rmse_mean = rmse_per_repeat.mean()
        rmse_std = rmse_per_repeat.std()

        r2_scores = cross_val_score(model, x, y, scoring="r2", cv=rkf, n_jobs=-1)

        print(f"Degree {degree} → RMSE médio das {n_repeats} repetições: {rmse_mean:.2f} ± {rmse_std:.2f}")
        print(f"R² médio: {np.mean(r2_scores):.2f} ± {np.std(r2_scores):.2f}")

        if rmse_mean < best_score:
            best_score = rmse_mean
            best_model = model

    return best_model
