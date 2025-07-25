import pandas as pd
from ucimlrepo import fetch_ucirepo
from scipy.stats import randint, uniform, loguniform
from models import random_forest_model, decision_tree_model, svr_model, elastic_net_model, \
    lasso_regression_model, ridge_regression_model
from tools import prepare_features, repeated_hyperparam_search

student_performance = fetch_ucirepo(id=320)
x = student_performance.data.features
y = student_performance.data.targets
x = prepare_features(x, y, extra_vars=["G1", "G2"])

prediction_key = "G3"

models_tests = [
    {
        "model_name": "Random Forest",
        "model": random_forest_model(x),
        "param_dist": {
            "regressor__n_estimators": randint(50, 200),
            "regressor__max_depth": [5, 10, 15, None],
            "regressor__min_samples_split": randint(2, 10),
            "regressor__min_samples_leaf": randint(1, 5),
            "regressor__max_features": ["sqrt", "log2", None]
        },
        "target": y
    },
    {
        "model_name": "Decision Tree",
        "model": decision_tree_model(x),
        "param_dist": {
            "regressor__max_depth": [5, 10, 15, None],
            "regressor__min_samples_split": randint(2, 10),
            "regressor__min_samples_leaf": randint(1, 5),
            "regressor__max_features": ["sqrt", "log2", None]
        },
        "target": y
    },
    {
        "model_name": "SVR",
        "model": svr_model(x),
        "param_dist": {
            "regressor__C": uniform(0.1, 10),
            "regressor__epsilon": uniform(0.01, 1),
            "regressor__kernel": ["linear", "rbf", "poly"],
            "regressor__degree": [2, 3, 4],
            "regressor__gamma": ["scale", "auto"]
        },
        "target": y[prediction_key]
    },
    {
        "model_name": "Elastic Net",
        "model": elastic_net_model(x),
        "param_dist": {
            "regressor__alpha": loguniform(1e-4, 1e1),
            "regressor__l1_ratio": uniform(0, 1)
        },
        "target": y
    },
    {
        "model_name": "Lasso Regression",
        "model": lasso_regression_model(x),
        "param_dist": {
            "regressor__alpha": loguniform(1e-5, 1e1)
        },
        "target": y
    },
    {
        "model_name": "Ridge Regression",
        "model": ridge_regression_model(x),
        "param_dist": {
            "regressor__alpha": loguniform(1e-4, 1e1)
        },
        "target": y
    }
]

summary_data = []

for test in models_tests:
    print(f"\nExecutando: {test['model_name']}")
    r2_mean, r2_std, r2_best = repeated_hyperparam_search(
        model_name=test["model_name"],
        x=x,
        y=test["target"],
        n_repeats=30,
        model=test["model"],
        param_dist=test["param_dist"]
    )

    row = {
        "Modelo": test["model_name"],
        "R² médio": round(r2_mean, 4),
        "Desvio padrão": round(r2_std, 4),
        "Melhor R²": round(r2_best, 4)
    }

    summary_data.append(row)

# Gera planilha resumo
df_summary = pd.DataFrame(summary_data)
df_summary.to_excel("results/resumo_geral_modelos.xlsx", index=False)
