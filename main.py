from ucimlrepo import fetch_ucirepo
from scipy.stats import randint, loguniform, uniform
from models import linear_regression_model, random_forest_model, decision_tree_model, svr_model, elastic_net_model, \
    lasso_regression_model, ridge_regression_model
from tools import prepare_features, evaluate_model, repeated_hyperparam_search, run_polynomial_search

student_performance = fetch_ucirepo(id=320)
x = student_performance.data.features
y = student_performance.data.targets
x = prepare_features(x, y, extra_vars=["G1", "G2"])

prediction_key = "G3"

# print(y)

# _linear_regression = linear_regression_model(x)

# evaluate_model(_linear_regression, x, y, metric_name="RMSE")
# evaluate_model(_random_forest, x, y, metric_name="RMSE")

# repeated_hyperparam_search(
#     model_name="Linear Regression",
#     x=x,
#     y=y,
#     n_repeats=5,
#     param_dist={
#         "regressor__alpha": loguniform(1e-3, 100)  # Busca em escala log
#     }
# )

# repeated_hyperparam_search(
#     model_name="Random Forest",
#     x=x,
#     y=y,
#     n_repeats=5,
#     model=random_forest_model(x),
#     param_dist={
#         "regressor__n_estimators": randint(50, 200),
#         "regressor__max_depth": [5, 10, 15, None],
#         "regressor__min_samples_split": randint(2, 10),
#         "regressor__min_samples_leaf": randint(1, 5),
#         "regressor__max_features": ["sqrt", "log2", None]
#     },
# )

# repeated_hyperparam_search(
#     model_name="Decision Tree",
#     x=x,
#     y=y,
#     n_repeats=5,
#     model=decision_tree_model(x),  # sua função definida anteriormente
#     param_dist={
#         "regressor__max_depth": [5, 10, 15, None],
#         "regressor__min_samples_split": randint(2, 10),
#         "regressor__min_samples_leaf": randint(1, 5),
#         "regressor__max_features": ["sqrt", "log2", None]
#     },
# )

# repeated_hyperparam_search(
#     model_name="SVR",
#     x=x,
#     y=y[prediction_key],
#     n_repeats=5,
#     model=svr_model(x),
#     param_dist={
#         "regressor__C": uniform(0.1, 10),
#         "regressor__epsilon": uniform(0.01, 1),
#         "regressor__kernel": ["linear", "rbf", "poly"],
#         "regressor__degree": [2, 3, 4],
#         "regressor__gamma": ["scale", "auto"]
#     },
# )

# repeated_hyperparam_search(
#     model_name="Elastic Net",
#     x=x,
#     y=y,
#     n_repeats=5,
#     model=elastic_net_model(x),
#     param_dist={
#         "regressor__alpha": loguniform(1e-4, 1e1),
#         "regressor__l1_ratio": uniform(0, 1)
#     },
# )


# repeated_hyperparam_search(
#     model_name="Lasso Regression",
#     x=x,
#     y=y,
#     n_repeats=5,
#     model=lasso_regression_model(x),
#     param_dist={
#         "regressor__alpha": loguniform(1e-s
#     },
# )

# repeated_hyperparam_search(
#     model_name="Ridge Regression",
#     x=x,
#     y=y,
#     n_repeats=5,
#     model=ridge_regression_model(x),
#     param_dist={
#         "regressor__alpha": loguniform(1e-4, 1e1)
#     },
# )

run_polynomial_search(
    x=x,
    y=y,
    n_repeats=5,
    degrees=[2, 3, 4]
)

