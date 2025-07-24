from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from tools import get_preprocessor


def linear_regression_model(x):
    model = Pipeline(steps=[
        ("preprocessing", get_preprocessor(x)),
        ("regressor", LinearRegression())
    ])

    return model


def random_forest_model(x):
    model = Pipeline(steps=[
        ("preprocessing", get_preprocessor(x)),
        ("regressor", RandomForestRegressor())
    ])

    return model


def decision_tree_model(x):
    model = Pipeline(steps=[
        ("preprocessing", get_preprocessor(x)),
        ("regressor", DecisionTreeRegressor())
    ])

    return model


def svr_model(x):
    model = Pipeline(steps=[
        ("preprocessing", get_preprocessor(x)),
        ("regressor", SVR())
    ])

    return model


def elastic_net_model(x):
    model = Pipeline(steps=[
        ("preprocessing", get_preprocessor(x)),
        ("regressor", ElasticNet())
    ])

    return model


def lasso_regression_model(x):
    model = Pipeline(steps=[
        ("preprocessing", get_preprocessor(x)),
        ("regressor", Lasso())
    ])

    return model


def ridge_regression_model(x):
    model = Pipeline(steps=[
        ("preprocessing", get_preprocessor(x)),
        ("regressor", Ridge())
    ])

    return model


def polynomial_regression_model(x, degree=2):
    model = Pipeline(steps=[
        ("preprocessing", get_preprocessor(x)),
        ("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),
        ("regressor", LinearRegression())
    ])
    return model