from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
import numpy as np
import pandas as pd
from csv_animation import plot_2d, plot_3d, plotit
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import joblib


def file_to_traintest(path_data_3d, path_data_2d):
    df_pos = pd.read_csv(path_data_3d)
    df_2d = pd.read_csv(path_data_2d)

    # exclude time column.
    df_pos_col_chose = df_pos.columns.difference(["time"])
    df_2d_col_chose = df_2d.columns.difference(["time"])

    # Input data without time columns.
    X = df_2d[df_2d_col_chose]
    y = df_pos[df_pos_col_chose]

    # Test-train split.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def linear_regression_tuning(
    path_data_3d="data/data_CMU_3d_01.csv", path_data_2d="data/data_CMU_2d_01.csv"
):
    start_time = time.time()
    X_train, X_test, y_train, y_test = file_to_traintest(path_data_3d, path_data_2d)

    model = LinearRegression()

    # Parameters to search.
    param_grid = {
        "fit_intercept": [True, False],
        "copy_X": [True, False],
        "n_jobs": [1, 5, 10, 15, None],
        "positive": [True, False],
    }

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        n_jobs=4,
    )
    grid_search.fit(X_train, y_train)

    print("\n\tLinear Regression")
    print(f"Best Parameters of Linear Regression: {grid_search.best_params_}")
    print(f"Best mean cross-validation R^2 Score: {grid_search.best_score_}")

    test_score = grid_search.score(X_test, y_test)
    print(f"Test R^2 score with best parameters: {test_score}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.4f} s")
    return


def ridge_search_alpha(
    path_data_3d="data/data_CMU_3d_01.csv", path_data_2d="data/data_CMU_2d_01.csv"
):
    start_time = time.time()
    X_train, X_test, y_train, y_test = file_to_traintest(path_data_3d, path_data_2d)

    # Parameters to search.
    param_grid = {"alpha": [1e-8, 1e-7, 1e-6, 1e-5, 0.1, 1.0]}

    # Apply regression and train.
    model = Ridge()

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        n_jobs=4,
    )
    grid_search.fit(X_train, y_train)

    print("\n\tRidge")
    print(f"Best Alpha of Ridge: {grid_search.best_params_}")
    print(f"Best mean cross-validation R^2 Score: {grid_search.best_score_}")

    test_score = grid_search.score(X_test, y_test)
    print(f"Test R^2 score with best parameters: {test_score}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.4f} s")
    return


def lasso_search_alpha(
    path_data_3d="data/data_CMU_3d_01.csv", path_data_2d="data/data_CMU_2d_01.csv"
):
    start_time = time.time()
    X_train, X_test, y_train, y_test = file_to_traintest(path_data_3d, path_data_2d)

    # Parameters to search.
    param_grid = {"alpha": [0.005, 0.01, 0.1, 1.0]}

    # Apply regression and train.
    model = Lasso()

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        n_jobs=4,
    )
    grid_search.fit(X_train, y_train)

    print("\n\tLasso")
    print(f"Best Alpha of Lasso: {grid_search.best_params_}")
    print(f"Best mean cross-validation R^2 Score: {grid_search.best_score_}")

    test_score = grid_search.score(X_test, y_test)
    print(f"Test R^2 score with best parameters: {test_score}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.4f} s")
    return


def KNN_tuning(
    path_data_3d="data/data_CMU_3d_01.csv", path_data_2d="data/data_CMU_2d_01.csv"
):
    start_time = time.time()
    X_train, X_test, y_train, y_test = file_to_traintest(path_data_3d, path_data_2d)

    model = KNeighborsRegressor()

    # Parameters to search.
    param_grid = {
        "n_neighbors": [2, 3],
        "weights": ["distance"],
        "metric": ["manhattan"],
    }
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=4,
    )
    grid_search.fit(X_train, y_train)

    print("\n\tKNN")
    print(f"Best Parameters of KNN: {grid_search.best_params_}")
    print(f"Best mean cross-validation MSE Score: {-grid_search.best_score_}")

    test_score = grid_search.score(X_test, y_test)
    print(f"Test MSE score with best parameters: {-test_score}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.4f} s")
    return


def rforest_tuning(
    path_data_3d="data/data_CMU_3d_01.csv", path_data_2d="data/data_CMU_2d_01.csv"
):
    start_time = time.time()
    X_train, X_test, y_train, y_test = file_to_traintest(path_data_3d, path_data_2d)

    model = RandomForestRegressor(random_state=42)

    # Parameters to search.
    param_grid = {
        "ccp_alpha": [1e-7],
        "max_depth": [10],
        "max_features": ["sqrt"],
        "min_impurity_decrease": [1e-7],
        "min_samples_split": [10],
        "n_estimators": [400],
    }
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=4,
    )
    grid_search.fit(X_train, y_train)

    print("\n\tRandom Forest")
    print(f"Best Parameters of Random Forest: {grid_search.best_params_}")
    print(f"Best mean cross-validation MSE Score: {-grid_search.best_score_}")

    test_score = grid_search.score(X_test, y_test)
    print(f"Test MSE score with best parameters: {-test_score}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.4f} s")
    return


def xgb_tuning(
    path_data_3d="data/data_CMU_3d_01.csv", path_data_2d="data/data_CMU_2d_01.csv"
):
    start_time = time.time()
    X_train, X_test, y_train, y_test = file_to_traintest(path_data_3d, path_data_2d)

    model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

    # Parameters to search.
    param_grid = {
        "colsample_bytree": [0.7],
        "gamma": [0],
        "learning_rate": [0.15],
        "max_depth": [7],
        "n_estimators": [300],
        "reg_lambda": [1],
        "reg_alpha": [0.01],
        "subsample": [0.7],
    }
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=4,
    )
    grid_search.fit(X_train, y_train)

    print("\n\tXGBoosting")
    print(f"Best Parameters of XGBoosting: {grid_search.best_params_}")
    print(f"Best mean cross-validation MSE Score: {-grid_search.best_score_}")

    test_score = grid_search.score(X_test, y_test)
    print(f"Test MSE score with best parameters: {-test_score}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.4f} s")
    return


def MLP_tuning(
    path_data_3d="data/data_CMU_3d_01.csv", path_data_2d="data/data_CMU_2d_01.csv"
):
    start_time = time.time()
    X_train, X_test, y_train, y_test = file_to_traintest(path_data_3d, path_data_2d)

    model = MLPRegressor(learning_rate_init=0.001, random_state=42)

    # Parameters to search.
    param_grid = {
        "hidden_layer_sizes": [
            (512, 256, 128),
        ],
        "alpha": [1e-6],
    }
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=4,
    )
    grid_search.fit(X_train, y_train)

    print("\n\tMLP")
    print(f"Best Parameters of MLP: {grid_search.best_params_}")
    print(f"Best mean cross-validation MSE Score: {-grid_search.best_score_}")

    test_score = grid_search.score(X_test, y_test)
    print(f"Test MSE score with best parameters: {-test_score}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.4f} s")
    return


def save_trained_model(
    path_data_3d="data/data_CMU_3d_01.csv", path_data_2d="data/data_CMU_2d_01.csv"
):

    X_train, X_test, y_train, y_test = file_to_traintest(path_data_3d, path_data_2d)

    start_time = time.time()
    print("\n Training Linear Regression Model...")
    model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, positive=False)
    model.fit(X_train, y_train)
    joblib.dump(model, "trained_model/linear_regression_model.joblib")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training Finished! Time taken: {elapsed_time:.4f} s")

    start_time = time.time()
    print("\n Training KNN Model...")
    model = KNeighborsRegressor(metric="manhattan", n_neighbors=2, weights="distance")
    model.fit(X_train, y_train)
    joblib.dump(model, "trained_model/KNN_model.joblib")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training Finished! Time taken: {elapsed_time:.4f} s")

    start_time = time.time()
    print("\n Training Random Forest Model...")
    model = RandomForestRegressor(
        random_state=42,
        ccp_alpha=1e-7,
        max_depth=20,
        max_features="sqrt",
        min_impurity_decrease=1e-7,
        min_samples_split=10,
        n_estimators=500,
    )
    model.fit(X_train, y_train)
    joblib.dump(model, "trained_model/RF_model.joblib")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training Finished! Time taken: {elapsed_time:.4f} s")

    start_time = time.time()
    print("\n Training XGB Model...")
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        colsample_bytree=0.7,
        gamma=0,
        learning_rate=0.15,
        max_depth=7,
        n_estimators=300,
        reg_lambda=1,
        reg_alpha=0.01,
        subsample=0.7,
    )
    model.fit(X_train, y_train)
    joblib.dump(model, "trained_model/XGB_model.joblib")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training Finished! Time taken: {elapsed_time:.4f} s")

    start_time = time.time()
    print("\n Training MLP Model...")
    model = MLPRegressor(
        learning_rate_init=0.001,
        random_state=42,
        alpha=1e-06,
        hidden_layer_sizes=(512, 256, 128),
    )
    model.fit(X_train, y_train)
    joblib.dump(model, "trained_model/MLP_model.joblib")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training Finished! Time taken: {elapsed_time:.4f} s")

    start_time = time.time()
    print("\n Training MLP Model 2...")
    model = MLPRegressor(
        learning_rate_init=0.001,
        random_state=42,
        alpha=1e-06,
        hidden_layer_sizes=(256, 128, 64),
    )
    model.fit(X_train, y_train)
    joblib.dump(model, "trained_model/MLP_model_2.joblib")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training Finished! Time taken: {elapsed_time:.4f} s")
    return


def test_trained_model(
    path_data_3d="data/data_CMU_3d_01.csv", path_data_2d="data/data_CMU_2d_01.csv"
):

    X_train, X_test, y_train, y_test = file_to_traintest(path_data_3d, path_data_2d)
    start_time = time.time()
    print("\n Testing Linear Regression Model...")
    model = joblib.load("trained_model/linear_regression_model.joblib")
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"R-squared: {r2}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Testing Finished! Time taken: {elapsed_time:.4f} s")

    start_time = time.time()
    print("\n Testing KNN Model...")
    model = joblib.load("trained_model/KNN_model.joblib")
    y_pred = model.predict(X_test)
    MSE = mean_squared_error(y_test, y_pred)
    print(f"MSE: {MSE}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Testing Finished! Time taken: {elapsed_time:.4f} s")

    start_time = time.time()
    print("\n Testing Random Forest Model...")
    model = joblib.load("trained_model/RF_model.joblib")
    y_pred = model.predict(X_test)
    MSE = mean_squared_error(y_test, y_pred)
    print(f"MSE: {MSE}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Testing Finished! Time taken: {elapsed_time:.4f} s")

    start_time = time.time()
    print("\n Testing XGB Model...")
    model = joblib.load("trained_model/XGB_model.joblib")
    y_pred = model.predict(X_test)
    MSE = mean_squared_error(y_test, y_pred)
    print(f"MSE: {MSE}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Testing Finished! Time taken: {elapsed_time:.4f} s")

    start_time = time.time()
    print("\n Testing MLP Model...")
    model = joblib.load("trained_model/MLP_model.joblib")
    y_pred = model.predict(X_test)
    MSE = mean_squared_error(y_test, y_pred)
    print(f"MSE: {MSE}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Testing Finished! Time taken: {elapsed_time:.4f} s")

    start_time = time.time()
    print("\n Testing MLP Model...")
    model = joblib.load("trained_model/MLP_model_2.joblib")
    y_pred = model.predict(X_test)
    MSE = mean_squared_error(y_test, y_pred)
    print(f"MSE: {MSE}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Testing Finished! Time taken: {elapsed_time:.4f} s")
    return


if __name__ == "__main__":
    input1, input2 = "data/data_CMU_3d_01.csv", "data/data_CMU_2d_01.csv"

    print("\nHyperparameter Tuning:")
    linear_regression_tuning(input1, input2)
    ridge_search_alpha(input1, input2)
    lasso_search_alpha(input1, input2)
    KNN_tuning(input1, input2)
    rforest_tuning(input1, input2)
    xgb_tuning(input1, input2)
    MLP_tuning(input1, input2)
    print("\nSave model trained with hyperparameters found:")
    save_trained_model(input1, input2)
    print("\nTest the time it takes to run the trained models:")

    test_trained_model(input1, input2)
