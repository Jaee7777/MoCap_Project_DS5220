from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import time
import joblib
from regression import file_to_traintest
from generate_2d_data import read_large_csv


def large_test_train(
    file_2d="data_CMU_2d.csv", file_3d="data_CMU_3d.csv", chunksize=50000
):
    # Read data.
    df_2d = read_large_csv(file=file_2d, chunksize=chunksize)
    df_3d = read_large_csv(file=file_3d, chunksize=chunksize)

    # Exclude time column.
    df_2d_col_chose = df_2d.columns.difference(["time"])
    df_3d_col_chose = df_3d.columns.difference(["time"])

    # Input data without time columns.
    X = df_2d[df_2d_col_chose]
    y = df_3d[df_3d_col_chose]

    # Test-train split.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train_MLP_model_chunk(file_2d, file_3d, chunksize=50000):
    start_time = time.time()
    print("\n Reading Data by chunks...")
    X_train, X_test, y_train, y_test = large_test_train(
        file_2d=file_2d, file_3d=file_3d, chunksize=chunksize
    )

    print("\n Training MLP Model-All by chunks...")
    model = MLPRegressor(
        learning_rate_init=0.001,
        random_state=42,
        alpha=1e-06,
        hidden_layer_sizes=(512, 256, 128),
    )
    print("Training on chunk 1...")
    model.fit(X_train[:chunksize], y_train[:chunksize])

    # Enable warm start.
    model.set_params(warm_start=True)

    # Train by chunks.
    n_chunks = len(X_train) // chunksize + 1

    for i in range(1, n_chunks):
        # Slice the chunk.
        i_head = i * chunksize
        i_tail = min((i + 1) * chunksize, len(X_train))

        if i_head >= len(X_train):
            break

        print(f"Training on chunk {i+1}/{n_chunks}...")

        # Separate the chunk.
        chunk_X = X_train[i_head:i_tail]
        chunk_y = y_train[i_head:i_tail]

        # Train the chunk.
        model.fit(chunk_X, chunk_y)

    # Save model.
    joblib.dump(model, "trained_model/MLP_model_all.joblib")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.4f} s")

    # Test model.
    start_time = time.time()
    print("Testing model...")
    y_pred = model.predict(X_test)
    MSE = mean_squared_error(y_test, y_pred)
    print(f"MSE: {MSE}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.4f} s")
    return


def train_XGB_model(file_2d, file_3d, chunksize=50000):
    start_time = time.time()
    print("\n Reading Data by chunks...")
    X_train, X_test, y_train, y_test = large_test_train(
        file_2d=file_2d, file_3d=file_3d, chunksize=chunksize
    )

    print("\n Training XGB Model-All...")
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

    # Save model.
    joblib.dump(model, "trained_model/XGB_model_all.joblib")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.4f} s")

    # Test model.
    start_time = time.time()
    print("Testing model...")
    y_pred = model.predict(X_test)
    MSE = mean_squared_error(y_test, y_pred)
    print(f"MSE: {MSE}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.4f} s")
    return


def train_rf_model(file_2d, file_3d, chunksize=50000):
    start_time = time.time()
    print("\n Reading Data by chunks...")
    X_train, X_test, y_train, y_test = large_test_train(
        file_2d=file_2d, file_3d=file_3d, chunksize=chunksize
    )

    print("\n Training Random Forest Model-All...")
    model = RandomForestRegressor(
        random_state=42,
        ccp_alpha=1e-7,
        max_depth=10,
        max_features="sqrt",
        min_impurity_decrease=1e-7,
        min_samples_split=10,
        n_estimators=400,
    )
    model.fit(X_train, y_train)

    # Save model.
    joblib.dump(model, "trained_model/RF_model_all.joblib")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.4f} s")

    # Test model.
    start_time = time.time()
    print("Testing model...")
    y_pred = model.predict(X_test)
    MSE = mean_squared_error(y_test, y_pred)
    print(f"MSE: {MSE}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.4f} s")
    return


def train_rf_model_2(file_2d, file_3d, chunksize=50000):
    start_time = time.time()
    print("\n Reading Data by chunks...")
    X_train, X_test, y_train, y_test = large_test_train(
        file_2d=file_2d, file_3d=file_3d, chunksize=chunksize
    )

    print("\n Training Random Forest Model-All by chunks...")
    model = RandomForestRegressor(
        random_state=42,
        ccp_alpha=1e-7,
        max_depth=10,
        max_features="sqrt",
        min_impurity_decrease=1e-7,
        min_samples_split=10,
        n_estimators=100,
    )
    print("Training on chunk 1...")
    model.fit(X_train[:chunksize], y_train[:chunksize])

    # Enable warm start.
    model.set_params(warm_start=True)

    # Train by chunks.
    n_chunks = len(X_train) // chunksize + 1

    for i in range(1, n_chunks):
        # Slice the chunk.
        i_head = i * chunksize
        i_tail = min((i + 1) * chunksize, len(X_train))

        if i_head >= len(X_train):
            break

        print(f"Training on chunk {i+1}/{n_chunks}...")

        # Separate the chunk.
        chunk_X = X_train[i_head:i_tail]
        chunk_y = y_train[i_head:i_tail]

        # Train the chunk.
        model.fit(chunk_X, chunk_y)

    # Save model.
    joblib.dump(model, "trained_model/RF_model_all_2.joblib")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.4f} s")

    # Test model.
    start_time = time.time()
    print("Testing model...")
    y_pred = model.predict(X_test)
    MSE = mean_squared_error(y_test, y_pred)
    print(f"MSE: {MSE}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.4f} s")
    return


def train_knn_model(file_2d, file_3d, chunksize=50000):
    start_time = time.time()
    print("\n Reading Data by chunks...")
    X_train, X_test, y_train, y_test = large_test_train(
        file_2d=file_2d, file_3d=file_3d, chunksize=chunksize
    )

    print("\n Training KNN Model-All...")
    model = KNeighborsRegressor(metric="manhattan", n_neighbors=2, weights="distance")
    model.fit(X_train, y_train)

    # Save model.
    joblib.dump(model, "trained_model/KNN_model_all.joblib")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.4f} s")

    # Test model.
    start_time = time.time()
    print("Testing model...")
    y_pred = model.predict(X_test)
    MSE = mean_squared_error(y_test, y_pred)
    print(f"MSE: {MSE}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.4f} s")
    return


def test_small_dataset(
    path_data_3d="data/data_CMU_3d_01.csv",
    path_data_2d="data/data_CMU_2d_01.csv",
    model_joblib="trained_model/MLP_model_all.joblib",
):

    X_train, X_test, y_train, y_test = file_to_traintest(path_data_3d, path_data_2d)
    start_time = time.time()
    print("\n Testing Model-All on small dataset...")
    model = joblib.load(model_joblib)
    y_pred = model.predict(X_test)
    MSE = mean_squared_error(y_test, y_pred)
    print(f"MSE: {MSE}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Testing Finished! Time taken: {elapsed_time:.4f} s")
    return


if __name__ == "__main__":
    train_MLP_model_chunk(
        file_2d="data/data_CMU_2d.csv", file_3d="data/data_CMU_3d.csv", chunksize=50000
    )
    test_small_dataset(
        path_data_3d="data/data_CMU_3d_01.csv",
        path_data_2d="data/data_CMU_2d_01.csv",
        model_joblib="trained_model/MLP_model_all.joblib",
    )
    train_XGB_model(
        file_2d="data/data_CMU_2d.csv", file_3d="data/data_CMU_3d.csv", chunksize=50000
    )
    test_small_dataset(
        path_data_3d="data/data_CMU_3d_01.csv",
        path_data_2d="data/data_CMU_2d_01.csv",
        model_joblib="trained_model/XGB_model_all.joblib",
    )
    train_rf_model_2(
        file_2d="data/data_CMU_2d.csv", file_3d="data/data_CMU_3d.csv", chunksize=50000
    )
    test_small_dataset(
        path_data_3d="data/data_CMU_3d_01.csv",
        path_data_2d="data/data_CMU_2d_01.csv",
        model_joblib="trained_model/RF_model_all_2.joblib",
    )
    train_knn_model(
        file_2d="data/data_CMU_2d.csv", file_3d="data/data_CMU_3d.csv", chunksize=50000
    )
    test_small_dataset(
        path_data_3d="data/data_CMU_3d_01.csv",
        path_data_2d="data/data_CMU_2d_01.csv",
        model_joblib="trained_model/KNN_model_all.joblib",
    )
