from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from csv_animation import read_CMU_data, plot_2d, plot_3d, CMU_to_MPipe
import matplotlib.pyplot as plt
import matplotlib.animation as animation

if __name__ == "__main__":
    file_pos = "data/01_01_pos.csv"
    file_rot = "data/01_01_rot.csv"
    df, df_pos, df_rot, joints_number, joints_name = read_CMU_data(
        file_pos=file_pos, file_rot=file_rot
    )

    df_2d = pd.read_csv("data/01_01_2d.csv")

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

    # Apply regression and train.
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    df_pred = pd.DataFrame(y_pred, columns=y_test.columns)

    r2 = r2_score(y_test, y_pred)
    print(f"R-squared score: {r2}")

    joints_2d = []
    for i in range(int(len(df_2d.columns) / 2)):
        joints_2d.append(df_2d.columns[2 * i][:-2])

    joints_3d = []
    for i in range(int(len(y.columns) / 3)):
        joints_3d.append(y.columns[3 * i][:-2])

    # 3D plot.
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection="3d")
    plot_3d(
        frame=100,
        df=df_pred,
        joints_number=joints_number,
        joints_name=joints_3d,
        ax=ax,
    )
    plt.legend()
    plt.show()
