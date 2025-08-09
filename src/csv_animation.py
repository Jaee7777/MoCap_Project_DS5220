import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


def read_CMU_data(file_pos="data/01_01_pos.csv", file_rot="data/01_01_rot.csv"):
    scale = (1.0 / 0.45) * 2.54 / 100.0  # scale from CMU mocap.

    df_pos = pd.read_csv(file_pos)
    df_rot = pd.read_csv(file_rot)

    joints_number = int(len(df_pos.columns) / 3)
    df = pd.merge(df_pos, df_rot, on="time", how="inner")
    # print(df.head())

    joints_name = []
    for i in range(joints_number):
        joints_name.append(df_pos.columns[3 * i + 1][:-2])
    # print(joints_name)

    col_scale = df.columns.difference(["time"])  # exclude time column
    df[col_scale] = df[col_scale] * scale
    col_pos_scale = df_pos.columns.difference(["time"])  # exclude time column
    df_pos[col_pos_scale] = df_pos[col_pos_scale] * scale
    return df, df_pos, df_rot, joints_number, joints_name


def plot_3d(frame, df, joints_name, ax):
    joints_number = int(len(df.columns) / 3)
    plt.cla()
    X = []
    Y = []
    Z = []

    for i in range(joints_number):
        X.append(df.iloc[frame, 3 * i])
        Y.append(df.iloc[frame, 3 * i + 1])
        Z.append(df.iloc[frame, 3 * i + 2])

    i = 0
    joints_mp = [
        "lhipjoint",
        "lfemur",
        "ltibia",
        "lfoot",
        "ltoes",
        "rhipjoint",
        "rfemur",
        "rtibia",
        "rfoot",
        "rtoes",
        "head",
        "lhumerus",
        "lradius",
        "lwrist",
        "lfingers",
        "lthumb",
        "rhumerus",
        "rradius",
        "rwrist",
        "rfingers",
        "rthumb",
    ]
    for joint in joints_name:
        if joint == "upperback":
            marker, size = "X", 200
        elif joint == "head":
            marker, size = "o", 500
        elif joint in joints_mp:
            marker, size = "o", 50
        else:
            marker, size = "x", 50
        ax.scatter(X[i], Y[i], Z[i], marker=marker, s=size, label=joints_name[i])
        # ax.scatter(X[i], Y[i], Z[i], marker="o", label=joints_name[i])
        i += 1

    ax.scatter(X[0], Y[0], Z[0], marker="X", s=100)

    ax.plot3D(
        [X[0], X[30], X[9], X[29], X[8], X[22]],
        [Y[0], Y[30], Y[9], Y[29], Y[8], Y[22]],
        [Z[0], Z[30], Z[9], Z[29], Z[8], Z[22]],
        color="purple",
        linestyle="-",
        linewidth=2,
        label="Spine",
    )
    ax.plot3D(
        [X[6], X[2], X[12], X[4], X[13]],
        [Y[6], Y[2], Y[12], Y[4], Y[13]],
        [Z[6], Z[2], Z[12], Z[4], Z[13]],
        color="red",
        linestyle="-",
        linewidth=2,
        label="Leg-Left",
    )
    ax.plot3D(
        [X[20], X[16], X[25], X[18], X[26]],
        [Y[20], Y[16], Y[25], Y[18], Y[26]],
        [Z[20], Z[16], Z[25], Z[18], Z[26]],
        color="blue",
        linestyle="-",
        linewidth=2,
        label="Leg-Right",
    )
    ax.plot3D(
        [X[1], X[7], X[10], X[14], X[5], X[11], X[3]],
        [Y[1], Y[7], Y[10], Y[14], Y[5], Y[11], Y[3]],
        [Z[1], Z[7], Z[10], Z[14], Z[5], Z[11], Z[3]],
        color="green",
        linestyle="-",
        linewidth=2,
        label="Arm-Left",
    )
    ax.plot3D(
        [X[15], X[21], X[23], X[27], X[19], X[24], X[17]],
        [Y[15], Y[21], Y[23], Y[27], Y[19], Y[24], Y[17]],
        [Z[15], Z[21], Z[23], Z[27], Z[19], Z[24], Z[17]],
        color="cyan",
        linestyle="-",
        linewidth=2,
        label="Arm-Right",
    )
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 4)
    ax.set_zlim(-2, 2)
    ax.legend()
    return


def plot_2d(frame, df, joints):
    plt.cla()
    X = []
    Y = []

    for i in range(len(joints)):
        X.append(df.iloc[frame, 2 * i])
        Y.append(df.iloc[frame, 2 * i + 1])

    i = 0
    for joint in joints:
        if joint == "head":
            marker, size = "o", 500
        else:
            marker, size = "o", 50
        plt.scatter(X[i], Y[i], marker=marker, s=size, label=joint)
        i += 1

    plt.plot(
        [X[1], X[7], X[3], X[8]],
        [Y[1], Y[7], Y[3], Y[8]],
        color="red",
        linestyle="-",
        linewidth=2,
        label="Leg-Left",
    )
    plt.plot(
        [X[10], X[16], X[12], X[17]],
        [Y[10], Y[16], Y[12], Y[17]],
        color="blue",
        linestyle="-",
        linewidth=2,
        label="Leg-Right",
    )
    plt.plot(
        [X[4], X[5], X[9], X[6], X[2]],
        [Y[4], Y[5], Y[9], Y[6], Y[2]],
        color="green",
        linestyle="-",
        linewidth=2,
        label="Arm-Left",
    )
    plt.plot(
        [X[13], X[14], X[18], X[15], X[11]],
        [Y[13], Y[14], Y[18], Y[15], Y[11]],
        color="cyan",
        linestyle="-",
        linewidth=2,
        label="Arm-Right",
    )
    plt.plot(
        [X[13], X[4], X[1], X[10], X[13]],
        [Y[13], Y[4], Y[1], Y[10], Y[13]],
        color="purple",
        linestyle="-",
        linewidth=2,
        label="Torso",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend()
    return


def CMU_to_MPipe():
    dict = {
        2: "lfemur",
        3: "ltibia",
        4: "lfoot",
        5: "ltoes",
        7: "rfemur",
        8: "rtibia",
        9: "rfoot",
        10: "rtoes",
        16: "head",
        18: "lhumerus",
        19: "lradius",
        20: "lwrist",
        22: "lfingers",
        23: "lthumb",
        25: "rhumerus",
        26: "rradius",
        27: "rwrist",
        29: "rfingers",
        30: "rthumb",
    }
    return dict


def plotit(X_test, df_pred):

    joints_2d = []
    for i in range(int(len(X_test.columns) / 2)):
        joints_2d.append(X_test.columns[2 * i][:-2])
    joints_2d = sorted(joints_2d)

    joints_3d = []
    for i in range(int(len(df_pred.columns) / 3)):
        joints_3d.append(df_pred.columns[3 * i][:-2])
    joints_3d = sorted(joints_3d)

    # 3D plot.
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection="3d")
    plot_3d(
        frame=1000,
        df=df_pred,
        joints_name=joints_3d,
        ax=ax,
    )
    # 2D plot.
    fig = plt.figure(2)
    plot_2d(frame=1000, df=X_test, joints=joints_2d)
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    path_data_3d = "data/data_CMU_3d_01.csv"
    path_data_2d = "data/data_CMU_2d_01.csv"
    df_pos = pd.read_csv(path_data_3d)
    df_2d = pd.read_csv(path_data_2d)

    df_col_chose = df_pos.columns.difference(["time"])
    df_2d_col_chose = df_2d.columns.difference(["time"])

    df_2d = df_2d[df_2d_col_chose]
    df_pos = df_pos[df_col_chose]

    joints_2d = []
    for i in range(int(len(df_2d.columns) / 2)):
        joints_2d.append(df_2d.columns[2 * i][:-2])
    joints_2d = sorted(joints_2d)

    joints_3d = []
    for i in range(int(len(df_pos.columns) / 3)):
        joints_3d.append(df_pos.columns[3 * i][:-2])
    joints_3d = sorted(joints_3d)

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection="3d")
    """
    plot_3d(
        frame=0,
        df=df_pos,
        joints_number=joints_number,
        joints_name=joints_3d,
        ax=ax,
    )
    """
    ani_3d = animation.FuncAnimation(
        fig=fig,
        func=plot_3d,
        fargs=(df_pos, joints_3d, ax),
        interval=8.333,
    )

    # 2D animation
    fig = plt.figure(2)
    ani_2d = animation.FuncAnimation(
        fig=fig,
        func=plot_2d,
        fargs=(df_2d, joints_2d),
        interval=8.333,
    )
    plt.legend()
    plt.show()
