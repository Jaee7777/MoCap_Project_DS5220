import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


def read_data(file_pos="data/01_01_pos.csv", file_rot="data/01_01_rot.csv"):
    df_pos = pd.read_csv(file_pos)
    df_rot = pd.read_csv(file_rot)

    joints_number = int(len(df_pos.columns) / 3)
    df = pd.merge(df_pos, df_rot, on="time", how="inner")
    # print(df.head())

    joints_name = []
    for i in range(joints_number):
        joints_name.append(df_pos.columns[3 * i + 1][:-2])
    # print(joints_name)
    return df, df_pos, df_rot, joints_number, joints_name


def plot_frame(frame, df, joints_number, joints_name, ax):
    plt.cla()
    X = []
    Y = []
    Z = []

    for i in range(joints_number):
        X.append(df.iloc[frame, 3 * i + 1])
        Y.append(df.iloc[frame, 3 * i + 2])
        Z.append(df.iloc[frame, 3 * i + 3])

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
        if joint == "root":
            marker, size = "X", 100
        elif joint == "head":
            marker, size = "o", 500
        elif joint in joints_mp:
            marker, size = "o", 50
        else:
            marker, size = "x", 100
        ax.scatter(X[i], Y[i], Z[i], marker=marker, s=size, label=joints_name[i])
        # ax.scatter(X[i], Y[i], Z[i], marker="o", label=joints_name[i])
        i += 1

    ax.scatter(X[0], Y[0], Z[0], marker="X", s=100)

    ax.plot3D(
        X[1:6],
        Y[1:6],
        Z[1:6],
        color="red",
        linestyle="-",
        linewidth=2,
        label="Leg-Left",
    )
    ax.plot3D(
        X[6:11],
        Y[6:11],
        Z[6:11],
        color="blue",
        linestyle="-",
        linewidth=2,
        label="Leg-Right",
    )
    ax.plot3D(
        X[11:17],
        Y[11:17],
        Z[11:17],
        color="green",
        linestyle="-",
        linewidth=2,
        label="Spine",
    )
    ax.plot3D(
        X[17:23],
        Y[17:23],
        Z[17:23],
        color="cyan",
        linestyle="-",
        linewidth=2,
        label="Arm-Left",
    )
    ax.plot3D(
        X[24:30],
        Y[24:30],
        Z[24:30],
        color="purple",
        linestyle="-",
        linewidth=2,
        label="Arm-Right",
    )
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_xlim(-15, 15)
    ax.set_ylim(0, 30)
    ax.set_zlim(-15, 15)
    return


if __name__ == "__main__":
    file_pos = "data/01_01_pos.csv"
    file_rot = "data/01_01_rot.csv"
    df, df_pos, df_rot, joints_number, joints_name = read_data(
        file_pos=file_pos, file_rot=file_rot
    )
    print(joints_name)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_frame(
        frame=0, df=df, joints_number=joints_number, joints_name=joints_name, ax=ax
    )
    """
    ani = animation.FuncAnimation(
        fig=fig,
        func=plot_frame,
        fargs=(df, joints_number, joints_name, ax),
        interval=8.333,
    )"""
    plt.legend()
    plt.show()
