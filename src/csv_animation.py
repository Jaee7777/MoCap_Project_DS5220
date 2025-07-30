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


def plot_2d(frame, df, joints):
    plt.cla()
    X = []
    Y = []

    for i in range(len(joints)):
        X.append(df.iloc[frame, 2 * i])
        Y.append(df.iloc[frame, 2 * i + 1])

    i = 0
    for joint in joints.values():
        if joint == "head":
            marker, size = "o", 500
        else:
            marker, size = "o", 50
        plt.scatter(X[i], Y[i], marker=marker, s=size, label=joint)
        i += 1

    plt.plot(
        X[0:4],
        Y[0:4],
        color="red",
        linestyle="-",
        linewidth=2,
        label="Leg-Left",
    )
    plt.plot(
        X[4:8],
        Y[4:8],
        color="blue",
        linestyle="-",
        linewidth=2,
        label="Leg-Right",
    )
    plt.plot(
        X[9:14],
        Y[9:14],
        color="green",
        linestyle="-",
        linewidth=2,
        label="Arm-Left",
    )
    plt.plot(
        X[14:19],
        Y[14:19],
        color="purple",
        linestyle="-",
        linewidth=2,
        label="Arm-right",
    )
    plt.plot(
        [X[0], X[4], X[14], X[9], X[0]],
        [Y[0], Y[4], Y[14], Y[9], Y[0]],
        color="cyan",
        linestyle="-",
        linewidth=2,
        label="Torso",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim([0, 0.01])
    plt.ylim([0, 0.01])
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


if __name__ == "__main__":
    file_pos = "data/01_01_pos.csv"
    file_rot = "data/01_01_rot.csv"
    df, df_pos, df_rot, joints_number, joints_name = read_data(
        file_pos=file_pos, file_rot=file_rot
    )
    print(joints_name)

    df_2d = pd.read_csv("data/01_01_2d.csv")
    print(df_2d.iloc[0, 3])
    fig = plt.figure()
    """
    ax = fig.add_subplot(111, projection="3d")
    plot_frame(
        frame=0, df=df, joints_number=joints_number, joints_name=joints_name, ax=ax
    )
    """
    """
    ani = animation.FuncAnimation(
        fig=fig,
        func=plot_frame,
        fargs=(df, joints_number, joints_name, ax),
        interval=8.333,
    )"""
    # plot_2d(frame=0, df=df_2d, joints=CMU_to_MPipe())
    ani = animation.FuncAnimation(
        fig=fig,
        func=plot_2d,
        fargs=(df_2d, CMU_to_MPipe()),
        interval=8.333,
    )
    plt.legend()
    plt.show()
