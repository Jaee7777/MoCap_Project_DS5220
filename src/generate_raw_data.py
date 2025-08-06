import pandas as pd
import sys


def read_CMU_pos_data(file_pos="data/all_asfamc/subjects/01/01_01_pos.csv"):
    scale = (1.0 / 0.45) * 2.54 / 100.0  # Scale from CMU mocap.

    df = pd.read_csv(file_pos)

    joints_number = int(len(df.columns) / 3)  # Number of the joints.

    joints_name = []
    for i in range(joints_number):
        joints_name.append(df.columns[3 * i + 1][:-2])  # Name of the joints.

    df[df.columns] = df[df.columns] * scale  # Scale units.
    return df


if __name__ == "__main__":
    # The input of this script is: input 3d csv files that you want to merge together, and output file name.
    if len(sys.argv) > 1:
        df1 = read_CMU_pos_data(sys.argv[1])
        for i, file in enumerate(sys.arv):
            if i > 0 and i < len(sys.argv) - 2:
                df2 = read_CMU_pos_data(sys.argv[i + 1])
                df1 = pd.concat([df1, df2], axis=0)
        df1.to_csv(sys.argv[-1], index=False, float_format="%.5f")
    else:
        print("No input received.")
