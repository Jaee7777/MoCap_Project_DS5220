import pandas as pd
import numpy as np
import sys


def read_large_csv(file="file.csv", chunksize=50000):
    print("\nReading large csv file...")
    # Read data by chunk in float32 type.
    chunks = []
    i = 0
    for chunk in pd.read_csv(file, chunksize=chunksize, dtype=np.float32):
        print(f"Chunk {i}")
        chunks.append(chunk)
        i += 1

    # Concatenate all chunks.
    df = pd.concat(chunks, ignore_index=True)
    print("Reading large csv done!")
    return df


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


def pic_3d_to_2d(x_3d, y_3d, z_3d, f=1, d=10):
    ratio = f / (z_3d + d)
    x_2d = ratio * x_3d
    y_2d = ratio * y_3d
    return x_2d, y_2d


def csv_3d_to_2d(list_i, df_3d, focal_length=1, distance=10):
    print("Generating 2D csv file...")
    col_pos = list(df_3d.columns)
    col_result = []
    for i in list_i:
        col_result.append(col_pos[3 * i + 1])
        col_result.append(col_pos[3 * i + 2])

    data_list = [[] for _ in range(df_3d.shape[0])]
    total_row = len(data_list)
    print(f"Number of rows: {total_row}")
    for row in range(total_row):
        for i in list_i:
            x_2d, y_2d = pic_3d_to_2d(
                df_3d.iloc[row, 3 * i + 1],
                df_3d.iloc[row, 3 * i + 2],
                df_3d.iloc[row, 3 * i + 3],
                f=focal_length,
                d=distance,
            )
            data_list[row].append(x_2d)
            data_list[row].append(y_2d)

    df_result = pd.DataFrame(data_list, columns=col_result)
    print("2D csv file created!")
    return df_result


def generate_2d_csv(
    chunksize=50000, global_x_max=None, global_x_min=None, global_y_max=None
):
    if len(sys.argv) > 1:
        df_3d = read_large_csv(file=sys.argv[1], chunksize=chunksize)
        dict = CMU_to_MPipe()
        df_2d = csv_3d_to_2d(list(dict.keys()), df_3d, focal_length=0.05, distance=10)

        # Find global min-max if not given.
        if global_x_max is None or global_x_min is None or global_y_max is None:
            x_min = []
            x_max = []
            y_max = []
            for item in df_2d.columns:
                if item[-1] == "x":
                    x_min.append(df_2d[item].min())
                    x_max.append(df_2d[item].max())
                else:
                    y_max.append(df_2d[item].max())
            global_x_max = max(x_max)
            global_x_min = min(x_min)
            global_y_max = max(y_max)
            print(f"Global x max: {global_x_max}")
            print(f"Global x min: {global_x_min}")
            print(f"Global y max: {global_y_max}")

        # Scale df_2d.
        for item in df_2d.columns:
            if item[-1] == "x":
                df_2d[item] = df_2d[item] - global_x_min
        if global_y_max > (global_x_max - global_x_min):
            df_2d = df_2d / global_y_max
        else:
            df_2d = df_2d / (global_x_max - global_x_min)
        df_2d.to_csv(sys.argv[2], index=False, float_format="%.5f")
    else:
        print("No input received.")
    return


if __name__ == "__main__":
    # This script requires 2 inputs: input 3d csv file name and output 2d csv file name.
    if sys.argv[3] is None:
        generate_2d_csv(chunksize=50000)
    else:
        generate_2d_csv(
            chunksize=50000,
            global_x_max=np.float32(sys.argv[3]),
            global_x_min=np.float32(sys.argv[4]),
            global_y_max=np.float32(sys.argv[5]),
        )
