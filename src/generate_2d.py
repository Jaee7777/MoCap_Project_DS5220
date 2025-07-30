import pandas as pd


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


def pic_3d_to_2d(x_3d, y_3d, z_3d, f=1):
    scale = (1.0 / 0.45) * 2.54 / 100.0  # scale from CMU mocap.
    ratio = f / (scale * z_3d + 10)
    x_2d = scale * ratio * x_3d
    y_2d = scale * ratio * y_3d
    return x_2d, y_2d


def csv_3d_to_2d(list_i, file="data/01_01_pos.csv", focal_length=1):
    df_pos = pd.read_csv(file)
    col_pos = list(df_pos.columns)
    col_result = []
    for i in list_i:
        col_result.append(col_pos[3 * i + 1])
        col_result.append(col_pos[3 * i + 2])

    data_list = [[] for _ in range(df_pos.shape[0])]
    for row in range(len(data_list)):
        for i in list_i:
            x_2d, y_2d = pic_3d_to_2d(
                df_pos.iloc[row, 3 * i + 1],
                df_pos.iloc[row, 3 * i + 2],
                df_pos.iloc[row, 3 * i + 3],
                f=focal_length,
            )
            data_list[row].append(x_2d)
            data_list[row].append(y_2d)

    df_result = pd.DataFrame(data_list, columns=col_result)
    return df_result


if __name__ == "__main__":
    file_pos = "data/01_01_pos.csv"
    dict = CMU_to_MPipe()
    df_2d = csv_3d_to_2d(list(dict.keys()), file=file_pos, focal_length=0.05)
    df_2d.to_csv("data/01_01_2d.csv", index=False, float_format="%.5f")
