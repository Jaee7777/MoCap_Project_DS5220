def parse_asf(file_path):
    with open(file_path) as file:
        lines = file.readlines()  # Each line in the file is the list elements.

    frame = []
    frame_data = []
    for line in lines:
        line = line.strip()  # Seperate each line out as a variable.

        if line.startswith("#") or line.startswith(":") or not line:  # Remove comments
            continue
        if line.isdigit() is True:  # Check for the frame number.
            frame.append(int(line))
        else:
            frame_data.append(line.split())

    # List all the joint names, number of joints and dimension of joint data.
    joint = {}  # {Joint name: Joint dimension}
    joint_name = frame_data[0][0]
    joint_number = 0
    while joint_name not in joint:
        joint.update({joint_name: len(frame_data[joint_number]) - 1})
        joint_number += 1
        joint_name = frame_data[joint_number][0]
    print(joint)
    print(sum(joint.values()))
    return


if __name__ == "__main__":
    print("hi")
    file_path = "data/test.asf"
    parse_asf(file_path)
