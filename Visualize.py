import matplotlib.pyplot as plt


with open("log/n.log", "r") as temp_file:
    all_lines = temp_file.readlines()

veh_num_matrix = []

for single_line in all_lines:
    split_info = single_line.split(",")
    if len(split_info) < 10:
        continue
    veh_num_matrix.append([float(val) for val in split_info[:-1]])

plt.figure()
plt.imshow(veh_num_matrix, aspect="auto", cmap="binary")
plt.colorbar()

plt.xlabel("time")
plt.ylabel("index of cell")
plt.show()