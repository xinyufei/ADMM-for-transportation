# you might need to install:
# pip install opencv-python
# and also tqdm for progress bar

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

block_size = 10


def get_color_tuple(val, max_val=8):
    if val >= max_val:
        color = (1, 0, 0)
    elif val < max_val / 4:
        prop = val / max_val * 4
        prop = np.max([0, prop])
        g_channel = 1
        r_channel = prop
        color = (r_channel, g_channel, 0)
    else:
        prop = (val - max_val * 1 / 4) / (3 / 4 * max_val)
        prop = np.min([1, prop])
        r_channel = 1
        g_channel = 1 - prop
        color = (r_channel, g_channel, 0)
    return color


def draw_chain_blocks(axis, cord, val_list, mod="horizon"):
    for idx in range(len(val_list)):
        val = val_list[idx]
        color = get_color_tuple(val, 8)
        if mod == "horizon":
            rect = Rectangle((cord[0] * block_size + idx * block_size, cord[1] * block_size), block_size, block_size,
                             color=color)
        else:
            rect = Rectangle((cord[0] * block_size, block_size * cord[1] + block_size * idx), block_size, block_size,
                             color=color)
        axis.add_patch(rect)
    return axis


# # Test the color function
# fig, ax = plt.subplots()
# fig.set_size_inches(18, 4)
# draw_chain_blocks(ax, [0, 0], np.linspace(0, 8, 20))
#
# plt.xlim([0, 200])
# plt.ylim([0, 10])
# plt.show()
# exit()


def draw_snap_shot(veh_list, time_step):
    fig, ax = plt.subplots()
    fig.set_size_inches(18, 4)

    corridor_lower_x = 2
    corridor_lower_y = 10
    corridor_upper_x = 2
    corridor_upper_y = 12

    draw_chain_blocks(ax, [corridor_lower_x, corridor_lower_y], veh_list[0: 10])
    draw_chain_blocks(ax, [corridor_lower_x + 10, corridor_lower_y - 1], veh_list[10: 13][::-1], mod="vertical")
    draw_chain_blocks(ax, [corridor_upper_x + 11, corridor_upper_y + 2], veh_list[308: 318][::-1], mod="vertical")
    draw_chain_blocks(ax, [corridor_upper_x + 10, corridor_upper_y + 1], veh_list[318: 321][::-1])
    draw_chain_blocks(ax, [corridor_upper_x + 13, corridor_upper_y + 2], veh_list[298: 308][::-1], mod="vertical")
    plt.text((corridor_lower_x + 10) * block_size, (corridor_upper_y - 1) * block_size, "Barton", fontsize=12)

    draw_chain_blocks(ax, [corridor_lower_x + 15, corridor_lower_y], veh_list[13: 46])
    draw_chain_blocks(ax, [corridor_lower_x + 15, corridor_upper_y], veh_list[252: 285][::-1])
    draw_chain_blocks(ax, [corridor_lower_x + 14, corridor_upper_y - 1], veh_list[285: 288], mod="vertical")

    draw_chain_blocks(ax, [corridor_upper_x, corridor_upper_y], veh_list[288: 298][::-1])
    draw_chain_blocks(ax, [corridor_lower_x + 53, corridor_lower_y], veh_list[49: 76])
    draw_chain_blocks(ax, [corridor_upper_x + 53, corridor_upper_y], veh_list[222: 249][::-1])
    draw_chain_blocks(ax, [corridor_upper_x + 52, corridor_upper_y - 1], veh_list[249: 252], mod="vertical")
    plt.text((corridor_lower_x + 48) * block_size, (corridor_upper_y - 1) * block_size, "Murfin", fontsize=12)

    draw_chain_blocks(ax, [corridor_lower_x + 48, corridor_lower_y - 1], veh_list[46: 49][::-1], mod="vertical")
    draw_chain_blocks(ax, [corridor_lower_x + 49, corridor_lower_y - 11], veh_list[357: 367], mod="vertical")
    draw_chain_blocks(ax, [corridor_upper_x + 49, corridor_upper_y + 2], veh_list[344: 354], mod="vertical")
    draw_chain_blocks(ax, [corridor_upper_x + 48, corridor_upper_y + 1], veh_list[354: 357][::-1])
    draw_chain_blocks(ax, [corridor_lower_x + 50, corridor_lower_y - 1], veh_list[331: 334])
    draw_chain_blocks(ax, [corridor_upper_x + 51, corridor_upper_y + 2], veh_list[334: 344], mod="vertical")
    draw_chain_blocks(ax, [corridor_lower_x + 51, corridor_lower_y - 11], veh_list[321: 331], mod="vertical")
    draw_chain_blocks(ax, [corridor_lower_x + 80, corridor_lower_y - 1], veh_list[76: 79][::-1], mod="vertical")
    plt.text((corridor_lower_x + 78) * block_size, (corridor_upper_y - 1) * block_size, "Traverwood", fontsize=12)
    draw_chain_blocks(ax, [corridor_upper_x + 80, corridor_upper_y + 1], veh_list[387: 390])
    draw_chain_blocks(ax, [corridor_upper_x + 81, corridor_upper_y + 2], veh_list[377: 387][::-1], mod="vertical")
    draw_chain_blocks(ax, [corridor_upper_x + 83, corridor_upper_y + 2], veh_list[367: 377], mod="vertical")

    draw_chain_blocks(ax, [corridor_lower_x + 85, corridor_lower_y], veh_list[79: 91])
    draw_chain_blocks(ax, [corridor_upper_x + 84, corridor_upper_y - 1], veh_list[219: 222], mod="vertical")
    draw_chain_blocks(ax, [corridor_upper_x + 85, corridor_upper_y], veh_list[207: 219][::-1])
    draw_chain_blocks(ax, [corridor_lower_x + 97, corridor_lower_y - 1], veh_list[91: 94][::-1], mod="vertical")
    plt.text((corridor_lower_x + 97) * block_size, (corridor_upper_y - 1) * block_size, "Nixon", fontsize=12)
    draw_chain_blocks(ax, [corridor_upper_x + 97, corridor_upper_y + 1], veh_list[423: 426][::-1])
    draw_chain_blocks(ax, [corridor_upper_x + 98, corridor_upper_y + 2], veh_list[413: 423][::-1], mod="vertical")
    draw_chain_blocks(ax, [corridor_upper_x + 100, corridor_upper_y + 2], veh_list[403:413][::-1], mod="vertical")
    draw_chain_blocks(ax, [corridor_lower_x + 98, corridor_lower_y - 11], veh_list[426: 436][::-1], mod="vertical")
    draw_chain_blocks(ax, [corridor_lower_x + 99, corridor_lower_y - 1], veh_list[400: 403])
    draw_chain_blocks(ax, [corridor_lower_x + 100, corridor_lower_y - 11], veh_list[390: 400], mod="vertical")

    draw_chain_blocks(ax, [corridor_lower_x + 102, corridor_lower_y], veh_list[94: 101])
    draw_chain_blocks(ax, [corridor_upper_x + 101, corridor_upper_y - 1], veh_list[204: 207][::-1], mod="vertical")
    draw_chain_blocks(ax, [corridor_upper_x + 102, corridor_upper_y], veh_list[197: 204][::-1])
    draw_chain_blocks(ax, [corridor_lower_x + 109, corridor_lower_y - 1], veh_list[101: 104][::-1], mod="vertical")
    plt.text((corridor_lower_x + 109) * block_size, (corridor_upper_y - 1) * block_size, "Huron", fontsize=12)
    draw_chain_blocks(ax, [corridor_upper_x + 109, corridor_upper_y + 1], veh_list[469: 472][::-1])
    draw_chain_blocks(ax, [corridor_upper_x + 110, corridor_upper_y + 2], veh_list[459: 469][::-1], mod="vertical")
    draw_chain_blocks(ax, [corridor_upper_x + 112, corridor_upper_y + 2], veh_list[449: 459], mod="vertical")
    draw_chain_blocks(ax, [corridor_lower_x + 110, corridor_lower_y - 11], veh_list[472: 482][::-1], mod="vertical")
    draw_chain_blocks(ax, [corridor_lower_x + 111, corridor_lower_y - 1], veh_list[492: 495])
    draw_chain_blocks(ax, [corridor_lower_x + 112, corridor_lower_y - 11], veh_list[482: 492], mod="vertical")

    draw_chain_blocks(ax, [corridor_lower_x + 114, corridor_lower_y], veh_list[104: 136])
    draw_chain_blocks(ax, [corridor_upper_x + 113, corridor_upper_y - 1], veh_list[194: 197], mod="vertical")
    draw_chain_blocks(ax, [corridor_upper_x + 114, corridor_upper_y], veh_list[162: 194][::-1])
    draw_chain_blocks(ax, [corridor_lower_x + 146, corridor_lower_y - 1], veh_list[136: 139][::-1], mod="vertical")
    plt.text((corridor_lower_x + 146) * block_size, (corridor_upper_y - 1) * block_size, "Green", fontsize=12)
    draw_chain_blocks(ax, [corridor_upper_x + 146, corridor_upper_y + 1], veh_list[515: 518][::-1])
    draw_chain_blocks(ax, [corridor_upper_x + 147, corridor_upper_y + 2], veh_list[505: 515][::-1], mod="vertical")
    draw_chain_blocks(ax, [corridor_upper_x + 149, corridor_upper_y + 2], veh_list[495: 505], mod="vertical")
    draw_chain_blocks(ax, [corridor_lower_x + 147, corridor_lower_y - 11], veh_list[518: 528], mod="vertical")
    draw_chain_blocks(ax, [corridor_lower_x + 148, corridor_lower_y - 1], veh_list[492: 495])
    draw_chain_blocks(ax, [corridor_lower_x + 149, corridor_lower_y - 11], veh_list[482: 492], mod="vertical")

    draw_chain_blocks(ax, [corridor_lower_x + 151, corridor_lower_y], veh_list[139: 149])
    draw_chain_blocks(ax, [corridor_upper_x + 150, corridor_upper_y - 1], veh_list[159: 162], mod="vertical")
    draw_chain_blocks(ax, [corridor_upper_x + 151, corridor_upper_y], veh_list[149: 159][::-1])
    plt.xlim([0, 1650])
    plt.ylim([-50, 300])
    plt.title("Time step " + str(time_step))
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("buffer/buffer.png")
    # plt.show()
    # exit()
    plt.close()


with open("log/n.log", "r") as temp_file:
    all_lines = temp_file.readlines()

veh_num_matrix = []

for single_line in all_lines:
    split_info = single_line.split(",")
    if len(split_info) < 10:
        continue
    veh_num_matrix.append([float(val) for val in split_info[:-1]])

# record video
img = cv2.imread("buffer/buffer.png")
height, width, layers = img.shape
video = cv2.VideoWriter("animation.avi", cv2.VideoWriter_fourcc("X", "V", "I", "D"), 10, (width, height))

cells, time_steps = np.shape(veh_num_matrix)
figures = []
for i_step in tqdm(range(time_steps)):
    veh_num_vec = [veh_num_matrix[i_c][i_step] for i_c in range(cells)]
    # print(veh_num_vec)
    draw_snap_shot(veh_num_vec, i_step)
    img = cv2.imread("buffer/buffer.png")

    video.write(img)

video.release()
