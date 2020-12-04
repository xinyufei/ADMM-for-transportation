import numpy as np
import gurobipy as gb
# from data import Network
import matplotlib.pyplot as plt
# from data_global import *
from warmup import Warmup
# from draw_signal import *
import matplotlib as mpl

def plot_vehicle(iteration, T, num_scenario, mode, add, num_cells, n_value_eval, y_value_eval):
    fetch_cell_index = [5, 7, 8, add[1], 1+add[1], 3+add[1], 5+add[1], 6+add[1], 7+add[1], add[2], 1+add[2], 
    2+add[2], 4+add[2], 6+add[2], 7+add[2], add[3], 1+add[3], 3+add[3], 5+add[3], 6+add[3], 7+add[3]]
    fetch_cell_index_east = [14+add[3], 16+add[3], 17+add[3], 8+add[2], 9+add[2], 11+add[2], 13+add[2], 14+add[2], 15+add[2], 
    8+add[1], 9+add[1], 10+add[1], 12+add[1], 14+add[1], 15+add[1], 9, 10, 12, 14, 15, 16]
    # fetch_column_index = []

    # fetch n from grb vars
    """ n_value = m.getAttr('X',n)
    y_value = m.getAttr('X',y)
    w_value = m.getAttr('X',w) """

    corridor_index = 0
    corridor_cell_num = add[4]
    # print(add)
    draw_number = 4

    signal_tick = [0]
    for i_c in range(8):
        signal_tick.append(signal_tick[-1] + 15)
        signal_tick.append(signal_tick[-1] + 10)

    colors = [(0, 1, 0), (1, 1, 0), (1, 0, 0)]          # list of the color map
    n_bins = 1000               # number of bins of the colorbar
    cmap = mpl.colors.LinearSegmentedColormap.from_list("cmap_name", colors, n_bins)

    plt.figure(dpi=300, figsize=[15, 4])
    for i_corridor in range(draw_number):
        current_fetch_cells = [val + i_corridor * corridor_cell_num for val in fetch_cell_index]
        print(current_fetch_cells)
        # plt.subplot(draw_number * 2, 1, i_corridor+2)
        
        result_n_evaluation = []
        result_n_combination = []
        for i in range(num_cells):
            if not (i in current_fetch_cells):
                continue
            local_veh_num = []
            local_veh_num_eval = []
            for t in range(T):
                # local_veh_num.append(n_value[i, t])
                local_veh_num_eval.append(n_value_eval[i, t])
            result_n_combination.append(local_veh_num)
            result_n_evaluation.append(local_veh_num_eval)

        result_y_combination = []
        result_y_evaluation = []
        for i in range(num_cells):
            if not (i in current_fetch_cells):
                continue
            local_veh_num = []
            local_veh_num_eval = []
            for t in range(T):
                # local_veh_num.append(y_value[i, t])
                local_veh_num_eval.append(y_value_eval[i, t])
            result_y_combination.append(local_veh_num)
            result_y_evaluation.append(local_veh_num_eval)

        """ result_w = np.zeros((N,4,T))
        for i in range(N):
            for j in range(4):
                for t in range(T):
                    result_w[i, j, t] = w_tilde[i,j,t] """

        # print("corridor number", i_corridor, )
        """ plt.subplot(draw_number * 2, 2, i_corridor * 4+1)
        plt.imshow(np.array(result_n_combination), aspect="auto", origin="lower", cmap="binary")
        plt.yticks(range(len(current_fetch_cells)), [str(val) for val in current_fetch_cells])
        # plt.xticks(signal_tick, [str(val) for val in signal_tick])
        plt.title("Number of vehicles per cell")
        plt.colorbar() """

        plt.subplot(1, draw_number, i_corridor+1)
        plt.imshow(np.array(result_n_evaluation), aspect="auto", origin="lower", cmap="binary")
        plt.yticks([0,5,12,17], [str(1+(i_corridor)*draw_number), str(2+(i_corridor)*draw_number), str(3+(i_corridor)*draw_number), str(4+(i_corridor)*draw_number)])
        plt.xticks(range(0,T,80), [str(int(val*3/60)) for val in range(0,T,80)])
        plt.xlabel('Time(min)')
        plt.ylabel('Intersection')
        # plt.title("Number of vehicles per cell")
        plt.colorbar()

        """ plt.subplot(draw_number * 2, 2, i_corridor * 4 + 3)
        plt.imshow(np.array(result_y_combination), aspect="auto", origin="lower", cmap="binary")
        plt.yticks(range(len(current_fetch_cells)), [str(val) for val in current_fetch_cells])
        # plt.xticks(signal_tick, [str(val) for val in signal_tick])
        plt.title("Flow")
        plt.colorbar() """

        """ plt.subplot(draw_number * 2, 1, i_corridor * 2 + 2)
        plt.imshow(np.array(result_y_evaluation), aspect="auto", origin="lower", cmap="binary")
        plt.yticks(range(len(current_fetch_cells)), [str(val) for val in current_fetch_cells])
        # plt.xticks(signal_tick, [str(val) for val in signal_tick])
        plt.title("Flow")
        plt.colorbar() """

    plt.tight_layout()
    plt.savefig("figure/T_"+str(T)+"_S_"+str(num_scenario)+"_"+mode+"_"+str(iteration)+"_west_bound.png")
    # plt.show()
    plt.close()

    plt.figure(dpi=300, figsize=[15, 4])
    for i_corridor in range(draw_number):
        current_fetch_cells_east = [val + i_corridor * corridor_cell_num for val in fetch_cell_index_east]
        print(current_fetch_cells_east)
        # plt.subplot(draw_number * 2, 1, i_corridor+2)
        
        result_n_evaluation = []
        result_n_combination = []
        for c in current_fetch_cells_east:
            local_veh_num_eval = []
            for t in range(T):
                local_veh_num_eval.append(n_value_eval[int(c), t])
            result_n_evaluation.append(local_veh_num_eval)

        plt.subplot(1, draw_number, i_corridor+1)
        plt.imshow(np.array(result_n_evaluation), aspect="auto", origin="lower", cmap="binary")
        plt.yticks([0,5,12,17], [str(4+(i_corridor)*draw_number), str(3+(i_corridor)*draw_number), str(2+(i_corridor)*draw_number), str(1+(i_corridor)*draw_number)])
        plt.xticks(range(0,T,80), [str(int(val*3/60)) for val in range(0,T,80)])
        plt.xlabel('Time(min)')
        plt.ylabel('Intersection')
        # plt.title("Number of vehicles per cell")
        plt.colorbar()

    plt.tight_layout()
    plt.savefig("figure/T_"+str(T)+"_S_"+str(num_scenario)+"_"+mode+"_"+str(iteration)+"_east_bound.png")
    # plt.show()
    plt.close()

    fetch_cell_index_north_1 = [28, 30, 31, 25+add[4], 26+add[4], 28+add[4], 30+add[4], 31+add[4], 
    25+add[8], 26+add[8], 28+add[8], 30+add[8], 31+add[8], 
    25+add[12], 26+add[12], 28+add[12], 30+add[12]]
    fetch_cell_index_north_2 = [26, 28, 29, 23+add[4], 24+add[4], 26+add[4], 28+add[4], 29+add[4], 
    23+add[8], 24+add[8], 26+add[8], 28+add[8], 29+add[8], 
    23+add[12], 24+add[12], 26+add[12], 28+add[12]]
    plt.figure(dpi=300, figsize=[15, 4])
    for i_corridor in range(draw_number):
        if i_corridor == 0 or i_corridor == 3:
            current_fetch_cells_north = [val + add[i_corridor] for val in fetch_cell_index_north_1]
        if i_corridor == 1 or i_corridor == 2:
            current_fetch_cells_north = [val + add[i_corridor] for val in fetch_cell_index_north_2]
        print(current_fetch_cells_north)
        # plt.subplot(draw_number * 2, 1, i_corridor+2)
        
        result_n_evaluation = []
        result_n_combination = []
        for c in current_fetch_cells_north:
            # print(c)
            local_veh_num_eval = []
            for t in range(T):
                local_veh_num_eval.append(n_value_eval[int(c), t])
            result_n_evaluation.append(local_veh_num_eval)

        plt.subplot(1, draw_number, i_corridor+1)
        plt.imshow(np.array(result_n_evaluation), aspect="auto", origin="lower", cmap="binary")
        plt.yticks([0,5,10,15], [str(1+(i_corridor)+j*draw_number) for j in range(4)])
        plt.xticks(range(0,T,80), [str(int(val*3/60)) for val in range(0,T,80)])
        plt.xlabel('Time(min)')
        plt.ylabel('Intersection')
        # plt.title("Number of vehicles per cell")
        plt.colorbar()

    plt.tight_layout()
    plt.savefig("figure/T_"+str(T)+"_S_"+str(num_scenario)+"_"+mode+"_"+str(iteration)+"_north_bound.png")
    # plt.show()
    plt.close()

    fetch_cell_index_south_1 = [21+3*corridor_cell_num, 23+3*corridor_cell_num, 24+3*corridor_cell_num, 
    18+2*corridor_cell_num, 19+2*corridor_cell_num, 21+2*corridor_cell_num, 23+2*corridor_cell_num, 24+2*corridor_cell_num, 
    18+corridor_cell_num, 19+corridor_cell_num, 21+corridor_cell_num, 23+corridor_cell_num, 24+corridor_cell_num,
    18,19,21,23]
    fetch_cell_index_south_2 = [19+3*corridor_cell_num, 21+3*corridor_cell_num, 22+3*corridor_cell_num, 
    16+2*corridor_cell_num, 17+2*corridor_cell_num, 19+2*corridor_cell_num, 21+2*corridor_cell_num, 22+2*corridor_cell_num, 
    16+corridor_cell_num, 17+corridor_cell_num, 19+corridor_cell_num, 21+corridor_cell_num, 22+corridor_cell_num,
    16,17,19,21]
    plt.figure(dpi=300, figsize=[15, 4])
    for i_corridor in range(draw_number):
        if i_corridor == 0 or i_corridor == 3:
            current_fetch_cells_south = [val + add[i_corridor] for val in fetch_cell_index_south_1]
        if i_corridor == 1 or i_corridor == 2:
            current_fetch_cells_south = [val + add[i_corridor] for val in fetch_cell_index_south_2]
        print(current_fetch_cells_south)
        # plt.subplot(draw_number * 2, 1, i_corridor+2)
        
        result_n_evaluation = []
        result_n_combination = []
        for c in current_fetch_cells_south:
            local_veh_num_eval = []
            for t in range(T):
                local_veh_num_eval.append(n_value_eval[int(c), t])
            result_n_evaluation.append(local_veh_num_eval)

        plt.subplot(1, draw_number, i_corridor+1)
        plt.imshow(np.array(result_n_evaluation), aspect="auto", origin="lower", cmap="binary")
        plt.yticks([0,5,10,15], [str(1+(i_corridor)+j*draw_number) for j in range(3,-1,-1)])
        plt.xticks(range(0,T,80), [str(int(val*3/60)) for val in range(0,T,80)])
        plt.xlabel('Time(min)')
        plt.ylabel('Intersection')
        # plt.title("Number of vehicles per cell")
        plt.colorbar()

    plt.tight_layout()
    plt.savefig("figure/T_"+str(T)+"_S_"+str(num_scenario)+"_"+mode+"_"+str(iteration)+"_south_bound.png")
    # plt.show()
    plt.close()
    # plot signal status



    """ for i_intersection in range(N):
        plt.figure(dpi=300)
        for i_phase in range(4):
            plt.plot(result_w[i_intersection, i_phase, :], ".-", label="Phase" +str(i_phase +1))
        plt.legend()
        plt.savefig("figure/spat/" + str(i_intersection) + "_spat.png")
        plt.close() """
