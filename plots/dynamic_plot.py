import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import itertools
import gc
from matplotlib import animation

path="/home/xinchi/GNN-control/demo_data/model_6"

# velocity_array=np.load(os.path.join(path,"velocity_array_scene.npy"))

def gabriel(pose_array):
    node_mum = np.shape(pose_array)[0]
    gabriel_graph=[[1]*node_mum for _ in range(node_mum)]
    position_array=pose_array[:,-1,:2]
    for u in range(node_mum):
        for v in range(node_mum):
            m=(position_array[u]+position_array[v])/2
            for w in range(node_mum):
                if w==v:
                    continue
                if  np.linalg.norm(position_array[w]-m)<np.linalg.norm(position_array[u]-m):
                    gabriel_graph[u][v]=0
                    gabriel_graph[v][u]=0
                    break
    return gabriel_graph

def formation(path,time,dt):
    gc.enable()
    pose_array = np.load(os.path.join(path, "pose_array_scene.npy"))
    node_num = np.shape(pose_array)[0]
    iters=int(min(pose_array.shape[1],time/dt))
    fig,ax=plt.subplots(figsize=(10, 10))
    plt.pause(3)
    for time_step in range(iters):
        gabriel_graph = [[1] * node_num for _ in range(node_num)]
        position_array = pose_array[:, time_step, :2]
        for u in range(node_num):
            for v in range(node_num):
                m = (position_array[u] + position_array[v]) / 2
                for w in range(node_num):
                    if w == v:
                        continue
                    if np.linalg.norm(position_array[w] - m) < np.linalg.norm(position_array[u] - m):
                        gabriel_graph[u][v] = 0
                        gabriel_graph[v][u] = 0
                        break
        rob_num=node_num
        ax.scatter(position_array[:, 0], position_array[:, 1])
        for i in range(rob_num):
            for j in range(i + 1, rob_num):
                if gabriel_graph[i][j] == 0:
                    continue
                xlist = [position_array[i][0], position_array[j][0]]
                ylist = [position_array[i][1], position_array[j][1]]
                distance = math.sqrt((xlist[0] - xlist[1]) ** 2 + (ylist[0] - ylist[1]) ** 2)
                ax.plot(xlist, ylist, label="Distane: {d:.2f}".format(d=distance))
        ax.legend(fontsize=20)
        ax.grid()
        ax.set_xlim(-5,5)
        ax.set_ylim(-5,5)
        plt.xlabel("x(m)", fontsize=30)
        plt.ylabel("y(m)", fontsize=30)
        ax.text(-1,4,'Time={:.2f}s'.format(time_step*dt),size=20)
        plt.subplots_adjust(left=0.12,
                            bottom=0.1,
                            right=0.98,
                            top=0.98,
                            wspace=0.0,
                            hspace=0.0)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        # fig.text(0.5, 0.03, 'X(m)', va='center', fontsize=30)
        # fig.text(0.03, 0.5, "Y(m)", va='center', rotation='vertical', fontsize=30)
        plt.pause(0.01)
        ax.cla()
        gc.collect(generation=2)
    plt.draw()
def distance(path,time,dt):
    gc.enable()
    pose_array = np.load(os.path.join(path, "pose_array_scene.npy"))
    rob_num = np.shape(pose_array)[0]
    iters = iters = int(min(pose_array.shape[1],time/dt))
    gabriel_graph = gabriel(pose_array)
    distance_dict = {}

    for i in range(rob_num):
        for j in range(i + 1, rob_num):
            if gabriel_graph[i][j] == 0:
                continue
            name = str(i + 1) + " to " + str(j + 1)
            distance_array = np.sqrt(np.square(pose_array[i, :, 0] - pose_array[j, :, 0]) + np.square(
                pose_array[i, :, 1] - pose_array[j, :, 1]))
            distance_dict[name] = distance_array
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.pause(3)
    xlist = []
    for time_step in range(iters):
        # ax.scatter(position_array[:, 0], position_array[:, 1])
        xlist.append(time_step*dt)
        for key in distance_dict:
            ax.plot(xlist, distance_dict[key][:time_step+1], label=key)
        ax.legend(fontsize=20)
        # ax.set_title("Time histories of inter-robot distance", size=30)
        # ax.set_xlabel("Time(s)", size=30)
        # ax.set_ylabel("Distance(m)", size=30)
        ax.grid()
        ax.set_xticklabels([x for x in range(0,time+1, 10)], fontsize=30)
        ax.set_yticklabels([y for y in range(0, 10, 2)], fontsize=30)
        ax.set_xlim(0, time+1)
        ax.set_ylim(0, 10)
        # ax.text(20, 9, 'Time={:.2f}s'.format(time_step * 0.05), size=20)
        plt.subplots_adjust(left=0.12,
                            bottom=0.1,
                            right=0.98,
                            top=0.98,
                            wspace=0.0,
                            hspace=0.0)
        plt.xticks(fontsize=30)
        ax.text(20, -1, 'Time(s)', va='center', fontsize=30)
        ax.text(-5, 5, "Distance(m)", va='center', rotation='vertical', fontsize=30)
        plt.pause(0.01)
        plt.cla()
        gc.collect(generation=2)
    plt.draw()
def distance_anime(path,time,dt):
    gc.enable()
    pose_array = np.load(os.path.join(path, "pose_array_scene.npy"))
    rob_num = np.shape(pose_array)[0]
    iters = iters = int(min(pose_array.shape[1],time/dt))
    gabriel_graph = gabriel(pose_array)
    distance_dict = {}
    for i in range(rob_num):
        for j in range(i + 1, rob_num):
            if gabriel_graph[i][j] == 0:
                continue
            name = str(i + 1) + " to " + str(j + 1)
            distance_array = np.sqrt(np.square(pose_array[i, :, 0] - pose_array[j, :, 0]) + np.square(
                pose_array[i, :, 1] - pose_array[j, :, 1]))
            distance_dict[name] = distance_array
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.pause(0.01)
    def plot_one(time_step):
        xlist=[i for i in range(time_step)]
        plt.cla()
        for key in distance_dict:
            # print(xlist,time_step)
            # print(distance_dict[key])
            plt.plot(xlist, distance_dict[key][:time_step], label=key)
            ax.legend(fontsize=20)
            # ax.set_title("Time histories of inter-robot distance", size=30)
            # ax.set_xlabel("Time(s)", size=30)
            # ax.set_ylabel("Distance(m)", size=30)
            ax.grid()
            ax.set_xticklabels([x for x in range(0, time + 1, 10)], fontsize=30)
            ax.set_yticklabels([y for y in range(0, 10, 2)], fontsize=30)
            ax.set_xlim(0, time_step + 1)
            ax.set_ylim(0, 10)
            plt.subplots_adjust(left=0.12,
                                bottom=0.1,
                                right=0.98,
                                top=0.98,
                                wspace=0.0,
                                hspace=0.0)
            plt.xticks(fontsize=30)
            fig.text(0.5, 0.03, 'Time(s)', va='center', fontsize=30)
            fig.text(0.03, 0.5, "Distance(m)", va='center', rotation='vertical', fontsize=30)


    anime=animation.FuncAnimation(plt.gcf(),func=plot_one,frames=np.arange(0,iters),interval=50)
    plt.show()
def velocity(path,time,dt):
    gc.enable()
    velocity_array=np.load(os.path.join(path,"velocity_array_scene.npy"))
    iters = int(min(velocity_array.shape[1],time/dt))
    rob_num = np.shape(velocity_array)[0]
    fig, ax = plt.subplots(rob_num, 1,figsize=(10, 10))
    plt.pause(3)
    xlist = []
    for time_step in range(iters):
        xlist.append(time_step*dt)
        colors = itertools.cycle(mcolors.TABLEAU_COLORS)
        for i in range(rob_num):
            color = next(colors)
            ax[i].plot(xlist, velocity_array[i, :time_step+1, 0], color=color,label="Robot " + str(i) + " left wheel speed")
            ax[i].plot(xlist, velocity_array[i, :time_step+1, 1], '--',color=color,label="Robot " + str(i) + " right wheel speed")
            ax[i].set_xlim(-2, time+2)
            ax[i].set_ylim(-1.5, 1.5)
            ax[i].set_xticklabels([x for x in range(-10,time+1,10) ],fontsize=30)
            ax[i].set_yticklabels([-2,-1,0,1],fontsize=20)
            # ax[i].set_xlabel('Time(s)',size=30)
            # ax[i].set_xlabel('Wheel control(m/s)', size=30)
            ax[i].grid()
        # ax.text(-1, 4, 'Time={:.2f}s'.format(time_step * 0.05), size=20)
        plt.subplots_adjust(left=0.12,
                            bottom=0.1,
                            right=0.98,
                            top=0.98,
                            wspace=0.0,
                            hspace=0.0)
        ax[rob_num-1].text(20, -3, 'Time(s)', va='center', fontsize=30)
        ax[rob_num-1].text(-8, 6, 'Wheel control(m/s)', va='center', rotation='vertical', fontsize=30)
        plt.pause(0.01)
        for i in range(rob_num):
            ax[i].cla()

        gc.collect(generation=2)
    plt.draw()


# def plot_triangle(ax,pose,length,color):
#     x=pose[0]
#     y=pose[1]
#     theta=pose[2]
#     p1=[x+2*length*math.cos(theta),y+2*length*math.sin(theta)]
#     p2=[x+length*math.cos(theta-2*math.pi/3),y+length*math.sin(theta-2*math.pi/3)]
#     p3 = [x + length * math.cos(theta + 2*math.pi / 3), y + length * math.sin(theta + 2*math.pi / 3)]
#     # ax.scatter(x,y,c=color)
#     ax.plot([p1[0],p2[0]],[p1[1],p2[1]],color=color)
#     ax.plot([p2[0],p3[0]],[p2[1],p3[1]],color=color)
#     ax.plot([p3[0],p1[0]],[p3[1],p1[1]],color=color)
# def triangle_trace(path,time,dt):
#     pose_array = np.load(os.path.join(path, "pose_array_scene.npy"))
#     rob_num = np.shape(pose_array)[0]
#     iters = int(min(pose_array.shape[1],time/dt))
#     fig, ax = plt.subplots(figsize=(10, 10))
#     plt.pause(1)
#     for time_step in range(iters):
#         colors = itertools.cycle(mcolors.TABLEAU_COLORS)
#         for i in range(rob_num):
#             color = next(colors)
#             xtrace = []
#             ytrace = []
#             for p in range(0, int(time_step/dt), 100):
#                 pose = pose_array[i][p]
#                 plot_triangle(ax, pose, 0.3, color)
#                 xtrace.append(pose_array[i][p][0])
#                 ytrace.append(pose_array[i][p][1])
#                 ax.plot(xtrace, ytrace, color=color, linestyle='--')
#         gabriel_graph = gabriel(pose_array)
#         position_array = pose_array[:, int(time_step/dt) - 1, :2]
#         for i in range(rob_num):
#             for j in range(i + 1, rob_num):
#                 if gabriel_graph[i][j] == 0:
#                     continue
#                 xlist = [position_array[i][0], position_array[j][0]]
#                 ylist = [position_array[i][1], position_array[j][1]]
#                 distance = math.sqrt((xlist[0] - xlist[1]) ** 2 + (ylist[0] - ylist[1]) ** 2)
#                 ax.plot(xlist, ylist, color="black")
#         ax.set_title("Formation", size=20)
#         ax.set_xlabel("X(m)", size=20)
#         ax.set_ylabel("Y(m)", size=20)
#         ax.grid()
#         ax.set_xlim(-5, 5)
#         ax.set_ylim(-5, 5)
#         ax.text(-1, 4, 'Time={:.2f}s'.format(time_step * 0.05), size=20)
#         plt.pause(0.05)
#         ax.cla()
#     plt.show()
formation(path,50,0.05)
# distance(path,50,0.05)
# velocity(path,50,0.05)
# triangle_trace(path,50,0.05)