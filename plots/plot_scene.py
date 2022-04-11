import math

import matplotlib.pyplot  as plt
import matplotlib.colors as mcolors
import itertools
import scene
import numpy as np
import os

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
                    gabriel_graph[v][u] = 0
                    break
    return gabriel_graph
def plot_wheel_speed(dt,velocity_array,save_path):
    rob_num=np.shape(velocity_array)[0]
    xlist=[]
    colors=itertools.cycle(mcolors.TABLEAU_COLORS)
    for i in range(np.shape(velocity_array)[1]):
        xlist.append(i*dt)
    plt.figure(figsize=(10, 10))
    for i in range(rob_num):
        color=next(colors)
        plt.plot(xlist, velocity_array[i, :, 0], color=color, label="Robot " + str(i) + " left wheel speed")
        plt.plot(xlist, velocity_array[i, :, 1], '--', color=color, label="Robot " + str(i) + " right wheel speed")
    # plt.legend()
    plt.title("Wheel speeds")
    plt.xlabel("time(s)")
    plt.ylabel("velocity(m)")
    plt.grid()
    plt.savefig(os.path.join(save_path, "wheel_speed_" + str(rob_num) + ".png"))
    # plt.show()
def plot_relative_distance(dt,pose_array,save_path):
    rob_num=np.shape(pose_array)[0]
    distance_dict={}
    xlist = []
    for i in range(np.shape(pose_array)[1]):
        xlist.append(i * dt)
    for i in range(rob_num):
        for j in range(i+1,rob_num):
            name=str(i+1)+" to "+str(j+1)
            distance_array=np.sqrt(np.square(pose_array[i,:,0]-pose_array[j,:,0])+np.square(pose_array[i,:,1]-pose_array[j,:,1]))
            distance_dict[name]=distance_array
    plt.figure(figsize=(10, 10))
    for key in distance_dict:
        plt.plot(xlist,distance_dict[key],label=key)
    # plt.legend()
    plt.title("Relative distance")
    plt.xlabel("time(s)")
    plt.ylabel("distance(m)")
    plt.grid()
    plt.savefig(os.path.join(save_path, "relative_distance_" + str(rob_num) + ".png"))
    # plt.show()
    # fig = plt.gcf()
    # fig.savefig(os.path.join(save_path, "relative_distance_" + str(rob_num) + ".png"))
def plot_relative_distance_gabreil(dt,pose_array,save_path):
    rob_num = np.shape(pose_array)[0]
    gabriel_graph=gabriel(pose_array)
    distance_dict={}
    xlist = []
    for i in range(np.shape(pose_array)[1]):
        xlist.append(i * dt)
    for i in range(rob_num):
        for j in range(i+1,rob_num):
            if gabriel_graph[i][j]==0:
                continue
            name=str(i+1)+" to "+str(j+1)
            distance_array=np.sqrt(np.square(pose_array[i,:,0]-pose_array[j,:,0])+np.square(pose_array[i,:,1]-pose_array[j,:,1]))
            distance_dict[name]=distance_array
    plt.figure(figsize=(10, 5))
    for key in distance_dict:
        plt.plot(xlist,distance_dict[key],label=key)
    # plt.legend()
    plt.title("Relative distance gabreil")
    plt.xlabel("time(s)")
    plt.ylabel("distance(m)")
    plt.grid()
    plt.savefig(os.path.join(save_path, "relative_distance_gabreil_" + str(rob_num) + ".png"))
    # plt.show()

def plot_formation_gabreil(pose_array,save_path):
    rob_num=np.shape(pose_array)[0]
    gabriel_graph = gabriel(pose_array)
    position_array = pose_array[:, -1, :2]
    plt.figure(figsize=(10, 10))
    plt.scatter(position_array[:,0],position_array[:,1])
    for i in range(rob_num):
        for j in range(i + 1, rob_num):
            if gabriel_graph[i][j] == 0:
                continue
            xlist = [position_array[i][0],position_array[j][0]]
            ylist = [position_array[i][1], position_array[j][1]]
            distance=math.sqrt((xlist[0]-xlist[1])**2+(ylist[0]-ylist[1])**2)
            plt.plot(xlist,ylist,label="Distane: {d:.2f}".format(d=distance))
    plt.legend()
    plt.title("Formation")
    plt.xlabel("distance(m)")
    plt.ylabel("distance(m)")
    plt.grid()
    plt.savefig(os.path.join(save_path, "formation_gabreil_" + str(rob_num) + ".png"))
    # plt.show()

def plot_scene(sc,path="",save_path=""):
    if not path=="":
        pose_array=np.load(os.path.join(path,"pose_array_scene.npy"))
        velocity_array=np.load(os.path.join(path,"velocity_array_scene.npy"))
    else:
        pose_list = []
        velocity_list = []
        for i in range(len(sc.robots)):
            pose_list.append(sc.robots[i].pose_list)
            velocity_list.append(sc.robots[i].velocity_list)
        pose_array = np.array(pose_list)
        velocity_array = np.array(velocity_list)
    dt=sc.dt
    # for root, dirs, files in os.walk(save_path, topdown=False):
    #     folders=0
    #     for name in dirs:
    #        folders+=1
    # os.makedirs(os.path.join(save_path,str(folders)))if
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plot_relative_distance(dt,pose_array,save_path)
    plot_relative_distance_gabreil(dt, pose_array,save_path)
    plot_wheel_speed(dt, velocity_array,save_path)
    plot_formation_gabreil(pose_array,save_path)

# pose_array=np.load(os.path.join("..","pose_array_scene.npy"))
# velocity_array=np.load(os.path.join("..","velocity_array_scene.npy"))
# sc=scene
# plot_scene(sc,path="..",save_path="../fig")