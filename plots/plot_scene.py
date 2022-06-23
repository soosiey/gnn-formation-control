import math

import matplotlib.pyplot  as plt
import matplotlib.colors as mcolors
import itertools
import scene
import numpy as np
import os
from matplotlib.animation import FuncAnimation

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
def plot_wheel_speed(dt,velocity_array,save_path):
    rob_num=np.shape(velocity_array)[0]
    xlist=[]
    colors=itertools.cycle(mcolors.TABLEAU_COLORS)
    for i in range(np.shape(velocity_array)[1]):
        xlist.append(i*dt)
    plt.figure(figsize=(5, 2))
    for i in range(rob_num):
        color=next(colors)
        plt.plot(xlist, velocity_array[i, :, 0], color=color, label="Robot " + str(i) + " left wheel speed")
        plt.plot(xlist, velocity_array[i, :, 1], '--', color=color, label="Robot " + str(i) + " right wheel speed")
    # plt.legend()
    plt.title("Wheel speeds")
    plt.xlabel("time(s)")
    plt.ylabel("velocity(m)")
    plt.grid()
    plt.savefig(os.path.join(save_path, "wheel_speed_" + str(rob_num) + ".png"),pad_inches=0.0)
    plt.close()
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
    plt.figure(figsize=(5, 2))
    for key in distance_dict:
        plt.plot(xlist,distance_dict[key],label=key)
    # plt.legend()
    plt.title("Relative distance")
    plt.xlabel("time(s)")
    plt.ylabel("distance(m)")
    plt.grid()
    plt.savefig(os.path.join(save_path, "relative_distance_" + str(rob_num) + ".png"),pad_inches=0.0)
    plt.close()
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
    plt.figure(figsize=(5, 2))
    for key in distance_dict:
        plt.plot(xlist,distance_dict[key],label=key)
    # plt.legend()
    plt.title("Relative distance gabreil",fontsize=15)
    plt.xlabel("time(s)",fontsize=15)
    plt.ylabel("distance(m)",fontsize=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid()
    plt.savefig(os.path.join(save_path, "relative_distance_gabreil_" + str(rob_num) + ".png"),pad_inches=0.0)
    plt.close()
    # plt.show()

def plot_formation_gabreil(pose_array,save_path):
    rob_num=np.shape(pose_array)[0]
    gabriel_graph = gabriel(pose_array)
    position_array = pose_array[:, -1, :2]
    print(pose_array.shape)
    plt.figure(figsize=(10, 10))
    plt.scatter(position_array[:,0],position_array[:,1],s=100)
    for i in range(rob_num):
        for j in range(i + 1, rob_num):
            if gabriel_graph[i][j] == 0:
                continue
            xlist = [position_array[i][0],position_array[j][0]]
            ylist = [position_array[i][1], position_array[j][1]]
            distance=math.sqrt((xlist[0]-xlist[1])**2+(ylist[0]-ylist[1])**2)
            plt.plot(xlist,ylist,label="Distane: {d:.2f}".format(d=distance),linewidth=2.5)
    # plt.legend(fontsize=30)
    plt.title("Formation",fontsize=15)
    plt.xlabel("distance(m)",fontsize=15)
    plt.ylabel("distance(m)",fontsize=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.grid()
    plt.savefig(os.path.join(save_path, "formation_gabreil_" + str(rob_num) + ".png"),pad_inches=0.0)
    plt.close()
    # plt.show()
def plot_trace(position_array,save_path):
    rob_num=np.shape(position_array)[0]

    colors=itertools.cycle(mcolors.TABLEAU_COLORS)

    plt.figure(figsize=(10, 10))
    for i in range(rob_num):
        color=next(colors)
        for p in range(np.shape(position_array)[1]):
            plt.scatter(position_array[i][p][0],position_array[i][p][1],s=10,c=color)
        plt.scatter(position_array[i][0][0], position_array[i][0][1], s=150, c=color,marker="x")
    # plt.legend()
    plt.title("Trace")
    plt.xlabel("x(m)")
    plt.ylabel("y(m)")
    plt.grid()
    plt.savefig(os.path.join(save_path, "robot_trace_" + str(rob_num) + ".png"))
    plt.close()
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
    plot_trace(pose_array,save_path)

# pose_array=np.load(os.path.join("..","pose_array_scene.npy"))
# velocity_array=np.load(os.path.join("..","velocity_array_scene.npy"))
def plot_load_scene(dt,dir):
    save_path=os.path.join(dir)
    pose_array = np.load(os.path.join(dir, "pose_array_scene.npy"))
    velocity_array = np.load(os.path.join(dir, "velocity_array_scene.npy"))

    plot_relative_distance(dt,pose_array,save_path)
    plot_relative_distance_gabreil(dt, pose_array,save_path)
    plot_wheel_speed(dt, velocity_array,save_path)
    plot_formation_gabreil(pose_array,save_path)
    # plot_trace(pose_array,save_path)
def plot_dynamic_gabreil(path):
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(0, 10 * np.pi, 100)
    y = np.sin(x)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1, = ax.plot(x, y, 'b-')

    for phase in np.linspace(0, 10 * np.pi, 100):
        line1.set_ydata(np.sin(0.5 * x + phase))
        fig.canvas.draw()
        fig.canvas.flush_events()
if __name__=="__main__":
    # for i in range(0,100):
    plot_load_scene(0.05,"/home/xinchi/GNN-results/stop_results/4_robots/model_4/"+str(0))
