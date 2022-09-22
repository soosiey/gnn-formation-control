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

    plt.grid()
    plt.subplots_adjust(left=0.15,
                        bottom=0.25,
                        right=0.99,
                        top=0.99,
                        wspace=0.0,
                        hspace=0.0)
    plt.xlabel("time(s)", fontsize=15)
    plt.ylabel("speed(m/s)", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(os.path.join(save_path, "wheel_speed_" + str(rob_num) + ".png"),pad_inches=0.0)
    plt.close()
    # plt.show()
def plot_wheel_speed_super(dt,velocity_array,save_path):
    rob_num=np.shape(velocity_array)[0]
    xlist=[]
    colors=itertools.cycle(mcolors.TABLEAU_COLORS)
    for i in range(np.shape(velocity_array)[1]):
        xlist.append(i*dt)
    fig, ax = plt.subplots(rob_num, 1,figsize=(6,6))
    for i in range(rob_num):
        color=next(colors)
        ax[i].plot(xlist, velocity_array[i, :, 0], color=color, label="Robot " + str(i) + " left wheel speed")
        ax[i].plot(xlist, velocity_array[i, :, 1], '--', color=color, label="Robot " + str(i) + " right wheel speed")
        ax[i].set_ylim(-1.5,1.5)
        ax[i].grid()
    # plt.legend()

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.98,
                        top=0.98,
                        wspace=0.0,
                        hspace=0.0)
    plt.xlabel("time(s)", fontsize=15)
    plt.xticks(fontsize=15)
    fig.text(0.01, 0.5, 'wheel control(m/s)', va='center', rotation='vertical',fontsize=15)
    plt.savefig(os.path.join(save_path, "wheel_speed_super_" + str(rob_num) + ".png"),pad_inches=0.0)
    plt.close()
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
    plt.figure(figsize=(5, 3))
    for key in distance_dict:
        plt.plot(xlist,distance_dict[key],label=key)
    # plt.legend()
    plt.xlabel("time(s)",fontsize=15)
    plt.ylabel("distance(m)",fontsize=15)
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
    fig, ax = plt.subplots(figsize=(5, 3))
    for key in distance_dict:
        ax.plot(xlist,distance_dict[key],label=key)
    # plt.legend()
    plt.subplots_adjust(left=0.15,
                        bottom=0.20,
                        right=0.98,
                        top=0.98,
                        wspace=0.0,
                        hspace=0.0)
    plt.xlabel("time(s)",fontsize=20)
    plt.ylabel("distance(m)",fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid()
    plt.savefig(os.path.join(save_path, "relative_distance_gabreil_" + str(rob_num) + ".png"),pad_inches=0.0)
    plt.close()
    # plt.show()

def plot_formation_gabreil(pose_array,save_path):
    rob_num=np.shape(pose_array)[0]
    gabriel_graph = gabriel(pose_array)
    position_array = pose_array[:, -1, :2]
    print(pose_array.shape)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(position_array[:,0],position_array[:,1],s=100)
    for i in range(rob_num):
        for j in range(i + 1, rob_num):
            if gabriel_graph[i][j] == 0:
                continue
            xlist = [position_array[i][0],position_array[j][0]]
            ylist = [position_array[i][1], position_array[j][1]]
            distance=math.sqrt((xlist[0]-xlist[1])**2+(ylist[0]-ylist[1])**2)
            ax.plot(xlist,ylist,label="Distane: {d:.2f}".format(d=distance),linewidth=3)
    # plt.legend(fontsize=30)
    plt.subplots_adjust(left=0.16,
                        bottom=0.16,
                        right=0.95,
                        top=0.98,
                        wspace=0.0,
                        hspace=0.0)
    plt.xlabel("x(m)",fontsize=20)
    plt.ylabel("y(m)",fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.grid()
    plt.savefig(os.path.join(save_path, "formation_gabreil_" + str(rob_num) + ".png"),pad_inches=0.0)
    plt.close()
    # plt.show()

def plot_triangle(ax,pose,length,color):
    x=pose[0]
    y=pose[1]
    theta=pose[2]
    p1=[x+2*length*math.cos(theta),y+2*length*math.sin(theta)]
    p2=[x+length*math.cos(theta-2*math.pi/3),y+length*math.sin(theta-2*math.pi/3)]
    p3 = [x + length * math.cos(theta + 2*math.pi / 3), y + length * math.sin(theta + 2*math.pi / 3)]
    # ax.scatter(x,y,c=color)
    ax.plot([p1[0],p2[0]],[p1[1],p2[1]],color=color)
    ax.plot([p2[0],p3[0]],[p2[1],p3[1]],color=color)
    ax.plot([p3[0],p1[0]],[p3[1],p1[1]],color=color)
def plot_trace_triangle(pose_array,save_path,stop_time):
    rob_num = np.shape(pose_array)[0]
    colors = itertools.cycle(mcolors.TABLEAU_COLORS)
    fig,ax=plt.subplots(figsize=(5, 5))
    for i in range(rob_num):
        color = next(colors)
        xtrace = []
        ytrace = []
        for p in range(0,stop_time*20,100):
            pose=pose_array[i][p]
            plot_triangle(ax, pose, 0.3, color)
            xtrace.append(pose_array[i][p][0])
            ytrace.append(pose_array[i][p][1])
            ax.plot(xtrace,ytrace,color=color,linestyle='--')
    gabriel_graph = gabriel(pose_array)
    position_array = pose_array[:, stop_time*20-1, :2]
    for i in range(rob_num):
        for j in range(i + 1, rob_num):
            if gabriel_graph[i][j] == 0:
                continue
            xlist = [position_array[i][0], position_array[j][0]]
            ylist = [position_array[i][1], position_array[j][1]]
            distance = math.sqrt((xlist[0] - xlist[1]) ** 2 + (ylist[0] - ylist[1]) ** 2)
            ax.plot(xlist, ylist,color="black")

    plt.subplots_adjust(left=0.13,
                        bottom=0.11,
                        right=0.98,
                        top=0.98,
                        wspace=0.0,
                        hspace=0.0)
    plt.xlabel("x(m)", fontsize=15)
    plt.ylabel("y(m)", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.grid()
    plt.savefig(os.path.join(save_path, "robot_trace_" + str(rob_num) + ".png"))
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
def plot_load_scene(time,dt,dir):
    save_path=os.path.join(dir)
    pose_array = np.load(os.path.join(dir, "pose_array_scene.npy"))
    velocity_array = np.load(os.path.join(dir, "velocity_array_scene.npy"))

    plot_relative_distance(dt,pose_array[:,:int(time/dt),:],save_path)
    plot_relative_distance_gabreil(dt, pose_array[:,:int(time/dt),:],save_path)
    plot_wheel_speed(dt, velocity_array[:,:int(time/dt),:],save_path)
    plot_wheel_speed_super(dt, velocity_array[:,:int(time/dt),:], save_path)
    plot_formation_gabreil(pose_array[:,:int(time/dt),:],save_path)
    plot_trace_triangle(pose_array[:,:int(time/dt),:], save_path, time)

if __name__=="__main__":
    # for i in range(0,100):
    plot_load_scene(50,0.05,"../results/model_8/demo")


