import matplotlib.pyplot as plt
import numpy as np
import os
import math
# # Creating dataset
# np.random.seed(10)
# data = np.random.normal(100, 20, 200)
#
# fig = plt.figure(figsize=(10, 7))
#
# # Creating plot
# plt.boxplot(data)
#
# # show plot
# plt.show()

def gabriel(pose_array):
    node_mum = np.shape(pose_array)[0]
    gabriel_graph=[[1]*node_mum for _ in range(node_mum)]
    position_array=pose_array[:,:2]
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
def get_convergence_time(raw_data,desired_distance=2,tolerrance=0.1,check_time=5):
    time_steps=raw_data.shape[1]
    realstop = 0
    for time_step in range(time_steps):
        data=raw_data[:,time_step,:]
        gabriel_graph = gabriel(data)
        stop=True
        for i in range(len(gabriel_graph)):
            for j in range(i, len(gabriel_graph)):
                if not i == j:
                    if gabriel_graph[i][j] == 1:
                        distance = ((data[i,0] - data[j,0])**2 + (data[i, 1] - data[j,1])**2)**0.5
                        if math.fabs(distance-desired_distance)/desired_distance>tolerrance:
                            stop=False
        if stop:
            realstop+=1
        else:
            realstop=0
        if realstop>=check_time*20:
            break
        # print(realstop)
    return time_step/20
def process_data(dir):
    paths = os.walk(dir)
    path_list=[]
    for path, dir_lst, file_lst in paths:
        for dir_name in dir_lst:

            path_list.append(os.path.join(path, dir_name))
    converge_time_all=[]
    average_formation_all=[]
    average_formation_error_all=[]
    unsuccess=0
    for path in path_list:
        file=os.path.join(path, "pose_array_scene.npy")
        raw_data=np.load(file)
        sim_time=raw_data.shape[1]*0.05
        convergence_time = get_convergence_time(raw_data)
        if convergence_time >= 50:
            unsuccess += 1
            print(path)
            continue
        observe_data=raw_data[:,-100:,:2]
        time_steps=observe_data.shape[1]
        for time_step in range(time_steps):
            data=observe_data[:,time_step,:]
            gabriel_graph=gabriel(data)
            reference=np.ones(data.shape[1])
            reference=reference*2
            distance_error_list=[]
            distance_list=[]
            for i in range(len(gabriel_graph)):
                for j in range(i,len(gabriel_graph)):
                    if not i==j:
                        if gabriel_graph[i][j]==1:
                            distance=np.sqrt(np.square(data[i,0]-data[j,0])+np.square(data[i,1]-data[j,1]))
                            # print(distance)
                            distance_list.append(distance)
                            distance_error=np.abs(distance-reference)
                            distance_error_list.append(distance_error)
        average_formation = np.average(np.array(distance_list))
        average_formation_error = np.average(np.array(distance_error_list)) / 2
        converge_time_all.append(convergence_time)
        average_formation_error_all.append(average_formation_error)
        average_formation_all.append(average_formation)
    print(dir,unsuccess)
    return converge_time_all,average_formation_all,average_formation_error_all
    # print(observe_data-reference)

def box(data_m,data_e,title,ylabel,save_dir):
    fig = plt.figure(figsize=(6, 2))
    labels=[4,5,6]
    color_model='#1f77b4'
    color_expert='#ff7f0e'
    model=plt.boxplot(data_m,
                      positions=np.array(range(len(data_m))) *2.0+0.4,
                      boxprops=dict(color=color_model),
                      capprops=dict(color=color_model),
                      whiskerprops=dict(color=color_model),
                      flierprops=dict(color=color_model,markeredgecolor=color_model),
                      medianprops=dict(color="black"),
                      widths=0.6)
    exp=plt.boxplot(data_e,
                      positions=np.array(range(len(data_e))) *2.0-0.4,
                      boxprops=dict(color=color_expert),
                      capprops=dict(color=color_expert),
                      whiskerprops=dict(color=color_expert),
                      flierprops=dict(color=color_expert,markeredgecolor=color_expert),
                      medianprops=dict(color="black"),
                      widths=0.6)
    plt.legend([model["boxes"][0], exp["boxes"][0]], ['GNN', 'Expert'], loc='upper left',borderpad=0.2,labelspacing=0.2)
    plt.xticks(np.array(range(len(data_m)))*2.0,labels=labels)
    plt.title(title,fontsize=12)
    plt.xlabel("Number of robots",fontsize=12)
    plt.ylabel(ylabel,fontsize=12)
    plt.savefig(os.path.join(save_dir,title+'.png'))



dir4= '/home/xinchi/GNN-results-50/model_4'
converge_time_all_4,average_formation_all_4,average_formation_error_all_4=process_data(dir4)
dir5= '/home/xinchi/GNN-results-50/model_5'
converge_time_all_5,average_formation_all_5,average_formation_error_all_5=process_data(dir5)
dir6= '/home/xinchi/GNN-results-50/model_6'
converge_time_all_6,average_formation_all_6,average_formation_error_all_6=process_data(dir6)
converge_time_all_model=[converge_time_all_4,converge_time_all_5,converge_time_all_6]
average_formation_all_model=[average_formation_all_4,average_formation_all_5,average_formation_all_6]
average_formation_error_all_model=[average_formation_error_all_4,average_formation_error_all_5,average_formation_error_all_6]


# dir= '/home/xinchi/6_robots/model_6'
dir4_e= '/home/xinchi/GNN-results-50/expert_adjusted_4'
converge_time_all_4_e,average_formation_all_4_e,average_formation_error_all_4_e=process_data(dir4_e)
dir5_e= '/home/xinchi/GNN-results-50/expert_adjusted_5'
converge_time_all_5_e,average_formation_all_5_e,average_formation_error_all_5_e=process_data(dir5_e)
dir6_e= '/home/xinchi/GNN-results-50/expert_adjusted_6'
converge_time_all_6_e,average_formation_all_6_e,average_formation_error_all_6_e=process_data(dir6_e)
converge_time_all_expert=[converge_time_all_4_e,converge_time_all_5_e,converge_time_all_6_e]
average_formation_all_expert=[average_formation_all_4_e,average_formation_all_5_e,average_formation_all_6_e]
average_formation_error_all_expert=[average_formation_error_all_4_e,average_formation_error_all_5_e,average_formation_error_all_6_e]


box(converge_time_all_model,converge_time_all_expert,"Converge time","Time(s)","/home/xinchi/GNN-results-50")
box(average_formation_all_model,average_formation_all_expert,"Average distance","Distance(m)","/home/xinchi/GNN-results-50")
box(average_formation_error_all_model,average_formation_error_all_expert,"Average group formation error","percentage(%)","/home/xinchi/GNN-results-50")