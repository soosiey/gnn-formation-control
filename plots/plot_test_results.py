import matplotlib.pyplot as plt
import numpy as np
import os
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
def process_data(dir):
    paths = os.walk(dir)
    path_list=[]
    for path, dir_lst, file_lst in paths:
        for dir_name in dir_lst:
            print(os.path.join(path, dir_name))
            path_list.append(os.path.join(path, dir_name))
    converge_time_all=[]
    average_formation_all=[]
    average_formation_error_all=[]
    for path in path_list:
        file=os.path.join(path, "pose_array_scene.npy")
        raw_data=np.load(file)
        sim_time=raw_data.shape[1]*0.05
        observe_data=raw_data[:,-400:,:2]
        gabriel_graph=gabriel(observe_data)
        reference=np.ones(observe_data.shape[1])
        reference=reference*2
        distance_error_list=[]
        distance_list=[]
        for i in range(len(gabriel_graph)):
            for j in range(i,len(gabriel_graph)):
                if not i==j:
                    if gabriel_graph[i][j]==1:
                        distance=np.sqrt(np.square(observe_data[i,:,0]-observe_data[j,:,0])+np.square(observe_data[i,:,1]-observe_data[j,:,1]))
                        # print(distance)
                        distance_list.append(distance)
                        distance_error=np.abs(distance-reference)
                        distance_error_list.append(distance_error)
        average_formation_error=np.average(np.array(distance_error_list))
        average_formation = np.average(np.array(distance_list))
        converge_time_all.append(sim_time-20)
        average_formation_error_all.append(average_formation_error)
        average_formation_all.append(average_formation)
    print(average_formation_all)
    print(average_formation_error_all)
    print(converge_time_all)
    return converge_time_all,average_formation_all,average_formation_error_all
    # print(observe_data-reference)
def box(data,title,save_dir):
    fig = plt.figure(figsize=(10, 7))
    plt.boxplot(data)
    plt.title(title)
    plt.savefig(os.path.join(save_dir,title+'.png'))

dir= '../results/2022.5.8 no dagger/model_5'
converge_time_all,average_formation_all,average_formation_error_all=process_data(dir)
box(converge_time_all,"converge time",dir)
box(average_formation_all,"average distance",dir)
box(average_formation_error_all,"average distance error",dir)