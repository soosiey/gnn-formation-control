import numpy as np

pose_list=np.load("/home/xinchi/GNN-control/gnn-formation-control/results/expert/single_test/pose_array_scene.npy")
print(pose_list[:,0])
pose_list=np.load("/home/xinchi/GNN-control/gnn-formation-control/results/expert_adjusted/single_test/pose_array_scene.npy")
print(pose_list[:,0])