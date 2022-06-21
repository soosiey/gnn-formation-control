import numpy as np
import matplotlib as plt
import os

root_dir="/home/xinchi/GNN-results/6_robots/expert_adjusted_6"
dest_dir="/home/xinchi/GNN-results/stop_results/expert_adjusted_6"
def adjust_data(root_dir,dest_dir):
    paths = os.walk(root_dir)
    path_list = []
    for path, dir_lst, file_lst in paths:
        for dir_name in dir_lst:
            path_list.append(os.path.join(path, dir_name))
            print(os.path.join(path, dir_name))
            pose_array = np.load(os.path.join(path, dir_name, "pose_array_scene.npy"))
            velocity_array = np.load(os.path.join(path, dir_name, "velocity_array_scene.npy"))

            pose_end=np.expand_dims(pose_array[:,-400,:],axis=1)
            pose_end=np.tile(pose_end,(100,1))
            new_pose_array=pose_array[:,:-400,:]
            new_pose_array=np.concatenate((new_pose_array,pose_end),axis=1)


            velocity_end=np.expand_dims(velocity_array[:,-400,:],axis=1)
            velocity_end=np.tile(velocity_end,(100,1))
            new_velocity_array=velocity_array[:,:-400,:]
            new_velocity_array=np.concatenate((new_velocity_array,velocity_end),axis=1)

            if not os.path.isdir(dest_dir) :
                os.mkdir(os.path.join(dest_dir))
            if not os.path.isdir(os.path.join(dest_dir, dir_name)):
                os.mkdir(os.path.join(dest_dir, dir_name))
            np.save(os.path.join(dest_dir,dir_name, "pose_array_scene.npy"),new_pose_array)
            np.save(os.path.join(dest_dir, dir_name, "velocity_array_scene.npy"), new_velocity_array)

adjust_data(root_dir,dest_dir)