# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 23:09:34 2017
This test file is dependent on vrep.
To run this file, please open vrep file scene/scene_double.ttt first
@author: cz
"""
import os
from scene import Scene
from scene import VrepError
from sceneplot import ScenePlot
# from robot import Robot
import numpy as np
import math
import random
# from data import Data
#from DeepFCL import DeepFCL
from suhaas_agent import Agent
from plots.plot_scene import plot_scene
import torch
import saver
import time
import matplotlib.pyplot as plt


import argparse
parser = argparse.ArgumentParser(description='Args for demo')

parser.add_argument('--expert_only', dest='expert_only', default=False,type=bool,help='Use expert control only')
parser.add_argument('--use_dagger', dest='use_dagger', default=True,type=bool,help='Use dagger for training only')
parser.add_argument('--if_train', dest='if_train', default=True,type=bool,help='Control demo mod(train/test)')
parser.add_argument('--if_continue', dest='if_continue', default=True,type=bool,help='Continue training')
parser.add_argument('--use_cuda', dest='use_cuda', default=True,type=bool,help='Use cuda')
parser.add_argument('--expert_velocity_adjust', dest='expert_velocity_adjust', default=False,type=bool,help=' Adjust controller output accoring to the ralative distance output when using expert control')
parser.add_argument('--model_path', dest='model_path', default='/home/xinchi/GNN-control/gnn-formation-control/models/seperate_dagger_0.9_50_2022.5.17',type=str,help='Path to save model')
parser.add_argument('--model_name', dest='model_name', default='model_train_episode-0_robot-5.pth',type=str,help='Name of model')
parser.add_argument('--robot_num', dest='robot_num', default=5,type=int,help='Number of robot for simulation')
parser.add_argument('--position_range', dest='position_range', default=5,type=int,help='Set robots position within the range')
parser.add_argument('--sim_dt', dest='sim_dt', default=0.05,type=float,help='Simulation time step')
parser.add_argument('--sim_time', dest='sim_time', default=10,type=float,help='Simulation time for one simulation')
parser.add_argument('--stop_thresh', dest='stop_thresh', default=0.05,type=float,help='Stopping thresh')
parser.add_argument('--stop_waiting_time', dest='stop_waiting_time', default=0.0,type=float,help='Stopping after this time')
parser.add_argument('--desire_distance', dest='desire_distance', default=2.0,type=float,help='Desire formation distance')
parser.add_argument('--train_episode', dest='train_episode', default=1,type=int,help='Episode for training')
parser.add_argument('--batch_size', dest='batch_size', default=16,type=int,help='Batch size for training')
parser.add_argument('--iter', dest='iter', default=1,type=int,help='Iter for testing multiple round')
parser.add_argument('--inW', dest='inW', default=100,type=int,help='Dataset shape')
parser.add_argument('--inH', dest='inH', default=100,type=int,help='Dataset shape')
parser.add_argument('--save_iteration', dest='save_iteration', default=10,type=int,help='Save after certain iterations')
parser.add_argument('--saved_figs', dest='saved_figs', default="results",type=str,help='Save after certain iterations')
# # modelname='model_'+str(robot_num)+'robots_'+str(simTime)+'s_'+str(trainEpisode)+'rounds'+'.pth'
args = parser.parse_args()



def Train(args,model_iter):
    #### store robot pose
    numRun = args.train_episode if args.if_train else 1
    #### training data will be stored
    dataList = []
    lossList = []
    sc = None
    #### Initial Agent
    fcl = Agent(batch_size=args.batch_size,inW=args.inW, inH=args.inH, nA=args.robot_num,cuda=args.use_cuda)
    if (not args.if_train):
        # model_name="suhaas_model_v13_dagger_final_more.pth"
        model_name = args.model_name
        fcl.model.to('cpu')
        fcl.model.load_state_dict(torch.load(os.path.join(args.model_path,model_name)))
        fcl.model.to('cuda')
    episode = 0
    if (args.if_continue):
        for path,dir,file_list in os.walk(args.model_path):
            for file in file_list:
                if file.split("_")[0]=="last":
                    episode=int(file.split("_")[1].split(".")[0])
                    model_name=file
        fcl.model.to('cpu')
        model_name='model_train_episode-'+str(model_iter)+'_robot-5.pth'
        fcl.model.load_state_dict(torch.load(os.path.join(args.model_path,model_name)))
        fcl.model.to('cuda')
        print('Loaded model')
    ##### code for training
    for i in range(episode,numRun):
        # print(lossList)
        print('Episode:', i + 1)
        ##########################################################################
        ######## Step 1: Start simulation rollouts to get training data ##########
        ##########################################################################
        dataListEpisode = []
        # First episode
        sc = generate_scene(args.sim_dt,i,args.robot_num,args.if_train,args.expert_only,args.use_dagger,args.sim_time,args.position_range,
                        args.desire_distance,args.stop_thresh,args.expert_velocity_adjust,agent=fcl)
        sc0 = simulate(args,sc)
        sc0.save_robot_states(os.path.join(args.saved_figs,str(args.stop_thresh),str(args.iter)))
        plot_scene(sc0,"",os.path.join(args.saved_figs,"train",str(args.stop_thresh),str(args.iter)))
        if sc0 is not None:
            # if the list is not empty
            sc = sc0
            saver.save(sc)  # save data
            for robot in sc.robots:
                dataListEpisode.append(robot.data)
            if not dataList:
                for robot in sc.robots:
                    dataList.append(robot.data)
            else:
                for j in range(len(sc.robots)):
                    dataList[j].append(sc.robots[j].data)
                    dataListEpisode[j].append(sc.robots[j].data)
        for j in range(1, len(sc.robots)):
            dataListEpisode[0].append(dataList[j])

        ##########################################################################
        ##### Step 2: Start training the NN model (e.g., supervised learning) #####
        ##########################################################################
        if (args.if_train):
            start=time.time()
            l = fcl.train(dataListEpisode)
            end=time.time()
            print(time)
            print(end-start)
            lossList.append(l)
            with open('document.csv', 'a') as fd:
                fd.write(str(model_iter)+","+str(l)+"\n")
        if (i % args.save_iteration == 0 and args.if_train):
            if not os.path.exists(args.model_path):
                os.makedirs(args.model_path)
            saved_model_name="model_train"+"_episode-"+str(i)+"_robot-"+str(args.robot_num)+".pth"
            saved_model=os.path.join(args.model_path,saved_model_name)
            time_file=os.path.join(args.model_path,"time.txt")
            fcl.save(saved_model)
            with open(time_file,'a+') as time_log:
                line=saved_model_name+","+str(time.asctime(time.localtime(time.time())))+"\n"
                time_log.write(line)

            for path, dir, file_list in os.walk(args.model_path):
                for file in file_list:
                    if file.split("_")[0] == "last":
                        os.remove(os.path.join(path,file))
            last_model_name="last_"+str(i)+".pth"
            last_model=os.path.join(args.model_path, last_model_name)
            fcl.save(last_model)
    saved_model_name = "model_train" + "_episode-" + str(numRun) + "_robot-" + str(args.robot_num) + ".pth"
    saved_model = os.path.join(args.model_path, saved_model_name)
    fcl.save(saved_model)
    time_file = os.path.join(args.model_path, "time.txt")
    with open(time_file, 'a+') as time_log:
        line = saved_model_name + "," + str(time.asctime(time.localtime(time.time()))) + "\n"
        time_log.write(line)
    if sc:
        print('data stored')
        print(sc.dt)
        for j in range(1, len(sc.robots)):
            dataList[0].append(dataList[j])
        dataList[0].store()
    else:
        print('data not stored')


def initRef(sc):

    # set desired velocity vector
    sc.xid.vRefMag = 0.7
    sc.xid.vRefAng = 2 * math.pi * random.random()#0.982793723# 2 * math.pi * random.random()
    sc.xid.theta = 0
    sc.xid.sDot = 0
    sc.xid.thetaDot = 0
    # scale desired formation separation
    #alphaList = [1.0, 1.5, 2.0]
    #alphaList = [1.0,2.0,3.0,4.0,4.5]
    alphaList = [2.0]
    alpha = random.choice(alphaList)
    sc.scaleDesiredFormation(alpha)
    message = "vRefMag: {0:.3f}, vRefAng: {1:.3f}, alpha: {2:.3f}"
    message = message.format(sc.xid.vRefMag, sc.xid.vRefAng, alpha)
    sc.log(message)
    print(message)

def plot(sp, tf,expert): #sp.plot(0, tf) sp.plot(2, tf) # Formation Separation
    sp.plot(2,tf,expert=expert)
    sp.plot(3, tf, expert=expert)
    sp.plot(4, tf,expert=expert)
    sp.plot(5, tf,expert=expert)
    sp.plot(6, tf,expert=expert)
    sp.plot(7, tf, expert=expert)
    sp.plot(8, tf, expert=expert)
    sp.plot(9, tf, expert=expert)

def generate_scene(dt,num_run,robot_num,if_train,expert_only,use_dagger,sim_time,position_range,
                   desired_distance,stop_thresh,expert_velocity_adjust,agent):

    sc = Scene(dt,num_run,robot_num,if_train,expert_only,use_dagger,desired_distance,stop_thresh,expert_velocity_adjust,fileName=__file__, recordData=True)
    sp = ScenePlot(sc)
    sp.saveEnabled = True  # save plots?
    sc.occupancyMapType = sc.OCCUPANCY_MAP_BINARY
    # sc.dynamics = 18 # robot dynamics
    sc.errorType = 0
    for i in range(robot_num):
        sc.addRobot(np.float32([[-2, 0, 1], [0.0, 0.0, 0.0]]),learnedController=agent.test)
    # No leader
    I = np.identity(robot_num, dtype=np.int8)
    M = np.ones(robot_num, dtype=np.int8)
    sc.setADjMatrix(M - I)

    # Set robot 0 as the leader.desired_distance,expert_velocity_adjust

    # vrep related
    sc.initVrep()
    # Choose sensor type
    # sc.SENSOR_TYPE = "VPL16" # None, 2d, VPL16, kinect
    sc.SENSOR_TYPE = "VPL16"  # None, 2d, VPL16, kinect
    sc.objectNames = ['Pioneer_p3dx', 'Pioneer_p3dx_leftMotor', 'Pioneer_p3dx_rightMotor']
    # change the # of instantiations according to "robot_num"
    # print(sc.SENSOR_TYPE)
    if sc.SENSOR_TYPE == "None":
        sc.setVrepHandles(0, '')
        for i in range(1, robot_num + 1):
            sc.setVrepHandles(i, '#' + str(i))
        # sc.setVrepHandles(1, '#0')

    elif sc.SENSOR_TYPE == "VPL16":
        sc.objectNames.append('velodyneVPL_16')  # _ptCloud
        print(sc.objectNames)
        for i in range(robot_num):
            checkn = i - 1
            s = ''
            if (i >= 1):
                s += '#' + str(checkn)
            sc.setVrepHandles(i, s)

    # sc.renderScene(waitTime = 3000)
    tf = sim_time  ## must lager than 3

    CheckerEnabled = False
    initRef(sc)  # sc.resetPosition(robot_num*np.sqrt(2)) # Random initial position
    sc.resetPosition(position_range)
    return sc



def simulate(args,sc):
    # if(args.if_continue):
    #     ep -=args.train_episode
    # sc = Scene(fileName = __file__, recordData = True)
    # sc.runNum=ep
    # sp = ScenePlot(sc)
    # sp.saveEnabled = True # save plots?
    # # global numRun
    # # global positionList
    # #sc.occupancyMapType = sc.OCCUPANCY_MAP_THREE_CHANNEL
    # sc.occupancyMapType = sc.OCCUPANCY_MAP_BINARY
    # # sc.dynamics = 18 # robot dynamics
    # sc.errorType = 0

    try:
        # for i in range(args.robot_num):
        #     sc.addRobot(np.float32([[-2, 0, 1], [0.0, 0.0, 0.0]]),args, learnedController = agent.test)
        # # No leader
        # I = np.identity(args.robot_num, dtype=np.int8)
        # M = np.ones(args.robot_num, dtype=np.int8)
        # sc.setADjMatrix(M-I)
        #
        # # Set robot 0 as the leader.
        #
        # # vrep related
        # sc.initVrep()
        # # Choose sensor type
        # #sc.SENSOR_TYPE = "VPL16" # None, 2d, VPL16, kinect
        # sc.SENSOR_TYPE = "VPL16" # None, 2d, VPL16, kinect
        # sc.objectNames = ['Pioneer_p3dx', 'Pioneer_p3dx_leftMotor', 'Pioneer_p3dx_rightMotor']
        #
        # # change the # of instantiations according to "robot_num"
        # # print(sc.SENSOR_TYPE)
        # if sc.SENSOR_TYPE == "None":
        #     sc.setVrepHandles(0, '')
        #     for i in range(1,args.robot_num+1):
        #         sc.setVrepHandles(i, '#'+str(i))
        #     # sc.setVrepHandles(1, '#0')
        #
        # elif sc.SENSOR_TYPE == "VPL16":
        #     sc.objectNames.append('velodyneVPL_16') # _ptCloud
        #     print(sc.objectNames)
        #     for i in range(args.robot_num):
        #         checkn = i - 1
        #         s = ''
        #         if(i >= 1):
        #             s += '#'+str(checkn)
        #         sc.setVrepHandles(i,s)

        #sc.renderScene(waitTime = 3000)
        tf = args.sim_time ## must lager than 3

        CheckerEnabled = False
        initRef(sc) #sc.resetPosition(robot_num*np.sqrt(2)) # Random initial position
        sc.resetPosition(args.position_range)
        # sc.robots[0].setPosition([.0, .0, 0.0])
        # sc.robots[1].setPosition([-3.0, .0, 0.0])
        # sc.robots[2].setPosition([3.0, .0, 0.0])
        # sc.robots[3].setPosition([.0, 3.0, 0.0])
        # sc.robots[4].setPosition([0.0, -3.0, 0.0])


        # sp.plot(4, tf,expert=args.expert_only)
        realstop = int(args.stop_waiting_time/args.sim_dt)
        while sc.simulate():
            # for r in range(len(sc.robots)):
            #     positionList[ep].append([sc.robots[r].xi.x,sc.robots[r].xi.y])
            # stop=True
            stop=sc.check_stop_condition(args.desire_distance,args.stop_thresh)
            # for i in range(len(sc.robots)):
            #     # print(sc.robots[i].wheel_velocity_1,sc.robots[i].wheel_velocity_2)
            #     if not(sc.robots[i].wheel_velocity_1==0 and sc.robots[i].wheel_velocity_2==0):
            #         stop=False
            if sc.t > tf or stop:
                print("stop")
                if realstop>0:
                    realstop-=1
                else:
                    print("Stop at")
                    print(sc.t)
                    break
    except KeyboardInterrupt:
        x = input('Quit?(y/n)')
        if x == 'y' or x == 'Y':
            tf = sc.t - 0.01
            raise Exception('Aborted.')
    except VrepError as err:
        sc.log(err.message)
        print(err.message)
        return None
    except:
        raise
    finally:
        sc.deallocate()

    return sc


Train(args,i)


