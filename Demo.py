# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 23:09:34 2017
This test file is dependent on vrep.
To run this file, please open vrep file scene/scene_double.ttt first
@author: cz
"""

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
import matplotlib.pyplot as plt


import argparse
parser = argparse.ArgumentParser(description='Args for demo')
parser.add_argument('--if_train', dest='if_train', default=False,type=bool,help='Control demo mod(train/test)')
parser.add_argument('--if_continue', dest='if_continue', default=False,type=bool,help='Continue training')
parser.add_argument('--expert_only', dest='expert_only', default=True,type=bool,help='Use expert control only')
parser.add_argument('--robotNum', dest='robotNum', default=9,type=int,help='Number of robot for simulation')
parser.add_argument('--simTime', dest='simTime', default=200,type=int,help='Simulation time for one simulation')
parser.add_argument('--trainEpisode', dest='trainEpisode', default=1,type=int,help='Episode for training')
parser.add_argument('--modelName', dest='modelName', default='v13/suhaas_model_v13_dagger_final_more.pth',type=str,help='Path to model')
# # modelname='model_'+str(robotNum)+'robots_'+str(simTime)+'s_'+str(trainEpisode)+'rounds'+'.pth'
args = parser.parse_args()
print(args.trainEpisode)



def demo_one(args):
    #### store robot pose
    numRun = args.trainEpisode if args.if_train else 1

    positionList = [[] for n in range(numRun)]
    #### training data will be stored
    dataList = []

    lossList = []
    sc = None
    args.modelName = 'v13/test_train.pth' if args.if_train else 'v13/suhaas_model_v13_dagger_final_more.pth'
    numRun = args.trainEpisode if args.if_train else 1

    #### Initial Agent
    fcl = Agent(inW=100, inH=100, nA=args.robotNum)
    if (not args.if_train):
        fcl.model.to('cpu')
        fcl.model.load_state_dict(torch.load('models/' + args.modelName))
        fcl.model.to('cuda')
    if (args.if_continue):
        fcl.model.to('cpu')
        fcl.model.load_state_dict(torch.load('models/' + args.modelName))
        fcl.model.to('cuda')
        print('Loaded model')
    for i in range(numRun):
        # print(lossList)
        print('Episode:', i + 1)
        ##########################################################################
        ######## Step 1: Start simulation rollouts to get training data ##########
        ##########################################################################

        dataListEpisode = []
        # First episode
        sc0 = generateData(args, fcl,i,positionList)
        sc0.save_robot_states("")
        plot_scene(sc0,"","figs")
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
            l = fcl.train(dataListEpisode)
            lossList.append(l)

        if (i % 25 == 0 and args.if_train):
            fcl.save(args.modelname)


    positionList = np.array(positionList)
    np.save('positionLists/' + 'positionList_expert_' + str(args.robotNum) + '_singles.npy', positionList)
    if sc:
        print('data stored')
        print(sc.dt)
        for j in range(1, len(sc.robots)):
            dataList[0].append(dataList[j])
        dataList[0].store()
    else:
        print('data not stored')


def initRef(sc, i):

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

def generateData(args,agent,ep,positionList):
    if(args.if_continue):
        ep -=args.if_trainEpisode
    sc = Scene(fileName = __file__, recordData = True, runNum = ep)
    sp = ScenePlot(sc)
    sp.saveEnabled = True # save plots?
    # global numRun
    # global positionList
    #sc.occupancyMapType = sc.OCCUPANCY_MAP_THREE_CHANNEL
    sc.occupancyMapType = sc.OCCUPANCY_MAP_BINARY
    # sc.dynamics = 18 # robot dynamics
    sc.errorType = 0

    try:
        for i in range(args.robotNum):
            sc.addRobot(np.float32([[-2, 0, 1], [0.0, 0.0, 0.0]]),args.robotNum,expert_controller=True, learnedController = agent.test)
        # No leader
        I = np.identity(args.robotNum, dtype=np.int8)
        M = np.ones(args.robotNum, dtype=np.int8)
        sc.setADjMatrix(M-I)

        # Set robot 0 as the leader.

        # vrep related
        sc.initVrep()
        # Choose sensor type
        #sc.SENSOR_TYPE = "VPL16" # None, 2d, VPL16, kinect
        sc.SENSOR_TYPE = "VPL16" # None, 2d, VPL16, kinect
        sc.objectNames = ['Pioneer_p3dx', 'Pioneer_p3dx_leftMotor', 'Pioneer_p3dx_rightMotor']

        # change the # of instantiations according to "robotNum"
        # print(sc.SENSOR_TYPE)
        if sc.SENSOR_TYPE == "None":
            sc.setVrepHandles(0, '')
            for i in range(1,args.robotNum+1):
                sc.setVrepHandles(i, '#'+str(i))
            # sc.setVrepHandles(1, '#0')

        elif sc.SENSOR_TYPE == "VPL16":
            sc.objectNames.append('velodyneVPL_16') # _ptCloud
            print(sc.objectNames)
            for i in range(args.robotNum):
                checkn = i - 1
                s = ''
                if(i >= 1):
                    s += '#'+str(checkn)
                sc.setVrepHandles(i,s)

        #sc.renderScene(waitTime = 3000)
        tf = args.simTime ## must lager than 3

        CheckerEnabled = False
        initRef(sc, i) #sc.resetPosition(robotNum*np.sqrt(2)) # Random initial position
        sc.resetPosition(5)
        # sp.plot(4, tf,expert=args.expert_only)
        realstop = 10
        while sc.expert_simulate():
            for r in range(len(sc.robots)):
                positionList[ep].append([sc.robots[r].xi.x,sc.robots[r].xi.y])
            stop=True

            for i in range(len(sc.robots)):
                if not(sc.robots[i].wheel_velocity_1==0 and sc.robots[i].wheel_velocity_2==0):
                    stop=False
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
    if True: #maxAbsError < 0.01:
        return sc
    else:
        return None

demo_one(args)


