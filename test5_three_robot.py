# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 23:09:34 2017
This test file is dependent on vrep.
To run this file, please open vrep file scene/scene_double.ttt first
@author: cz
"""

from scene import Scene
from robot import VrepError
from sceneplot import ScenePlot
# from robot import Robot
import numpy as np
import math
import random
# from data import Data
#from DeepFCL import DeepFCL
from suhaas_agent import Agent
import torch

TRAIN = False
fcl = Agent(inW = 100, inH = 100)
if(not TRAIN):
    fcl.model.to('cpu')
    fcl.model.load_state_dict(torch.load('models/v8/suhaas_model_v8_dagger_final.pth'))
    fcl.model.to('cuda')

def initRef(sc, i):
    if sc.dynamics == sc.DYNAMICS_MODEL_BASED_LINEAR:
        #radiusLeaderList = [2.0, 3.0, 4.0]
        #speedLeaderList = [0.2, 0.3, 0.4]
        radiusLeaderList = [2.0]
        speedLeaderList = [0.4]
        radiusLeader = random.choice(radiusLeaderList)
        sc.referenceSpeed = random.choice(speedLeaderList)
        sc.referenceOmega = sc.referenceSpeed / radiusLeader
        message = "Ref speed: {0:.3f} m/s; Ref omega: {1:.3f} rad/s; Ref radius: {2:.3f} m"
        message = message.format(sc.referenceSpeed, sc.referenceOmega, radiusLeader)
    elif (sc.dynamics == sc.DYNAMICS_MODEL_BASED_LINEAR_GOAL or
          sc.dynamics == sc.DYNAMICS_MODEL_BASED_DISTANCE_GOAL):
        g = 4.0
        goalList = [[g, g], [-g, g], [g, -g], [-g, -g]]
        #sc.xid.x, sc.xid.y = random.choice(goalList)
        sc.xid.x, sc.xid.y = goalList[i]
        sc.xid.vRef = 0.7
        message = "Goal: ({0:.3f}, {1:.3f})"
        message = message.format(sc.xid.x, sc.xid.y, sc.xid.vRef)
        sc.xid.theta = 0
        sc.xid.sDot = 0
        sc.xid.thetaDot = 0
    elif sc.dynamics == sc.DYNAMICS_MODEL_BASED_DISTANCE_REFVEL:
        # set desired velocity vector
        sc.xid.vRefMag = 0.36055512754
        sc.xid.vRefAng = 2 * math.pi * random.random()#0.982793723# 2 * math.pi * random.random()
        sc.xid.theta = 0
        sc.xid.sDot = 0
        sc.xid.thetaDot = 0
        # scale desired formation separation
        #alphaList = [1.0, 1.5, 2.0]
        #alphaList = [1.0,2.0,3.0,4.0,4.5]
        alphaList = [2.5]
        alpha = random.choice(alphaList)
        sc.scaleDesiredFormation(alpha)
        message = "vRefMag: {0:.3f}, vRefAng: {1:.3f}, alpha: {2:.3f}"
        message = message.format(sc.xid.vRefMag, sc.xid.vRefAng, alpha)
    sc.log(message)
    print(message)

def plot(sp, tf): #sp.plot(0, tf) sp.plot(2, tf) # Formation Separation
    if sp.sc.dynamics == 14:
        sp.plot(21, tf) # Formation Orientation
    if sp.sc.dynamics == 16:
        sp.plot(23, tf) # distance from goal
    elif sp.sc.dynamics == 14:
        sp.plot(22, tf) # distances from goals
    sp.plot(2,tf)
    sp.plot(4, tf)
    #sp.plot(5, tf)
    sp.plot(6, tf)

def generateData(i):
    sc = Scene(fileName = __file__, recordData = True, runNum = i)
    sp = ScenePlot(sc)
    sp.saveEnabled = True # save plots?
    #sc.occupancyMapType = sc.OCCUPANCY_MAP_THREE_CHANNEL
    sc.occupancyMapType = sc.OCCUPANCY_MAP_BINARY
    sc.dynamics = sc.DYNAMICS_MODEL_BASED_DISTANCE_REFVEL # robot dynamics
    sc.errorType = 0
    try:
#        sc.addRobot(np.float32([[0, 0, 0], [0.0, 0.0, 0.0]]), role = sc.ROLE_PEER)
#        sc.addRobot(np.float32([[-2, 0.001, 0], [1.0, 0.0, 0.0]]), role = sc.ROLE_PEER)
#        sc.addRobot(np.float32([[2, 0, 0], [0.0, 1.732, 0.0]]), role = sc.ROLE_PEER)

        sc.addRobot(np.float32([[-2, 0, 0], [0.0, 0.0, 0.0]]), role = sc.ROLE_PEER,learnedController = fcl.test)
        sc.addRobot(np.float32([[1, 3, 0], [-2.0/2, 0.0, 0.0]]), role = sc.ROLE_PEER,learnedController = fcl.test)
        sc.addRobot(np.float32([[2, 3, 0], [-1.0/2, 1.732/2, 0.0]]), role = sc.ROLE_PEER,learnedController = fcl.test)
#==============================================================================
#         sc.addRobot(np.float32([[1, 3, 0], [0, -1, 0]]),
#                     dynamics = sc.DYNAMICS_LEARNED,
#                     learnedController = fcl.test)
#==============================================================================

        # No leader
        sc.setADjMatrix(np.uint8([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))
        # Set robot 0 as the leader.

        # vrep related
        sc.initVrep()
        # Choose sensor type
        #sc.SENSOR_TYPE = "VPL16" # None, 2d, VPL16, kinect
        sc.SENSOR_TYPE = "VPL16" # None, 2d, VPL16, kinect
        sc.objectNames = ['Pioneer_p3dx', 'Pioneer_p3dx_leftMotor', 'Pioneer_p3dx_rightMotor']

        if sc.SENSOR_TYPE == "None":
            sc.setVrepHandles(0, '')
            sc.setVrepHandles(1, '#0')
            sc.setVrepHandles(2, '#1')
        elif sc.SENSOR_TYPE == "2d":
            sc.objectNames.append('LaserScanner_2D_front')
            sc.objectNames.append('LaserScanner_2D_rear')
            sc.setVrepHandles(0, '')
            sc.setVrepHandles(1, '#0')
            sc.setVrepHandles(2, '#1')
        elif sc.SENSOR_TYPE == "VPL16":
            sc.objectNames.append('velodyneVPL_16') # _ptCloud
            sc.setVrepHandles(0, '')
            sc.setVrepHandles(1, '#0')
            sc.setVrepHandles(2, '#1')
        elif sc.SENSOR_TYPE == "kinect":
            sc.objectNames.append('kinect_depth')
            sc.objectNames.append('kinect_rgb')
            sc.setVrepHandles(0, '')
            sc.setVrepHandles(1, '#0')
            sc.setVrepHandles(2, '#1')

        #sc.renderScene(waitTime = 3000)
        tf = 60 # must be greater than 1
        errorCheckerEnabled = False
        initRef(sc, i)
        sc.resetPosition(2) # Random initial position
        #sc.resetPosition(None)

        #sc.robots[0].setPosition([.0, .0, .0])
        #sc.robots[1].setPosition([-2.0, 0.001, 0.0])
        #sc.robots[2].setPosition([2.0, 0.0, 0.0])

        #sc.robots[0].setPosition([.0, .0, .0])
        #sc.robots[1].setPosition([-3.0, 4.0, 0.0])
        #sc.robots[2].setPosition([2.0, 1.0, 0.0])

        # Fixed initial position
        #sc.robots[0].setPosition([0.0, 0.0, math.pi/2])
        #sc.robots[1].setPosition([-2.2, -1.0, 0.3])
        sp.plot(4, tf)
        while sc.simulate():
            #sc.renderScene(waitTime = int(sc.dt * 1000))
            #sc.showOccupancyMap(waitTime = int(sc.dt * 1000))

            #print("---------------------")
            #print("t = %.3f" % sc.t, "s")
            #print(sc.robots[2].xid0.y)
            if sc.t > 1:
                maxAbsError = sc.getMaxFormationError()
                if maxAbsError < 0.01 and errorCheckerEnabled:
                    #tf = sc.t - 0.01
                    # set for how many seconds after convergence the simulator shall run
                    tExtra = 30
                    #tf = sc.t + tExtra
                    errorCheckerEnabled = False
                    print('Ending in ', str(tExtra), ' seconds...')

            plot(sp, tf)
            if sc.t > tf:
                message = "maxAbsError = {0:.3f} m".format(maxAbsError)
                sc.log(message)
                print(message)
                break


    except KeyboardInterrupt:
        x = input('Quit?(y/n)')
        if x == 'y' or x == 'Y':
            tf = sc.t - 0.01
            plot(sp, tf)
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



# main
import saver
numRun = 250 if TRAIN else 1 # This is to set the number of iterations of the Dagger algorithm
dataList = [] # This is where the training data will be stored
sc = None
#fcl.init_test()
lossList = []
for i in range(0, numRun):
    print('Episode:', i+1)
    ##########################################################################
    ######## Step 1: Start simulation rollouts to get training data ##########
    ##########################################################################

    dataListEpisode = []
    # First episode
    sc0 = generateData(i)
    if sc0 is not None:
        # if the list is not empty
        sc = sc0
        saver.save(sc) # save data
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
    if(TRAIN):
        l = fcl.train(dataListEpisode)
        lossList.append(l)

    nnn = 0
    nm = 0
    for robot in sc.robots:
        nnn += robot.numNN
        nm += robot.numMod
    print('Number of times neural network was selected:', nnn)
    print('Number of times expert model was selected:', nm)
    if(i % 50 == 0 and TRAIN):
        fcl.save('v8/suhaas_model_v8_dagger_' + str(i) + '.pth')
if sc:
    for j in range(1, len(sc.robots)):
        dataList[0].append(dataList[j])
    dataList[0].store()

if(TRAIN):
    import matplotlib.pyplot as plt
    fcl.save('v8/suhaas_model_v8_dagger_final.pth')
    print(lossList)
    plt.figure()
    plt.plot(lossList)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss over training episodes')
    plt.savefig('losslist.pdf')
    plt.show()
    np.save('losslist.npy',lossList)
#for j in range(len(sc.robots)):
#    dataList[j].store()


