# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 18:33:11 2018

@author: cz
"""

# save scene
import pickle
import os
import numpy as np

directory = 'data_scene'

#### Save the scene class to a pkl file
def save_scene(sc):
    if not os.path.exists(directory):
        os.makedirs(directory)
    count = 0
    for filename in os.listdir(directory):
        name, _ = os.path.splitext(filename)
        n = int(name[2:])
        if count < n:
            count = n
    count += 1
    for robot in sc.robots:
        robot.data.q = None
        if robot.learnedController is not None:
            robot.learnedController = None
    if sc.out is not None:
        sc.out = None
    path = os.path.join(directory, 'sc'+str(count).zfill(3)+'.pkl')
    with open(path, 'wb') as f:
        pickle.dump(sc, f)
    sc.log("Scene is written to " + path + '\n')
#### Save the robots states of one simulation to a npy file


        # self.t = 0
        # self.dt = 0.01
        #
        # # formation reference link
        # self.xid = State(0.0, 0.0, math.pi / 2)
        # self.xi = State(0.0, 0.0, math.pi / 2)
        # self.alpha = 1 # desired formation scale
        #
        # # for plots
        # self.ts = [] # timestamps
        # self.tss = [] # timestamps (sparse)
        # self.ydict = dict()
        # self.ydict2 = dict()
        # self.ploted = dict()
        #
        # # For visualization
        # self.wPix = 600
        # self.hPix = 600
        # self.xMax = 8
        # self.yMax = 8
        # self.image = np.zeros((self.hPix, self.wPix, 3), np.uint8)
        # if USE_CV2:
        #     self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #     self.out = cv2.VideoWriter('output.avi', self.fourcc, 20.0, (self.wPix, self.hPix))
        #     self.frameCounter = 0
        #
        # self.robots = []
        # self.adjMatrix = None
        # self.Laplacian = None
        #
        # # vrep related
        # self.vrepConnected = False
        # #self.vrepSimStarted = False
        # self.SENSOR_TYPE = "None"
        # self.objectNames = []
        # self.recordData = recordData
        #
        # self.occupancyMapType = None
        # self.OCCUPANCY_MAP_BINARY = 0
        # # 1 for 3-channel: mean height, height variance, visibility
        # self.OCCUPANCY_MAP_THREE_CHANNEL = 1
        #
        # # CONSTANTS
        # self.dynamics = 18
        # self.DYNAMICS_INTEGRATOR_MODEL = 5
        # self.DYNAMICS_MODEL_BASED_CICULAR = 11
        # self.DYNAMICS_MODEL_BASED_STABILIZER = 12
        # self.DYNAMICS_MODEL_BASED_LINEAR = 13
        # self.DYNAMICS_MODEL_BASED_LINEAR_GOAL = 14
        # self.DYNAMICS_MODEL_BASED_DISTANCE_GOAL = 16
        # self.DYNAMICS_MODEL_BASED_DISTANCE_REFVEL = 17
        # self.DYNAMICS_MODEL_BASED_DISTANCE2_REFVEL = 18
        # self.DYNAMICS_LEARNED = 30
        #
        # # follower does not have knowledge of absolute position
        # self.ROLE_LEADER = 0
        # self.ROLE_FOLLOWER = 1
        # self.ROLE_PEER = 2
        #
        # self.errorType = 0
        # self.logPriorityMax = 1 # Messages with lower priorities are not logged
        # self.logFileName = os.path.splitext(fileName)[0] + ".log"
        # self.runNum = runNum
        # self.log('A new scene is created for run #' + str(runNum))
def save_states(sc):
    # print(sc.xid)
    # print(sc.xi)
    # print(sc.alpha)
    print(sc.robots[0].xi.xp)
    print(len(sc.robots[0].position_hist))
    print(sc.adjMatrix)
    print(sc.runNum)
    print(sc.robots[0].neighbors)
def save(sc):
    save_states(sc)
    #save_scene(sc)
def load(path):
    with open(path, 'rb') as f:
        sc = pickle.load(f)
        return sc
path="/home/xinchi/GNN-control/gnn-formation-control/data_scene/sc001.pkl"
sc=load(path)
save(sc)

    