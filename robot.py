# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 23:08:18 2017

@author: cz
"""

try:
    import cv2
    USE_CV2 = True
except ImportError:
    USE_CV2 = False

import operator
import math
from state import State
import numpy as np
import sim as vrep
from data import Data
from pointcloud import PointCloud
#import time
#import random


def saturate(dxp, dyp, dxypMax):
    dxyp = (dxp**2 + dyp**2)**0.5
    if dxyp > dxypMax:
        dxp = dxp / dxyp * dxypMax
        dyp = dyp / dxyp * dxypMax
    return dxp, dyp

class Robot():
    def __init__(self, scene,numRobots):
        self.index=None
        self.scene = scene
        self.dynamics = 18
        self.numNN = 0
        self.numMod = 0
        # dynamics parameters

        self.l = 0.331
        self.nr = numRobots
        # state
        self.xi = State(0, 0, 0, self)
        #self.xi = State(random.random()*100, random.random()*100,2*math.pi*random.random(),self)
        self.xid = State(3, 0, 0, self)
        self.xid0 = State(3, 0, math.pi/4, self)
        self.reachedGoal = False
        # Control parameters
        self.kRho = 1
        self.kAlpha = 6
        self.kPhi = -1
        self.kV = 3.8
        self.gamma = 0.15
        self.p=0.8
        self.model_controller=False

        #
        self.pointCloud = PointCloud(self)
        # control parameters
        self.control_vmax=1.2
        self.control_vmin = 0.01
        self.LIMIT_MAX_ACC=False
        self.accMax = 0.5
        # Data to be recorded
        self.recordData = False
        self.data = Data(self)
        self.v1Desired = 0
        self.v2Desired = 0
        self.v1Desirednn = 0
        self.v2Desirednn = 0
        #### Robot's neighbor
        self.role = None
        self.neighbors = []
        self.leader = None # Only for data recording purposes

        self.ctrl1_sm = []
        self.ctrl2_sm = []

        self.position_hist = []


    def checkMove(self,hist,num = 1,thresh = .01):
        moving = False
        #if(abs(hist[0] - hist[-1]) > thresh):
        #   moving = True
        for i in range(1,len(hist)):
            if(abs(hist[i] - hist[i - 1]) > thresh):
                moving = True
        return moving
    #### Expert Controller
    def expert_control(self,omlist,index):
        ############## MODEL-BASED CONTROLLER (Most frequently used dynamics model) ##########
        ######################################################################################
        # For e-puk dynamics
        # Feedback linearization
        # v1: left wheel speed
        # v2: right wheel speed

        K3 = 0.0  # interaction between i and j
        dxypMax = float('inf')
        if self.role == "LEADER":  # I am a leader
            K1 = 1
            K2 = 1
        elif self.role == "FOLLOWER":
            K1 = 0  # Reference position information is forbidden
            K2 = 1
        elif self.role == "PEER":
            K1 = 0
            K2 = 0
            K3 = 1  # interaction between i and j
            dxypMax = 0.7

        # sort neighbors by distance

        # need all neighbors, but only diagonal of adjmatrix is 0 so it is okay
        if True:  # not hasattr(self, 'dictDistance'):
            self.dictDistance = dict()
            for j in range(len(self.scene.robots)):
                # if self.scene.adjMatrix[self.index, j] == 0:
                if self.index == j:
                    continue
                robot = self.scene.robots[j]  # neighbor
                self.dictDistance[j] = self.xi.distancepTo(robot.xi)
            self.listSortedDistance = sorted(self.dictDistance.items(),
                                             key=operator.itemgetter(1))

        # velocity in transformed space
        vxp = 0
        vyp = 0

        tauix = 0
        tauiy = 0
        # neighbors sorted by distances in descending order

        # gabriel graph connections
        lsd = self.listSortedDistance

        jList = []
        for i in range(len(lsd)):
            connected = True
            for k in range(len(lsd)):
                if i == k:
                    continue
                ri = lsd[i][0]
                rk = lsd[k][0]
                di = np.array([self.xi.xp - self.scene.robots[rk].xi.xp, self.xi.yp - self.scene.robots[rk].xi.yp])
                dj = np.array([self.scene.robots[ri].xi.xp - self.scene.robots[rk].xi.xp,
                               self.scene.robots[ri].xi.yp - self.scene.robots[rk].xi.yp])
                c = np.dot(di, dj) / (np.linalg.norm(di) * np.linalg.norm(dj))
                angle = np.degrees(np.arccos(c))
                if (angle > 89 and i != k):
                    connected = False
            if (connected):
                jList.append(lsd[i][0])
        for j in jList:
            robot = self.scene.robots[j]
            pijx = self.xi.xp - robot.xi.xp
            pijy = self.xi.yp - robot.xi.yp
            pij0 = self.xi.distancepTo(robot.xi)
            if self.dynamics == 18:
                pijd0 = self.scene.alpha
            else:
                pijd0 = self.xid.distancepTo(robot.xid)
            tauij0 = (pij0 - pijd0) / pij0
            tauix += tauij0 * pijx
            tauiy += tauij0 * pijy


        # Achieve and keep formation
        # tauix, tauiy = saturate(tauix, tauiy, dxypMax)
        vxp += -K3 * tauix
        vyp += -K3 * tauiy

        # Velocity control toward goal

        dxp = self.scene.xid.dpbarx  # + self.l / 2 * dCosTheta
        dyp = self.scene.xid.dpbary  # + self.l / 2 * dCosTheta
        # Velocity control toward goal
        # dxp = self.xi.xp - self.xid.xp
        # dyp = self.xi.yp - self.xid.yp
        # Limit magnitude
        dxp, dyp = saturate(dxp, dyp, dxypMax)
        vxp += -K1 * dxp
        vyp += -K1 * dyp

        # Take goal's speed into account
        vxp += K2 * self.xid.vxp
        vyp += K2 * self.xid.vyp

        kk = 1
        theta = self.xi.theta
        M11 = kk * math.sin(theta) + math.cos(theta)
        M12 = -kk * math.cos(theta) + math.sin(theta)
        M21 = -kk * math.sin(theta) + math.cos(theta)
        M22 = kk * math.cos(theta) + math.sin(theta)

        v1 = M11 * vxp + M12 * vyp
        v2 = M21 * vxp + M22 * vyp

        vmax = self.control_vmax  # wheel's max linear speed in m/s
        vmin = self.control_vmin # wheel's min linear speed in m/s

        # Find the factor for converting linear speed to angular speed
        if math.fabs(v2) >= math.fabs(v1) and math.fabs(v2) > vmax:
            alpha = vmax / math.fabs(v2)
        elif math.fabs(v2) < math.fabs(v1) and math.fabs(v1) > vmax:
            alpha = vmax / math.fabs(v1)
        else:
            alpha = 1
        v1 = alpha * v1
        v2 = alpha * v2
        if math.fabs(v1)<vmin:
            v1=0
        if math.fabs(v2)<vmin:
            v2=0
        return v1,v2
    def gnn_control(self,omlist,index):
        ####################### NN CONTROLLER ###########################################
        #################################################################################

        observation, action_1 = self.data.getObservation(-12)
        # if observation is None:
        #    action = np.array([[0, 0]])
        # else:
        action = self.learnedController(omlist, index)
        # action = self.learnedController(observation, self.graph_matrix, action_1[0][0],self.scene.alpha)
        # action = np.array([[0, 0]])
        action = action[0].cpu().detach().numpy()
        v1nn = action[0][0]
        v2nn = action[0][1]
        # smoothing
        self.ctrl1_sm.append(v1nn)
        self.ctrl2_sm.append(v2nn)
        if len(self.ctrl1_sm) < 10:
            v1nn = sum(self.ctrl1_sm) / len(self.ctrl1_sm)
            v2nn = sum(self.ctrl2_sm) / len(self.ctrl2_sm)
        else:
            v1nn = sum(self.ctrl1_sm[len(self.ctrl1_sm) - 10:len(self.ctrl1_sm)]) / 10
            v2nn = sum(self.ctrl2_sm[len(self.ctrl2_sm) - 10:len(self.ctrl2_sm)]) / 10

        # stopping condition

        current_position = (self.xi.xp ** 2 + self.xi.yp ** 2) ** 0.5
        self.position_hist.append(current_position)
        hist_len = len(self.position_hist)
        lcheck = 10
        if (hist_len > 100):
            currhist = self.position_hist[-1 * lcheck:]
            if (not self.checkMove(currhist, num=lcheck, thresh=.00001)):
                v1nn = 0
                v2nn = 0
            # for pos in range(1,len(currhist)):
            #    if(abs(currhist[pos] - currhist[pos - 1]) > .005):
            #        moving = True
            # if(not moving):
            #        print('stopped')
            #        v1nn = 0
            #        v2nn = 0
            # if(abs(self.position_hist[hist_len - 1] - self.position_hist[hist_len-2]) / self.position_hist[hist_len-1] < .0001):
            #    v1nn = 0
            #    v2nn = 0
        ### post-processing ###


        vmax = self.control_vmax  # wheel's max linear speed in m/s
        vmin = self.control_vmin  # wheel's min linear speed in m/s

        # Find the factor for converting linear speed to angular speed
        if math.fabs(v2nn) >= math.fabs(v1nn) and math.fabs(v2nn) > vmax:
            alpha = vmax / math.fabs(v2nn)
        elif math.fabs(v2nn) < math.fabs(v1nn) and math.fabs(v1nn) > vmax:
            alpha = vmax / math.fabs(v1nn)
        else:
            alpha = 1
        v1nn = alpha * v1nn
        v2nn = alpha * v2nn
        if math.fabs(v1nn) < vmin:
            v1nn = 0
        if math.fabs(v2nn) < vmin:
            v2nn = 0
        v1nn=1
        v2nn=1
        # Limit maximum acceleration (deprecated)

        if self.LIMIT_MAX_ACC:
            dvMax =  self.accMax * self.scene.dt

            # limit v1nn
            dv1nn = v1nn - self.v1Desirednn
            if dv1nn > dvMax:
                self.v1Desirednn += dvMax
            elif dv1nn < -dvMax:
                self.v1Desirednn -= dvMax
            else:
                self.v1Desirednn = v1nn
            v1nn = self.v1Desired

            # limit v2nn
            dv2nn = v2nn - self.v2Desirednn
            if dv2nn > dvMax:
                self.v2Desirednn += dvMax
            elif dv2nn < -dvMax:
                self.v2Desirednn -= dvMax
            else:
                self.v2Desirednn = v2nn
            v2nn = self.v2Desirednn
        elif not self.LIMIT_MAX_ACC:
            self.v1Desirednn = v1nn
            self.v2Desirednn = v2nn

        return v1nn,v2nn
    def control(self,omlist,index):
        v1,v2=self.expert_control(omlist,index)
        v1nn,v2nn=self.gnn_control(omlist,index)

        # Record data
        if (self.scene.vrepConnected and
            self.scene.SENSOR_TYPE == "VPL16" and
            self.VPL16_counter == 3 and self.recordData == True):
            self.data.add()

        # print('v = ', pow(pow(v1, 2) + pow(v2, 2), 0.5))

        ######## Select either model-based control of NN control ########
        #################################################################
        #print('\n NN output: ', v1nn, v2nn)
        #print('\n Expert output: ', v1, v2)

        ####### TO ADD THE CONTROLLER SELECTION MECHANISM HERE #############
        # use binomial distribution with probability \beta
        p = self.p # can be tweaked
        exp = (self.scene.runNum) // 20
        #exp = (self.scene.runNum-101)//20
        exp = max(0,exp)
        beta = p**(exp)  # Dagger algorithm paper, page 4
        model_controller = np.random.binomial(1, beta)

        ## Control Training or not
        TRAIN = False
        DAGGER = False
        if (model_controller and TRAIN) or (not DAGGER):
            model_controller=False
        else:
            model_controller=True
        #### decide to use which controller
        model_controller=False
        if model_controller:
            v1 = v1nn
            v2 = v2nn
            # print('\n NN control selected')
            self.numNN += 1
        else:
            self.numMod += 1


        if self.scene.vrepConnected:
            # omega1 = v1 * 10.25
            # omega2 = v2 * 10.25
            omega1 = v1
            omega2 = v2
            # return angular speeds of the two wheels
            self.nnv1 = omega1
            self.nnv2 = omega2


            # return omega1/5, omega2/5
            return omega1, omega2
        else:
            # return linear speeds of the two wheels
            return v1, v2

    ##### (not yet finish) For update_state
    #### Set one robot's Position and Orientation
    def update_pose(self,stateVector):
        """
        For update_state
        Args:
            stateVector: [x,y,theta] the desire pose of robot

        Returns: None

        """
        z0 = 0.1587
        if stateVector == None:
            x0 = self.xi.x
            y0 = self.xi.y
            theta0 = self.xi.theta
        elif len(stateVector) == 3:
            x0 = stateVector[0]
            y0 = stateVector[1]
            theta0 = stateVector[2]
            self.xi.x = x0
            self.xi.y = y0
            self.xi.theta = theta0
        else:
            raise Exception('Argument error!')
        position = [x0, y0, z0]
        orientation = [0, 0, theta0]
        handle = self.robotHandle
        self.scene.executor_setpose(self.index, handle, position, orientation)
    #### Set one robot's neighbors
    ##### (not yet finish) For update_state
    def update_neighbors(self,adjmatrix,robot_list):
        """
        Set the neighbors of one robot according to the adjmatrix
        Args:
            adjmatrix: Scene's adjacent matrix
            robot_list: All robots in the scene

        Returns: None

        """
        self.neighbors = []
        self.leader = None
        for j in range(len(robot_list)):
            if adjmatrix[self.index, j] == 0:
                continue
            robot = robot_list[j]  # neighbor
            self.neighbors.append(robot)
            if robot.role == "LEADER":
                if self.leader is not None:
                    raise Exception('There cannot be more than two leaders in a scene!')
                self.leader = robot
    #### (not yet finish)
    def update_state(self,stateVector):
        self.update_pose(stateVector)
        # self.update_neighbors(adjmatrix, robot_list)
#### Get one robot's Position, Orientation and Lidar reading
    def get_sensor_data(self,pos,ori,vel,omega,velodyne_points):
        """
        Get pose from simulator and sensor
        Args:
            pos: robot position from simulator
            ori: robot orientation from simulator
            vel: robot linear velocity from simulator
            omega: robot angular velocity from simulator
            velodyne_points: sensor data

        Returns: None

        """
        ##### Get absolute pose from simulator
        self.xi.x = pos[0]
        self.xi.y = pos[1]
        self.xi.alpha = ori[0]
        self.xi.beta = ori[1]
        self.xi.theta = ori[2]
        sgn = np.sign(np.dot(np.asarray(vel[0:2]),
                             np.asarray([math.cos(self.xi.theta),
                                         math.sin(self.xi.theta)])))
        self.vActual = sgn * (vel[0]**2 + vel[1]**2)**0.5
        self.omegaActual = omega[2]
        ##### Get laser/vision sensor data
        # Parse data
        if not velodyne_points==None:
            res = velodyne_points[0]
            if 'VPL16_counter' not in self.__dict__:
                self.VPL16_counter = 0
            # reset the counter every fourth time
            if self.VPL16_counter == 4:
                self.VPL16_counter = 0
            if self.VPL16_counter == 0:
                # Reset point cloud
                self.pointCloud.clearData()
            self.pointCloud.addRawData(velodyne_points[2]) # will rotate here
            if self.VPL16_counter == 3:
                self.pointCloud.crop()
                self.pointCloud.updateOccupancyMap() # option 1
            self.VPL16_counter += 1
            # add for gnn
            ###### (not yet finish)
            self.graph_matrix = self.scene.readADjMatrix(MaxRange=2)
    def getV1V2(self):
        v1 = self.vActual + self.omegaActual * self.l / 2
        v2 = self.vActual - self.omegaActual * self.l / 2
        return np.array([[v1, v2]])
##### might move to scene
##### pending decide
    def propagateDesired(self):
        """
        Update robot pose
        Returns:None

        """
        if self.dynamics == 17 or self.dynamics == 18:
            self.xid.theta = self.scene.xid.vRefAng

##### For simulate
    def precompute(self,adjmatrix,robot_list):
        self.xi.transform()
        self.xid.transform()
        self.update_neighbors(adjmatrix,robot_list)
#### Set the linear velocity of 2 wheels
##### move to scene and merger with simulate
    # def propagate(self,omega1,omega2):
    #
    #     if self.scene.vrepConnected == False:
    #         self.xi.propagate(self.control)
    #     else:
    #         vrep.simxSetJointTargetVelocity(self.scene.clientID,
    #                                         self.motorLeftHandle,
    #                                         omega1, vrep.simx_opmode_oneshot)
    #         vrep.simxSetJointTargetVelocity(self.scene.clientID,
    #                                         self.motorRightHandle,
    #                                         omega2, vrep.simx_opmode_oneshot)

    def get_control(self,omlist,i):
        if self.scene.vrepConnected == False:
            self.xi.propagate(self.control)
        else:
            omega1, omega2 = self.control(omlist,i)
            return omega1,omega2
    ##### For simulate in scene
    def getDataObs(self):
        observation, action_1 = self.data.getObservation(-12)
        return observation, self.graph_matrix, action_1[0][0],self.scene.alpha






