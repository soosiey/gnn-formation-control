# -*- coding: utf-8 -*-

try:
    import cv2
    USE_CV2 = True
except ImportError:
    USE_CV2 = False
import numpy as np
from robot import Robot
#import matplotlib.pyplot as plt
import sim as vrep
import math
import random
import datetime
import os
from state import State

class Scene():
    def __init__(self,dt,runNum,robot_num,if_train,expert_only,use_dagger,desired_distance,stop_thresh,expert_velocity_adjust, fileName = "Untitled", recordData = False):
        ##### useful artributes

        self.t = 0
        self.dt = dt
        self.runNum = runNum
        # formation reference link
        self.xid = State(0.0, 0.0, math.pi / 2)
        self.xi = State(0.0, 0.0, math.pi / 2)
        self.alpha = 1 # desired formation scale
        self.desired_distance=desired_distance
        self.stop_thresh=stop_thresh
        self.robot_num=robot_num
        self.if_train=if_train
        self.expert_only=expert_only
        self.use_dagger=use_dagger
        self.desired_distance=desired_distance
        self.expert_velocity_adjust=expert_velocity_adjust


        self.robots = []
        self.adjMatrix = None

        self.MIN_DISTANCE = 1
        self.MAX_DISTANCE = 5
        # self.Laplacian = None

        # vrep related
        self.vrepConnected = False
        # self.vrepSimStarted = False
        self.SENSOR_TYPE = "None"
        self.objectNames = []
        self.recordData = recordData
        self.occupancyMapType = None
        self.OCCUPANCY_MAP_BINARY = 0
        # 1 for 3-channel: mean height, height variance, visibility
        self.OCCUPANCY_MAP_THREE_CHANNEL = 1
        #log related
        self.errorType = 0
        self.logPriorityMax = 1  # Messages with lower priorities are not logged
        self.logFileName = os.path.splitext(fileName)[0] + ".log"
        self.log('A new scene is created for run #' + str(runNum))


        ##### useless artribute
        # for plots
        self.ts = [] # timestamps
        self.tss = [] # timestamps (sparse)
        self.ydict = dict()
        self.ydict2 = dict()
        self.ploted = dict()

        # For visualization
        self.wPix = 600
        self.hPix = 600
        self.xMax = 8
        self.yMax = 8
        self.image = np.zeros((self.hPix, self.wPix, 3), np.uint8)
        if USE_CV2:
            self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter('output.avi', self.fourcc, 20.0, (self.wPix, self.hPix))
            self.frameCounter = 0



        # follower does not have knowledge of absolute position


##### merge with resetPosition,scaleDesiredFormation
    def addRobot(self, arg,learnedController = None):
        robot = Robot(self,self.robot_num,self.if_train,self.expert_only,self.use_dagger,self.desired_distance,self.expert_velocity_adjust)
        robot.index = len(self.robots)
        robot.xi.x = arg[0, 0]
        robot.xi.y = arg[0, 1]
        robot.xi.theta = arg[0, 2]
        robot.xid.x = arg[1, 0]
        robot.xid.y = arg[1, 1]
        robot.xid.theta = arg[1, 2]
        robot.learnedController = learnedController
        robot.recordData = self.recordData
        self.robots.append(robot)
        message = ""
        message += " robot #" + str(robot.index) + " using "
        if learnedController is None:
            message += "a model-based controller"
        else:
            message += "a leanrned controller"
        message += " is added to the scene"
        self.log(message)

    #####(merge with addRobot)
    def resetPosition(self, radius):
        if radius is None:
            for i in range(0, len(self.robots)):
                self.robots[i].update_pose(None)
            return
        x_average = 0
        y_average = 0
        for i in range(0, len(self.robots)):
            while True:
                minDij = float("inf")
                alpha1 = math.pi * (2 * random.random())  # arbitrary
                rho1 = radius * random.random()
                x1 = rho1 * math.cos(alpha1)
                y1 = rho1 * math.sin(alpha1)
                theta1 = 2 * math.pi * random.random()
                for j in range(0, i):
                    dij = ((x1 - self.robots[j].xi.x) ** 2 +
                           (y1 - self.robots[j].xi.y) ** 2) ** 0.5
                    if dij < minDij:
                        minDij = dij  # find the smallest dij for all j
                dij = ((x1) ** 2 +(y1)**2) ** 0.5
                if dij < minDij:
                    minDij = dij  # find the smallest dij for all j
                print('Min distance: ', minDij, 'from robot #', i, 'to other robots.')
                # if the smallest dij is greater than allowed,
                if i == 0:
                    index, handle, position, orientation=self.robots[i].update_pose([x1, y1, theta1])
                    self.executor_setpose(index, handle, position, orientation)
                    break  # i++
                elif radius >= minDij >= self.MIN_DISTANCE:
                    index, handle, position, orientation = self.robots[i].update_pose([x1, y1, theta1])
                    self.executor_setpose(index, handle, position, orientation)
                    break  # i++
            x_average += x1
            y_average += y1
        self.xi.x = x_average / len(self.robots)
        self.xi.y = y_average / len(self.robots)
        self.xid.dpbarx = self.xi.x - self.xid.x
        self.xid.dpbary = self.xi.y - self.xid.y

    #####(merge with addRobot)
    def scaleDesiredFormation(self, alpha):
        self.alpha = alpha
        for robot in self.robots:
            robot.xid.x *= alpha
            robot.xid.y *= alpha

#####(rewrite)


    def setADjMatrix(self, adjMatrix):
        self.adjMatrix = adjMatrix
        # self.Laplacian = np.diag(np.sum(self.adjMatrix, axis = 1))

    #####(rewrite)
    #get latest communication graph according to robot positions (add for gnn)
    def readADjMatrix(self, MaxRange):
        ADjMatrix = np.zeros((0,len(self.robots)),dtype=np.float32)
        for i in range(0, len(self.robots)):
            for j in range(0, len(self.robots)):
                if i==j:
                    dij = 0
                else:
                    dij = ((self.robots[i].xi.x - self.robots[j].xi.x)**2 +
                           (self.robots[i].xi.y - self.robots[j].xi.y)**2)**0.5
                    if dij<=MaxRange:
                        dij = 1
                    else:
                        dij = 0
                ADjMatrix = np.append(ADjMatrix, dij)
        return ADjMatrix.reshape(1,len(self.robots)*len(self.robots))

#### Connections with Vrep via localhost port
    def initVrep(self):
        print ('Program started')
        vrep.simxFinish(-1) # just in case, close all opened connections
        self.clientID = vrep.simxStart('127.0.0.1',19997,True,True,5000,5) # Connect to V-REP
        if self.clientID!=-1:
            self.vrepConnected = True
            print('Connected to remote API server')
             # enable the synchronous mode on the client:
            vrep.simxSynchronous(self.clientID, True)
            # start the simulation:
            vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)
            # Laser Scanner Initialization
            #if self.SENSOR_TYPE == "2d":

        else:
            self.vrepConnected = False
            print ("Failed connecting to remote API server")
            raise Exception("Failed connecting to remote API server")
#### Get Vrep handles from simulator and pass them to robot_old.py. Handle group of parameters in simulator to define robot and sensor.
    def setVrepHandles(self, robotIndex, handleNameSuffix = ""):
        if self.vrepConnected == False:
            return False
        handleNames = self.objectNames
        res1, robotHandle = vrep.simxGetObjectHandle(
                self.clientID, handleNames[0] + handleNameSuffix,
                vrep.simx_opmode_oneshot_wait)

        res2, motorLeftHandle = vrep.simxGetObjectHandle(
                self.clientID, handleNames[1] + handleNameSuffix,
                vrep.simx_opmode_oneshot_wait)
        res3, motorRightHandle = vrep.simxGetObjectHandle(
                self.clientID, handleNames[2] + handleNameSuffix,
                vrep.simx_opmode_oneshot_wait)
        print("Vrep res: ", res1, res2, res3)
        self.robots[robotIndex].robotHandle = robotHandle
        self.robots[robotIndex].motorLeftHandle = motorLeftHandle
        self.robots[robotIndex].motorRightHandle = motorRightHandle
        #print(self.robots[robotIndex].robotHandle)

        if self.SENSOR_TYPE == "None":
            pass
        elif self.SENSOR_TYPE == "2d":
            res, laserFrontHandle = vrep.simxGetObjectHandle(
                    self.clientID, handleNames[3] + handleNameSuffix,
                    vrep.simx_opmode_oneshot_wait)
            print('2D Laser (front) Initilization:', 'Successful' if not res else 'error')
            res, laserRearHandle = vrep.simxGetObjectHandle(
                    self.clientID, handleNames[4] + handleNameSuffix,
                    vrep.simx_opmode_oneshot_wait)
            print('2D Laser (rear) Initilization:', 'Successful' if not res else 'error')
            self.robots[robotIndex].laserFrontHandle = laserFrontHandle
            self.robots[robotIndex].laserRearHandle = laserRearHandle
            self.robots[robotIndex].laserFrontName = handleNames[3] + handleNameSuffix
            self.robots[robotIndex].laserRearName = handleNames[4] + handleNameSuffix
        elif self.SENSOR_TYPE == "VPL16":
            res, pointCloudHandle = vrep.simxGetObjectHandle(
                    self.clientID, handleNames[3] + handleNameSuffix,
                    vrep.simx_opmode_oneshot_wait)
            print('Point Cloud Initilization:', 'Successful' if not res else 'error')
            self.robots[robotIndex].pointCloudHandle = pointCloudHandle
            self.robots[robotIndex].pointCloudName = handleNames[3] + handleNameSuffix
        elif self.SENSOR_TYPE == "kinect":
            res, kinectDepthHandle = vrep.simxGetObjectHandle(
                    self.clientID, handleNames[3] + handleNameSuffix,
                    vrep.simx_opmode_oneshot_wait)
            print('Kinect Depth Initilization: ', 'Successful' if not res else 'error')
            res, kinectRgbHandle = vrep.simxGetObjectHandle(
                    self.clientID, handleNames[4] + handleNameSuffix,
                    vrep.simx_opmode_oneshot_wait)
            print('Kinect RGB Initilization: ', 'Successful' if not res else 'error')
            self.robots[robotIndex].kinectDepthHandle = kinectDepthHandle
            self.robots[robotIndex].kinectRgbHandle = kinectRgbHandle
            self.robots[robotIndex].kinectDepthName = handleNames[3] + handleNameSuffix
            self.robots[robotIndex].kinectRgbName = handleNames[4] + handleNameSuffix
        #self.robots[robotIndex].setPosition()
        pos, ori, vel, omega,velodyne_points= self.read_sensor(self.robots[robotIndex])
        # print(pos, ori, vel, omega,velodyne_points)
        self.robots[robotIndex].get_sensor_data(pos, ori, vel, omega,velodyne_points)
#### Simulate for one simulation time step. Get sensor data, calculate control, simulate robot's movements
    def read_sensor(self,robot):

        ##### Read absolute pose from simulator
        if self.vrepConnected == False:
            return
        if "readSensorData_firstCall" not in self.__dict__:
            robot.readSensorData_firstCall = True
        else:
            robot.readSensorData_firstCall = False

        # Read robot states
        res, pos = vrep.simxGetObjectPosition(self.clientID,
                                              robot.robotHandle, -1,
                                              vrep.simx_opmode_blocking)
        if res != 0:
            raise VrepError("Cannot get object position with error code " + str(res))
        res, ori = vrep.simxGetObjectOrientation(self.clientID,
                                                 robot.robotHandle, -1,
                                                 vrep.simx_opmode_blocking)
        if res != 0:
            raise VrepError("Cannot get object orientation with error code " + str(res))
        res, vel, omega = vrep.simxGetObjectVelocity(self.clientID,
                                                     robot.robotHandle,
                                                     vrep.simx_opmode_blocking)
        if res != 0:
            raise VrepError("Cannot get object velocity with error code " + str(res))
        ##### Read sensors

        if self.SENSOR_TYPE == "2d_":
            # self.laserFrontHandle
            # self.laserRearHandle

            if self.readSensorData_firstCall:
                opmode = vrep.simx_opmode_streaming
            else:
                opmode = vrep.simx_opmode_buffer
            laserFront_points = vrep.simxGetStringSignal(
                    self.clientID, robot.laserFrontName + '_signal', opmode)
            print(robot.laserFrontName + '_signal: ', len(laserFront_points[1]))
            laserRear_points = vrep.simxGetStringSignal(
                    self.clientID, robot.laserRearName + '_signal', opmode)
            print(robot.laserRearName + '_signal: ', len(laserRear_points[1]))
        elif self.SENSOR_TYPE == "2d": # deprecated
            raise Exception('2d sensor is not supported!!!!')
        elif self.SENSOR_TYPE == "VPL16":
            # self.pointCloudHandle
            velodyne_points = vrep.simxCallScriptFunction(
                    self.clientID, robot.pointCloudName, 1,
                    'getVelodyneData_function', [], [], [], 'abc',
                    vrep.simx_opmode_blocking)
            res = velodyne_points[0]
            # Parse data
        elif self.scene.SENSOR_TYPE == "kinect":
            pass
        else:
            return
        return pos,ori,vel,omega,velodyne_points
    def propagate(self,robot,omega1,omega2):
        #### Set the linear velocity of 2 wheels
        if self.vrepConnected == False:
            robot.xi.propagate(robot.control)
        else:

            vrep.simxSetJointTargetVelocity(self.clientID,
                                            robot.motorLeftHandle,
                                            omega1, vrep.simx_opmode_oneshot)
            vrep.simxSetJointTargetVelocity(self.clientID,
                                            robot.motorRightHandle,
                                            omega2, vrep.simx_opmode_oneshot)
    def mock_simulator(self,robot):
        v1=robot.wheel_velocity_1
        v2=robot.wheel_velocity_2
        dt = robot.scene.dt
        l = robot.l
        d_x = math.cos(robot.xi.theta) * dt / 2 * (v1 + v2)
        d_y = math.sin(robot.xi.theta) * dt / 2 * (v1 + v2)
        d_theta = 1 / l * dt * (v2 - v1)
        new_x=robot.xi.x+d_x
        new_y=robot.xi.y+d_y
        new_theta=robot.xi.theta+d_theta
        pos=[new_x,new_y]
        ori=[0,0,new_theta]
        vel=[v1,v2]
        omega=[0,0,1 / l * dt * (v2 - v1)]
        velodyne_points=None
        return pos, ori, vel, omega, velodyne_points
    def get_average_gaberil_distance_error(self):
        # print("Distance")
        node_mum = len(self.robots)
        gabriel_graph = [[1] * node_mum for _ in range(node_mum)]
        position_list = []
        for i in range(node_mum):
            position = [self.robots[i].xi.x, self.robots[i].xi.y]
            position_list.append(position)
        position_array = np.array(position_list)
        for u in range(node_mum):
            for v in range(node_mum):
                m = (position_array[u] + position_array[v]) / 2
                for w in range(node_mum):
                    if w == v:
                        continue
                    if np.linalg.norm(position_array[w] - m) < np.linalg.norm(position_array[u] - m):
                        gabriel_graph[u][v] = 0
                        gabriel_graph[v][u] = 0
                        break
        total=0
        count=0
        for i in range(node_mum):
            for j in range(i + 1, node_mum):
                if gabriel_graph[i][j] == 1:
                    distance_error = abs(math.sqrt((self.robots[i].xi.x - self.robots[j].xi.x) ** 2 + (
                                self.robots[i].xi.y - self.robots[j].xi.y) ** 2)-self.desired_distance)
                    # print("distance between {r1:d} and {r2:d}".format(r1=i, r2=j))
                    # print(distance)
                    total+=distance_error
                    count+=1

        return total/count
    def check_stop_condition(self,desire_distance,thresh):
        print("Distance")

        node_mum = len(self.robots)
        gabriel_graph = [[1] * node_mum for _ in range(node_mum)]
        position_list=[]
        for i in range(node_mum):
            position=[self.robots[i].xi.x,self.robots[i].xi.y]
            position_list.append(position)
        position_array = np.array(position_list)
        for u in range(node_mum):
            for v in range(node_mum):
                m = (position_array[u] + position_array[v]) / 2
                for w in range(node_mum):
                    if w == v:
                        continue
                    if np.linalg.norm(position_array[w] - m) < np.linalg.norm(position_array[u] - m):
                        gabriel_graph[u][v] = 0
                        gabriel_graph[v][u] = 0
                        break
        stop=True
        for i in range(node_mum):
            for j in range(i+1,node_mum):
                if gabriel_graph[i][j]==1:
                    distance=math.sqrt((self.robots[i].xi.x-self.robots[j].xi.x)**2+(self.robots[i].xi.y-self.robots[j].xi.y)**2)
                    print("distance between {r1:d} and {r2:d}".format(r1=i,r2=j))
                    print(distance)
                    if abs((distance-desire_distance)/desire_distance)>thresh:
                        stop=False
        return stop
    def simulate(self):
        # vrep related
        '''
        cmd = input('Press <enter> key to step the simulation!')
        if cmd == 'q': # quit
            return False
        '''

        self.t += self.dt
        self.ts.append(self.t)
        self.propagateXid()
        countReachedGoal = 0
        omlist = []
        print(self.t)
        for robot in self.robots:
            robot.precompute(self.adjMatrix,self.robots)
        for robot in self.robots:
            pos, ori, vel, omega, velodyne_points = self.read_sensor(robot)
            robot.get_sensor_data(pos, ori, vel, omega, velodyne_points)
            # print("vel")
            # print(vel)
            # robot.xid.theta = self.xid.vRefAng
            o,g,r,a = robot.getDataObs()
            omlist.append((o,g,r,a))
        average_distance_gabreil_error = self.get_average_gaberil_distance_error()
        for i in range(len(self.robots)):
            o1,o2 = self.robots[i].get_control(omlist,i,average_distance_gabreil_error,self.stop_thresh)
            self.robots[i].record_state()
            self.propagate(self.robots[i],o1,o2)
            if self.robots[i].reachedGoal:
                countReachedGoal += 1
        self.calcCOG()
        if self.vrepConnected:
            vrep.simxSynchronousTrigger(self.clientID)
        if countReachedGoal == len(self.robots):
            return False
        else:
            return True

    def executor_setpose(self,index,handle,position,orientation):
        if self.vrepConnected == False:
            return
        vrep.simxSetObjectPosition(self.clientID, handle, -1,
                                   position, vrep.simx_opmode_oneshot)
        vrep.simxSetObjectOrientation(self.clientID, handle, -1,
                                      orientation, vrep.simx_opmode_oneshot)
        message = "Robot #" + str(index) + "'s pose is set to "
        message += "[{0:.3f}, {1:.3f}, {2:.3f}]".format(position[0], position[1], orientation[-1])
        self.log(message)
    def propagateXid(self):
        #### For simulate()
        t = self.t
        dt = self.dt
        sDot = 0
        thetaDot = 0

        omega = 0
        self.xid.dpbarx = -self.xid.vRefMag * math.cos(self.xid.vRefAng + self.t * omega)
        self.xid.dpbary = -self.xid.vRefMag * math.sin(self.xid.vRefAng + self.t * omega)
        #print('dpbarx: ', self.xid.dpbarx, ', dpbary: ', self.xid.dpbary)
    def calcCOG(self):
        #### For simulate()
        # Calculate Center Of Gravity
        for i in range(len(self.robots)):
            x = self.robots[i].xi.x
            y = self.robots[i].xi.y
            if len(self.ts) == 1:
                if i == 0:
                    self.centerTraj = np.array([[x, y]])
                else:
                    self.centerTraj += np.array([[x, y]])
            else:
                if i == 0:
                    self.centerTraj = np.append(self.centerTraj, [[x, y]], axis = 0)
                else:
                    #print('size', self.centerTraj.shape)
                    self.centerTraj[-1, :] += np.array([x, y])
            #print(self.centerTraj)
        self.centerTraj[-1, :] /= len(self.robots)

    ###### leave here for future needs
    def getRobotColor(self, i, brightness = 0.7, reverse = False):
        #### maybe useless
        if i == 0:
            c = (brightness, 0, 0)
        elif i == 1:
            c = (0, brightness, 0)
        elif i == 2:
            c = (0, 0, brightness)
        elif i == 3:
            c = (0, brightness, brightness)
        else:
            c = (brightness, 0, brightness)
        if reverse == True:
            return c[::-1]
        else:
            return c
    def showOccupancyMap(self, waitTime = 25):
        #### maybe useless
        if USE_CV2 == False:
            return
        pc = self.robots[0].pointCloud
        wPix = pc.wPix
        hPix = pc.hPix
        N = len(self.robots)
        resizeFactor = int(500/hPix)
        if self.occupancyMapType == self.OCCUPANCY_MAP_BINARY:
            self.occupancyMap = np.ones((hPix, (wPix+1) * N), np.uint8) * 255
            x0 = 0
            for robot in self.robots:
                x1 = x0 + wPix
                self.occupancyMap[:, x0:x1] = robot.pointCloud.occupancyMap
                self.occupancyMap[:, x1:(x1+1)] = np.zeros((hPix, 1), np.uint8)
                x0 += wPix + 1
            #print('self.occupancyMap shape: ', self.occupancyMap.shape)

            im = cv2.resize(self.occupancyMap,
                            (self.occupancyMap.shape[1] * resizeFactor,
                             self.occupancyMap.shape[0] * resizeFactor),
                            interpolation = cv2.INTER_NEAREST)
            cv2.imshow('Occupancy Map', im)
        elif self.occupancyMap == self.OCCUPANCY_MAP_THREE_CHANNEL:
            self.occupancyMap = np.zeros((hPix, (wPix+1) * N, 3), np.uint8)
            x0 = 0
            for robot in self.robots:
                x1 = x0 + wPix
                self.occupancyMap[:, x0:x1, :] = robot.pointCloud.occupancyMap
                self.occupancyMap[:, x1:(x1+1), :] = np.ones((hPix, 1, 3), np.uint8) * 255
                x0 += wPix + 1
            #print('self.occupancyMap shape: ', self.occupancyMap.shape)
            im = cv2.resize(self.occupancyMap,
                            (self.occupancyMap.shape[1] * resizeFactor,
                             self.occupancyMap.shape[0] * resizeFactor),
                            interpolation = cv2.INTER_NEAREST)
            cv2.imshow('Occupancy Map', im)
        cv2.waitKey(waitTime)
    def getMaxFormationError(self):
        # if 2 not in self.ydict.keys():
        #     raise Exception('Plot type 2 must be drawn in order to get formation error!')
        if self.errorType == 0:
            errors = self.ydict[2]
        else:
            errors = self.ydict[3]
        # check max formation error
        maxAbsError = 0
        for key in errors:
            absError = abs(errors[key][-1])
            if absError > maxAbsError:
                maxAbsError = absError
        return maxAbsError
    def deallocate(self):
        self.log("Scene is destructed")
        if USE_CV2 == True:
            cv2.destroyAllWindows() # Add this to fix the window freezing bug
            self.out.release()
        # vrep related
        if self.vrepConnected:
            self.vrepConnected = False
            # Before closing the connection to V-REP, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
            #vrep.simxGetPingTime(self.clientID)
            # Stop simulation:
            vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)
            # Now close the connection to V-REP:
            vrep.simxFinish(self.clientID)
    def log(self, message, priority=1):
        if priority <= self.logPriorityMax:
            with open(self.logFileName, "a+" ) as f:
                prefix = ("[" + str(datetime.datetime.now()) + "]"
                            + " [run #{0:03d}]"
                            + " [sim time: {1:.3f} s] ")
                prefix = prefix.format(self.runNum, self.t)
                f.write(prefix + message + '\n')
    def save_robot_states(self,path):
        pose_list=[]
        velocity_list=[]
        for i in range(len(self.robots)):
            pose_list.append(self.robots[i].pose_list)
            velocity_list.append(self.robots[i].velocity_list)
        pose_array=np.array(pose_list)
        velocity_array=np.array(velocity_list)
        if not os.path.exists(path):
            os.makedirs(path)
        pose_path = os.path.join(path,"pose_array_scene.npy")
        velocity_path = os.path.join(path,"velocity_array_scene.npy")
        np.save(pose_path, pose_array)
        np.save(velocity_path, velocity_array)
        print("Pose array of scene"+" saved at " + pose_path)
        print("Velocity array of robot"+" saved at " + velocity_path)


class VrepError(Exception):
    # Exception raised for errors related vrep.
    def __init__(self, message):
        self.message = message

