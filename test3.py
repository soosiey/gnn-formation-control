# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 23:09:34 2017
This test file is dependent on vrep.

@author: cz
"""

from scene import Scene
from sceneplot import ScenePlot
# from robot import Robot
import numpy as np
# from data import Data

def generateData():
    sc = Scene(recordData = True)
    sp = ScenePlot(sc)
    try:
        dynamics = 11
        sc.dynamics = dynamics
        sc.addRobot(np.float32([[-2, 0, 0], [0, 2/2, 0]]))
        sc.addRobot(np.float32([[1, 3, 0], [1.732/2, -1/2, 0]]))
        sc.addRobot(np.float32([[0, 0, 0], [-1.732/2, -1/2, 0]]))
        
        # No leader
        sc.setADjMatrix(np.uint8([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))
        # Set robot 0 as the leader.
        # sc.setADjMatrix(np.uint8([[0, 0, 0], [1, 0, 1], [1, 1, 0]]))
        
        # vrep related
        sc.initVrep()
        # Choose sensor type
        sc.SENSOR_TYPE = "VPL16" # None, 2d, VPL16, kinect
        sc.objectNames = ['Pioneer_p3dx', 'Pioneer_p3dx_leftMotor', 'Pioneer_p3dx_rightMotor']
        
        if sc.SENSOR_TYPE == "None":
            pass
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
        tf = 10
        sc.resetPosition()
        sp.plot(3, tf)
        while sc.simulate():
            #sc.renderScene(waitTime = int(sc.dt * 1000))
            sc.showOccupancyMap(waitTime = int(sc.dt * 1000))
            
            #print("---------------------")
            #print("t = %.3f" % sc.t, "s")
            
            #sp.plot(0, tf)
            sp.plot(2, tf)
            #sp.plot(1, tf) 
            sp.plot(3, tf)
            sp.plot(4, tf)
            sp.plot(5, tf)
            sp.plot(6, tf)
            if sc.t > tf:
                break
                
        
            #print('robot 0: ', sc.robots[0].xi.x, ', ', sc.robots[0].xi.y, ', ', sc.robots[0].xi.theta)
            #print('robot 1: ', sc.robots[1].xi.x, ', ', sc.robots[1].xi.y, ', ', sc.robots[1].xi.theta)
            #print('robot 2: ', sc.robots[2].xi.x, ', ', sc.robots[2].xi.y, ', ', sc.robots[2].xi.theta)
            #print('y01: ' + str(sc.robots[1].xi.y - sc.robots[0].xi.y))
            #print('x02: ' + str(sc.robots[2].xi.x - sc.robots[0].xi.x))
        sc.deallocate()
    except KeyboardInterrupt:
        x = input('Quit?(y/n)')
        sc.deallocate()
        if x == 'y' or x == 'Y':
            raise Exception('Aborted.')
        
    except:
        sc.deallocate()
        raise
    
    # check max formation error
    maxAbsError = 0
    for key in sc.ydict[2]:
        absError = abs(sc.ydict[2][key][-1])
        if absError > maxAbsError:
            maxAbsError = absError
    print('maxAbsError = ', maxAbsError)
    
    if maxAbsError < 0.5:
        return sc
    else:
        return None
# main
numRun = 100
dataList = []




for i in range(0, numRun):
    print('Run #: ', i, '...')
    # First episode
    sc = generateData()
    if sc != None:
        # if the list is empty
        if not dataList:
            for robot in sc.robots:
                dataList.append(robot.data)
        else:
            for j in range(len(sc.robots)):
                dataList[j].append(sc.robots[j].data)
        

for j in range(len(sc.robots)):
    dataList[j].store()

































