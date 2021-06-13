# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 23:09:34 2017

@author: cz
"""

from scene import Scene
from sceneplot import ScenePlot
from robot import Robot
import numpy as np

try:
    sc = Scene()
    sp = ScenePlot(sc)
    dynamics = 11
    sc.dynamics = dynamics
    sc.addRobot(np.float32([[-2, 0, 0], [0, 2, 0]]))
    sc.addRobot(np.float32([[1, 3, 0], [1.732, -1, 0]]))
    sc.addRobot(np.float32([[0, 0, 0], [-1.732, -1, 0]]))
    
    sc.setADjMatrix(np.uint8([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))
    
    #sc.renderScene(waitTime = 3000)
    tf = 5
    while sc.simulate():
        #sc.renderScene(waitTime = 50)
        print('t = ', sc.t, 's')
        
        sp.plot(0, tf)
        sp.plot(1, tf) 
        if sc.t > tf:
            break
    
        #print('robot 0: ', sc.robots[0].xi.x, ', ', sc.robots[0].xi.y, ', ', sc.robots[0].xi.theta)
        #print('robot 1: ', sc.robots[1].xi.x, ', ', sc.robots[1].xi.y, ', ', sc.robots[1].xi.theta)
        #print('robot 2: ', sc.robots[2].xi.x, ', ', sc.robots[2].xi.y, ', ', sc.robots[2].xi.theta)
        #print('y01: ' + str(sc.robots[1].xi.y - sc.robots[0].xi.y))
        #print('x02: ' + str(sc.robots[2].xi.x - sc.robots[0].xi.x))
    sc.deallocate()
except KeyboardInterrupt:
    sc.deallocate()
except:
    sc.deallocate()
    raise

