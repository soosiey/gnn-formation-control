# Note

+ **Robot Position**:  sc.robots[0].setPosition [x(m),y(m),theta(rad)]
+ **Robot velocity**: v1,v2,v1nn,v2nn(linear velocity)

## API interface

1. **sim.py**  
    Real Vrep apis call functions in sim.py to communicate with Vrep
    1. simxStart  
       Start connection.
    2. simxFinish  
       Finish connection.
    3. simxSynchronous  
       Enable the synchronous mode on the client.
    4. simxStartSimulation  
       Start simulation.
    5. simxStopSimulation
       End simulation.
    6. simxGetObjectHandle  
       Get objects(ie. robot, sensor) handle.
       Handles are a group of parameters in simulator to define robot and sensor.
    7. simxSynchronousTrigger
    8. simxSetJointTargetVelocity  
       Set robot wheel's linear velocities.
    9. simxGetJointTargetVelocity  
       Get robot wheel's linear velocities.
    10. simxSetObjectPosition  
        Set robots' position .
    11. simxSetObjectOrientation  
        Set robots' orientation.
    12. simxGetObjectPosition  
        Get robots' position.
    13. simxGetObjectOrientation  
        Get robots' orientation.
    14. simxGetStringSignal  
    15. simxCallScriptFunction
    
2. **scene.py**
    1. initVrep  
        Connections with Vrep via localhost port  .
    2. setVrepHandles  
        Get Vrep handles from simulator and pass them to robot.py.      
    3. simulate  
        Simulate for one simulation time step. 
        Get sensor data, calculate control, simulate robot's movements.
3. **robot.py**
    1. propagate  
        Set the linear velocity of 2 wheels.
    2. setPosition  
        Set one robot's Position and Orientation.
    3. readSensorData  
        Get one robot's Position, Orientation and Lidar reading.
#### Author Xinchi Huang





