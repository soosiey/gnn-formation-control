# Learning End-to-End Decentralized Formation Using Graph Neural
Networks

Core GNN Library used from: [https://github.com/proroklab/gnn_pathplanning](https://github.com/proroklab/gnn_pathplanning) 

Code for paper "Learning End-to-End Decentralized Formation Using Graph Neural
Networks"

## Installation
### System requirement
Ubuntu 18.04LTS\
Python 3.6
### Install coppeliaSim
https://coppeliarobotics.com/downloads
### Install Opencv
sudo apt install python3-opencv
### Install Dependencies:

```
pip3 install -r requirements.txt
```
****


## Testing
### Start coppeliasim 
```
./"path to coppeliasim"/coppeliaSim.sh
```

### Always make sure the number of robots in the simulator equals to the number of robot in Demo.py
### Run triangulation formation with default configuration
Open scene_5.ttt(in scene folder) in coppeliaSim
```
python Test.py
```
****

### Run circle formation with default configuration
Open scene_circle.ttt(in scene folder) in coppeliaSim
```
python Test_cirlce.py
```
****

### Run lin formation with default configuration
Open scene_line.ttt(in scene folder) in coppeliaSim
```
python Test_line.py
```
****


## Training
### Training with default configuration
Open scene_5.ttt(in scene folder) in coppeliaSim
### Open scene_5.ttt(in scene folder) in coppeliaSim
```
python Train.py
```




