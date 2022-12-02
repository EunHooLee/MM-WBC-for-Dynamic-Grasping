# Mobile Manipulator Whole Body Control for Efficient Dynamic Grasping for a Random Trajectory Object conditioned on Reachability
Code release for the xx 2023 paper:
## Dependency
* Ubuntu==20.04.5 LTS
* conda==22.9.0
* python==3.7.13
* gymnasium==0.26.3 (gymnasium-robotics install시 같이 자동 다운됨)
* stable-baselines3==1.6.2
* mujoco==2.2.2 (gymnasium-robotics install시 같이 자동 다운됨)
* mujoco_py=2.1.2.14
* torch==1.12.0
* torchvision==0.13.0
## Comment
파일 다운하시고, 
```
$ cd
$ cd .mujoco/mujoco210/bin/
$ ./simulate
```
해서 mujoco simulator 실행 후 
```
dynamic_grasping.xml
```
파일을 simulator에 끌어다 놓으면 model 확인할 수 있습니다.

오른쪽에 joint property를 조정해 mobile manipulator를 움직여 볼 수 있습니다.

<img src="./figures/Mobile_Manipulator.png" width= "400px" height="350px" alt="Mobile Manipulator"></img>

---
gym 중요파일의 대략적인 실행 구조
```python
$ import gym
$ env = gym.make('Walker2d-v4')
```

## Installation


## Usage

## Citation
