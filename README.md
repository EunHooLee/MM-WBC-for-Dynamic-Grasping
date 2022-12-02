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
```
Execute sequence (Exited from last number)
1, core.py - Env
2, envs/registration.py - EnvSpec
3, envs/registration.py - make
4, envs/mujoco/waler2d_v4.py - Walker2DEnv
5, envs/mujoco/mujoco_env.py - MujocoEnv
6, envs/mujoco/mujoco_env.py - BaseMujocoEnv
```
- 5,6 번은 mujoco와 직접 연동되어서 mujoco로 계산된 데이터(action)를 전달하고, 변화된 상황(next state)을 반환하는 기능을 한다. 
- 3 번은 5,6,을 wrapping 하는 class로 5,6의 데이터를 이용해 step, get_obs등의 main() 에서 사용될만한 것들을 core.py 의 Env 형태로 정의하고 있다.
- 2 번은 자료형 정리
- 1 번은 기본 틀만 있고 함수 내부는 구현되지 않았다. 아마 4번에서 구현된것이 그대로 사용되는 wrapper class 역할을 하고 있는 것 같다. 

우리는 MujocoEnv or MujocopyEnv 둘 중 하나를 선택해 사용하고, BaseMujocoEnv 까지는 그대로 사용해야 될 것 같다
내 생각대로라면 4번만 혹은 1,4 번만 배껴서 새로 만들면 될 것 같다. 

Readme Fix


## Installation


## Usage

## Citation
