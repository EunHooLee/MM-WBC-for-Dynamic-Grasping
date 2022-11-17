import gymnasium as gym

from wbc4dg.envs.mujoco import MujocoMMDynamicGraspingEnv


"""
남은 것

1. mobile_manipulator_env.py/ BaseMMEnv/ compute_reward() 구현 ** 
2. mobile_manipulator_env.py/ BaseMMEnv/ _sample_goal() 구현 : 이거 뭔지 모르겠음. goal은 계속 똑같은거아닌가?
3. mobile_manipulator_env.py/ MujocoMMEnv/ _set_action() 내부 object 경로 구현 -> 함수나 클래스 이용해서 매번 랜덤하게 움직을 수 있게 하기 (+ rotation motion 구현(필요시))
4. robot_env.py/ BaseRobotEnv/ compute_terminated() 함수 구현 -> 이거 내가 구현해야되는건지 모르겠다, 내생각에는 저렇게 냅둬도 될꺼같다.
5. robot_env.py/ BaseRobotEnv/ compute_truncated() 함수 구현 -> 이거 내가 구현해야되는건지 모르겠다, 내생각에는 저렇게 냅둬도 될꺼같다.
6. 주석달기
"""


def  main():
    env = MujocoMMDynamicGraspingEnv(reward_type="dense",)

    observation, info = env.reset()
    
    max_timesteps=10000
    for _ in range(max_timesteps):
        # action limit 내에서 action sample
        action = env.action_space.sample()
        # step
        observation, reward, terminated, truncated, info = env.step(action)
        
    env.close()


if __name__=='__main__':
    main()