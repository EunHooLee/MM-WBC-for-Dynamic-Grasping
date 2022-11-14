from gymnasium.envs.registration import register

from gymnasium_robotics.core import GoalEnv


def register_robotics_envs():
    def _merge(a, b):
        a.update(b)
        return a

    for reward_type in ["sparse", "dense"]:
        suffix = "Dense" if reward_type == "dense" else ""
        kwargs = {
            "reward_type": reward_type,
        }

        # Fetch

        register(
            id=f"FetchPickAndPlace{suffix}-v1",
            entry_point="gymnasium_robotics.envs:MujocoPyFetchPickAndPlaceEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )


__version__ = "1.1.0"
