from gym.envs.registration import register

register(
    id='myenv_3D-v0',
    entry_point='myenv_3D_COMP.env:MyEnv'
)
