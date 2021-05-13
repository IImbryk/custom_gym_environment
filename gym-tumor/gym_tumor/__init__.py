from gym.envs.registration import register

register(
    id='tumor-v1',
    entry_point='gym_tumor.envs:TumorEnv',
)
register(
    id='tumor-extrahard-v1',
    entry_point='gym_tumor.envs:TumorExtraHardEnv',
)
