import matplotlib.pyplot as plt
import numpy as np

from thermal_regulation.envs.low_gear_ant_thr_env import LowGearAntSmallThermalRegulationEnv

interoception = []

env = LowGearAntSmallThermalRegulationEnv()
env.reset()
for i in range(5000):
    # action = np.ones_like(env.action_space.sample())
    action = env.action_space.sample()
    # action[-1] = -1
    if int(i / 1000) % 3 == 0 or int(i / 1000) % 3 == 1:
        action *= 0
        action[-1] = np.random.uniform(-1, 1)

    obs, rew, done, info = env.step(action)
    interoception.append(env.get_interoception())
    # env.render()
    if done:
        break

env.close()

interoception = np.array(interoception)
plt.plot(interoception)
plt.legend(["energy", "temp"])
plt.show()

