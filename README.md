# Thermal Environment

Installation and usage are similar to: https://github.com/ugo-nama-kun/trp_env

```python
import gym
import thermal_regulation

env = gym.make(
            SmallLowGearAntTHR-v3,
            random_climate=True # ENABLE CLOUDY OR SUNNY RANDOM DAYS EVERY RESET
        )
```
