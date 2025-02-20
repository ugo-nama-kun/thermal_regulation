# Thermal Environment

![thermal-ns](/uploads/c1a2a124fb6b6a7ee4cdc4cf03573515/thermal-ns.png)

```python
import gym
import thermal_regulation

env = gym.make(
            SmallLowGearAntTHR-v3,
            random_climate=True # ENABLE CLOUDY OR SUNNY RANDOM DAYS EVERY RESET
        )
```
