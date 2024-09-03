#  🤩 Run DriveArena Demo!

The communication between TrafficManager, WorldDreamer and DrivingAgent is based on **FastAPI**. **WorldDreamer** and **DrivingAgent** can be run on the **remote server**, while **TrafficManager** needs to be run on **a local machine with a screen**.

## Launch WorldDreamer Service
Please follow the [README.md](../WorldDreamer/README.md) to prepare the environment and download the weights.

Then you can run the following code.
```shell
cd WorldDreamer && python tools/dreamer_fast_api.py
```

## Launch DrivingAgent Service
Please follow the [README.md](../DrivingAgents/UniAD/README.md) to prepare the environment and download the weights.

```shell
cd DrivingAgents/UniAD && python demo/fast_api_uniad.py --resume=path/to/your/weight
```

## Launch TrafficManager and Start Simulation
Please follow the [README.md](../TrafficManager/README.md) to start the simulation.

If everything is all right, you can see a window like this!

![alt text](../assets/simulation.png)