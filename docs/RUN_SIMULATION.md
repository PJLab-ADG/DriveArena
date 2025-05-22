#  🤩 Run DriveArena Demo!

The communication between TrafficManager, WorldDreamer and DrivingAgent is based on **FastAPI**. 

⚠️⚠️**WorldDreamer** and **DrivingAgent** can be run on the **remote server**, while **TrafficManager** needs to be run on **a local machine with a screen**.

## Step 1: Launch WorldDremer Service
Please follow the README.md in WorldDreamer to prepare the environment and download the weights.

Then you can run the following code.

For original single-frame version (BaseDreamer):
```shell
cd WorldDreamer/BaseDreamer && python tools/dreamer_fast_api.py --resume=path/to/your/weight
```

For temporal version (DreamForge):
```shell
cd WorldDreamer/DreamForge && python tools/run_fastapi.py --model_single ./pretrained/dreamforge-s --model ./pretrained/dreamforge-t
```

For temporal version (DreamForge-DiT):
```shell
cd WorldDreamer/DreamForge-DiT && python tools/run_fastapi_dit_t.py
```

## Step 2: Launch DrivingAgent Service

### UniAD
Please follow the [README.md](../DrivingAgents/UniAD/README.md) to prepare the environment and download the weights.

```shell
cd DrivingAgents/UniAD && python demo/fast_api_uniad.py
```

### VAD
Please follow the [README.md](../DrivingAgents/VAD/README.md) to prepare the environment and download the weights.

```shell
cd DrivingAgents/VAD && python demo/fast_api_vad.py
```

## Step 3: Launch TrafficManager and Start Simulation
Please follow the [README.md](../TrafficManager/README.md) to start the simulation.

If everything is all right, you can see a window like this!

![alt text](../assets/simulation.png)