# 🤩 运行 DriveArena 模拟！

TrafficManager、WorldDreamer 和 DrivingAgent 之间的通信基于 **FastAPI**。

⚠️⚠️**WorldDreamer** 和 **DrivingAgent** 可以在**远程服务器**上运行，而 **TrafficManager** 需要在**带有显示器的本地机器**上运行。

## 步骤一：启动 WorldDreamer 服务

请按照 WorldDreamer中的 README.md 准备环境并下载权重。

然后你可以运行以下代码：

原单帧版（BaseDreamer）
```shell
cd WorldDreamer/BaseDreamer && python tools/dreamer_fast_api.py --resume=path/to/your/weight
```

时序版（DreamForge）
```shell
cd WorldDreamer/DreamForge && python tools/run_fastapi.py --model_single ./pretrained/dreamforge-s --model ./pretrained/dreamforge-t
```

## 步骤二：启动 DrivingAgent 服务

### UniAD
请按照 [README.md](../DrivingAgents/UniAD/README_CN.md) 准备环境并下载权重。

```shell
cd DrivingAgents/UniAD && python demo/fast_api_uniad.py
```

### VAD
请按照 [README.md](../DrivingAgents/VAD/README_CN.md) 准备环境并下载权重。

```shell
cd DrivingAgents/VAD && python demo/fast_api_vad.py
```

## 步骤三：启动 TrafficManager 并开始模拟
请按照 [README.md](../TrafficManager/README.md) 启动模拟。

如果一切顺利，你将看到一个类似这样的窗口!
![alt text](../assets/simulation.png)
