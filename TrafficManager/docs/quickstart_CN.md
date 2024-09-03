## 运行代码

以下所有代码请在具有显示器的PC下进行。

**1. 安装python 依赖项**
```bash
cd TrafficManager
conda create --name drivearena_tf python=3.8
conda activate drivearena_tf
pip install -r requirements.txt
```

配置完成后，可进行初步验证：
```bash
cd ../ # under DriveArena root folder
python ./TrafficManager/sim_manager_only.py
``` 
若程序可正常运行，GUI正常显示，则继续下一步。

**2. 在服务器上或者本机上部署 WorldDreamer及 Driving Agent 服务**   
详细说明： <span style="color: red;">此处需要link到相关文档</span>

**3. 进行端口映射**
如果WorldDreamer和Driving Agent在本地运行或可直接通过公网访问，可跳过此步骤。
将服务器端的服务端口映射到本机:
```bash
# Assume you can connect to server as: ssh username@server_adress -p server_port
ssh -N username@server_adress -p server_port -L 11000:localhost:server_diffusion_port -L 11001:localhost:server_driver_port
```
请保持该终端窗口开启。

**4. 配置设置**
在`DriveArena/config.yaml`文件中确认ip和端口无误：
    
```yaml
servers:
    diffusion: "http://127.0.0.1:11000/"
    driver: "http://127.0.0.1:11001/"
```

**5. 运行程序**
```bash
# under DriveArena root folder
python ./TrafficManager/sim_manager.py
``` 
程序应当显示如下画面。

程序会在ego_agent 完成规定路径或 ego_agent 发生碰撞后停止，您也可以随时手动终止程序。

**6. 查看结果**
   每次运行的结果文件都保存在`DriveArena/results/`目录的子文件夹中，以运行时间`mm-dd-hhmmss`命名。
   - 计算Driving Score分数。
        ```bash
        python ./TrafficManager/score_calculator.py ./results/mm-dd-hhmmss/ # modify to real path
        ```
   - 制作视频。
        结果文件夹中的`imgs/`子文件夹包含每帧WorldDreamer结果、GT BEV和Agent预测BEV的图像。使用以下脚本将它们制作成视频:

        ```bash
        python ./TrafficManager/generate_video.py --output_dir   ./results/mm-dd-hhmmss/ # modify to real path
        ```

        即可在`mm-dd-hhmmss`文件夹下生成名为`output_video.mp4`的文件。
