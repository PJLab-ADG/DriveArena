# 自定义地图配置

## 环境配置

1. python配置MapTR和LimSim的环境

2. python安装nuscenes-devkit，并且修改map_api.py文件，用本路径下的map_api.py替换`miniconda3/envs/[env_name]/lib/python3.8/site-packages/nuscenes/map_expansion/map_api.py`

3. 系统安装sumo


## 使用流程 

1. 从openstreetmap上选取一块想要的地图:
    ``` bash
    cd ./TrafficManager/networkFiles/
    $SUMO_HOME/tools/osmWebWizard.py
    ```
    完成后会打开sumo主界面，检查无误后即可关闭主界面。
    此时在运行命令的`networkFiles`目录下应当出现一个以`yyyy-mm-dd-hh-mm-ss`格式的文件夹。
    进入该文件夹，解压里面的`osm.net.xml.gz`得到`osm.net.xml`文件。
    重命名该文件夹为具体地名，如`location`。

2. 生成车流信息
    进入该文件夹，并生成车流信息。
    ```bash
    cd ./TrafficManager/networkFiles/location 
    $SUMO_HOME/tools/randomTrips.py -n osm.net.xml.gz -e 300 -r osm.rou.xml -p 0.5  
    ```

3. 根据xml文件转换为nuscence的地图数据格式，用json保存

   ```bash
   # under DriveArena root dir
   python ./TrafficManager/xml_explain2HD.py  --map_name location --root_dir ./TrafficManager/networkFiles
   ```

4. 修改配置并运行：
    修改`DriveArena/config.yaml`里相关字段：
    ```yaml
    map:
        name: 'location' #'boston-seaport'
    ```
    运行代码：
    ```bash
     # under DriveArena root folder
     python ./TrafficManager/sim_manager.py
    ``` 