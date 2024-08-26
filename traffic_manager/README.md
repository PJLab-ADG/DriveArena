## 环境配置

1. python配置MapTR和LimSim的环境

2. python安装nuscenes-devkit，并且修改map_api.py文件，用本路径下的map_api.py替换`miniconda3/envs/[env_name]/lib/python3.8/site-packages/nuscenes/map_expansion/map_api.py`

3. 系统安装sumo


## 使用流程 -- HD Map

1. 从openstreetmap上扣一块想要的地图，下载后解压里面的`osm.net.xml.gz`得到`osm.net.xml`文件
    ``` bash
    # 打开该网址可用以下命令
    $SUMO_HOME/tools/osmWebWizard.py
    ```
2. 生成车流信息

    ```bash
    $SUMO_HOME/tools/randomTrips.py -n osm.net.xml.gz -e 300 -r osm.rou.xml -p 0.5  
    ```
3. 根据xml文件转换为nuscence的地图数据格式，用json保存

   ```python
   python xml_explain2HD.py
   ```

4. 生成地图并保存
    ```python
    python gen_map_data.py
    ```

<!-- 4. 直接运行MapTR文件夹 `python MapTR`, 可以将数据保存在pth中

    - ps: 该代码写在 `MapTR/__main__.py` 中，因为需要用到该库的内部函数，脚本放在文件夹外面会有异常
    - 在运行前注意修改文件中的路径信息
    - `xml_explain.py`用到的包和`__main__.py`有冲突，所以最好是用两个环境分别运行 -->


## 使用流程 -- Nuscence Map（旧格式）

1. 从openstreetmap上扣一块想要的地图，下载后解压里面的`osm.net.xml.gz`得到`osm.net.xml`文件

2. 根据xml文件转换为nuscence的地图数据格式，用json保存 `python xml_explain2nus.py`

3. 运行 `nus_pth.py` 文件即可得到pth数据
