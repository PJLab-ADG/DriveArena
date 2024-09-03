# Custom Map Configuration

## Environment Setup

1.	Set up the Python environment for MapTR and LimSim.
2.	Install the `nuscenes-devkit` Python package and modify the map_api.py file. Replace the existing map_api.py located at `miniconda3/envs/[env_name]/lib/python3.8/site-packages/nuscenes/map_expansion/map_api.py`with the [map_api.py](TrafficManager/utils/map_api.py) provided in this directory.
3.	Install `SUMO` on your system.

## Workflow

1.	Select a map area from OpenStreetMap:
    ``` bash
    cd ./TrafficManager/networkFiles/
    $SUMO_HOME/tools/osmWebWizard.py
    ```
    Once completed, the SUMO main interface will open. After confirming that everything is correct, you can close the interface. A folder named with the format **yyyy-mm-dd-hh-mm-ss** should now appear in the **networkFiles/** directory where you ran the command. Navigate to this folder and extract the **osm.net.xml.gz** file to obtain the **osm.net.xml** file. Rename the folder to location name, such as `location`.

**2.	Generate traffic flow information:**
Navigate to the folder and generate the traffic flow data.

```bash
cd ./TrafficManager/networkFiles/location 
$SUMO_HOME/tools/randomTrips.py -n osm.net.xml.gz -e 300 -r osm.rou.xml -p 0.5  
```

**3.    Convert the XML file to the NuScenes map data format and save it as a JSON file:**
   ```bash
   # under DriveArena root dir
   python ./TrafficManager/xml_explain2HD.py  --map_name location --root_dir ./TrafficManager/networkFiles
   ```


**4.	Modify the configuration and run:**
Update the relevant fields in the [config.yaml](../../config.yaml) file:
```yaml
map:
    name: 'location' #'boston-seaport'
```
Run the code:
```bash
    # under DriveArena root folder
    python ./TrafficManager/sim_manager.py
``` 

