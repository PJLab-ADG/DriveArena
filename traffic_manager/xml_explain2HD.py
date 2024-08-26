import xml.etree.ElementTree as ET
import json
import math
import uuid
import matplotlib.pyplot as plt
from typing import Dict, List
import collections
import numpy as np
from LimSim.utils.cubic_spline import Spline2D
import alphashape
from shapely.geometry import LineString

nusc_map = {
    "polygon": [],
    "line": [],
    "node": [],
    "drivable_area": [],
    "ped_crossing": [],
    "walkway":[],
    "stop_line":[],
    "carpark_area":[],
    "road_divider":[],
    "lane_divider":[],
    "traffic_light":[],
    "canvas_edge": [
        12084.51,
        5318.44
    ],
    "version": "1.3",
    "arcline_path_3": [],
    "connectivity": [],
    "lane_connector": [],

    "road_segment": [],
    "road_block": [],
    "lane": []
}


# target_layer = [
#   "ped_crossing", # 人行横道
#   "road_divider", # 道路中间的隔离带，需要找edge_id正好为带有“-”的edge，所以也需要数据结构进行存储
#   "lane_divider", # 车道之间的分割线
#   "lane", # lane
#   'road_segment', # 和drivable_area一样，但是不需要合并
# ]

# 需要在数据结构里面预处理，再写入json
class Edge:
    """ 需要维护所有的drivable edge，以便后期合并edge
        id: edge id
        left_bound: [node_token]
        right_bound: [node_token]
        lane_divider: List[[node_token]]
        road_divider: [node_token]
    """
    def __init__(self, id: str, left_bound: list, right_bound: list, lane_divider: List[list], road_divider: list = []):
        self.id = id
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.lane_divider = lane_divider
        self.road_divider = road_divider

        self.left_line = LineString(left_bound)
        self.right_line = LineString(right_bound)

    def __repr__(self) -> str:
        return self.id

class NormalLine:
    """ 根据lane的中心线计算左右边界，用于绘制polygon
        id: lane id
        lane_width: 如果为drivable，默认为3.2m
        index: lane index，用于排序一个edge中的lane，以便获取lane dividers
        center_x: list[float]
        center_y: list[float]
        left_bound: list[list[float]]， [[x1, y1], [x2, y2], ...]
        right_bound: list[list[float]]
    """
    def __init__(self, lane: ET.Element, lane_width: float = 3.2):
        self.lane_width = abs(float(lane.attrib["width"])) if "width" in lane.attrib else lane_width
        self.id = lane.attrib["id"]
        self.index = int(self.id.split("_")[-1])

        
        # get center line
        shape = lane.attrib["shape"].split(" ")
        self.center_x = []
        self.center_y = []
        for i in range(len(shape)):
            self.center_x.append(float(shape[i].split(",")[0]))
            self.center_y.append(float(shape[i].split(",")[1]))
        

        # interpolate shape points for better represent shape
        self.center_x = np.interp(
                np.linspace(0, len(self.center_x)-1, 50),
                np.arange(0, len(self.center_x)),
                self.center_x
            )
        self.center_y = np.interp(
                np.linspace(0, len(self.center_y)-1, 50),
                np.arange(0, len(self.center_y)),
                self.center_y
            )
        
        self.course_spline = Spline2D(self.center_x, self.center_y)
        self.getPlotElem()

    def getPlotElem(self):
        s = np.linspace(0, self.course_spline.s[-1], num=50)
        self.center_line = [
            self.course_spline.calc_position(si) for si in s
        ]
        self.left_bound = [
            self.course_spline.frenet_to_cartesian1D(si, self.lane_width / 2) for si in s
        ]
        self.right_bound = [
            self.course_spline.frenet_to_cartesian1D(si, -self.lane_width / 2) for si in s
        ]
    
    def get_bound_lane(self, bias: float)->list:
        """已废弃，使用getPlotElem代替

        Args:
            bias (float): 偏移量

        Returns:
            list: [[x1, y1], [x2, y2], ...]
        """
        position_list = []
        # 1. points translation
        for i in range(len(self.center_x)-1):
            theta = math.atan2(self.center_y[i+1]-self.center_y[i], self.center_x[i+1]-self.center_x[i])
            delta_y = round(bias*math.cos(math.pi-theta), 3)
            delta_x = round(bias*math.sin(math.pi-theta), 3)
            if i == 0:
                position_list.append([self.center_x[i]+delta_x, self.center_y[i]+delta_y])
                position_list.append([self.center_x[i+1]+delta_x, self.center_y[i+1]+delta_y])
            else:
                # 对于每一段，计算与上一段的交点，从而形成连贯的边界
                current_lane_p1 = [self.center_x[i]+delta_x, self.center_y[i]+delta_y]
                current_lane_p2 = [self.center_x[i+1]+delta_x, self.center_y[i+1]+delta_y]
                last_lane_p2 = position_list.pop()
                intersection = self.calculate_intersection(position_list[-1], last_lane_p2, current_lane_p1, current_lane_p2)
                position_list.extend([intersection, current_lane_p2])
        
        return position_list

    def calculate_intersection(self, lane1_p1:list, lane1_p2:list, lane2_p1:list, lane2_p2:list):
        """计算两条线段的交点

        Args:
            lane1_p1 (list)
            lane1_p2 (list)
            lane2_p1 (list)
            lane2_p2 (list)

        Returns:
            list: [x, y]
        """
        try:
            t1 = round((lane2_p1[0]-lane1_p1[0])*(lane2_p2[1]-lane2_p1[1])-(lane2_p1[1]-lane1_p1[1])*(lane2_p2[0]-lane2_p1[0]), 3)
            t1 /= round((lane1_p2[0]-lane1_p1[0])*(lane2_p2[1]-lane2_p1[1])-(lane1_p2[1]-lane1_p1[1])*(lane2_p2[0]-lane2_p1[0]), 3)
            x = lane1_p1[0]+t1*(lane1_p2[0]-lane1_p1[0])
            y = lane1_p1[1]+t1*(lane1_p2[1]-lane1_p1[1])
        except:
            x = lane1_p2[0]
            y = lane1_p2[1]
        return [x, y]
            
LEFT = 1
RIGHT = -1
    
class XML2JSON:
    """
        pipeline: 
            get_data() _ get junction_lane and ped_crossing
                      |_ get_edge() -> judge drivable lane and walkway -> get class Edge
                      |_ draw_junction() -> judge drivable junction -> draw_drivable_area() or get walkway, get stop_line
                      |_ draw_edge() -> merge edge and get road_divider -> draw_drivable_area(), draw_line()
                      |_ draw_line2polygon() -> draw stop_line, walkway, crossing
                      |_ save_json() -> update node position, make the origin from (0, 0), then save json file


    """
    def __init__(self, xml_file: str):
        """初始化

        Args:
            xml_file (str): xml文件路径
            rule (_type_, optional): 该地区是靠左还是靠右行驶. Defaults to RIGHT.
        """
        self.edges: Dict[str, Edge] = {}
        self.junctionLanes: Dict[str, bool] = collections.defaultdict(lambda: False)
        self.stop_lines: List[NormalLine] = []
        self.walkways: List[NormalLine] = []
        self.crossings: List[NormalLine] = []

        
        self.root = ET.parse(xml_file).getroot()
        self.min_x = 10e9
        self.min_y = 10e9
        self.max_x = 0
        self.max_y = 0

        if self.root.attrib.get('lefthand') == 'true':
            self.rule = LEFT
        else:
            self.rule = RIGHT

    def add_polygon(self, position_list: List[list])->str:
        """在json中添加polygon

        Args:
            position_list (List[list]): polygon的点集

        Returns:
            str: polygon_token
        """
        node_token_list = []
        for point in position_list:
            # create node
            node_token = str(uuid.uuid4())
            nusc_map["node"].append({
                "token": node_token,
                "x": point[0],
                "y": point[1]
            })
            node_token_list.append(node_token)
    
        # create polygon
        polygon_token = str(uuid.uuid4())
        nusc_map["polygon"].append({
            "token":polygon_token,
            "exterior_node_tokens": node_token_list[::],
            "holes": list()
        })

        position_list.append([self.max_x, self.max_y])
        self.max_x, self.max_y = np.max(np.array(position_list), axis=0)
        position_list.pop()
        position_list.append([self.min_x, self.min_y])
        self.min_x, self.min_y = np.min(np.array(position_list), axis=0)

        return polygon_token
    
    def draw_road_segment_in_junction_area(self, position_list: List[list]):
        """绘制drivable_area，包括intersection和drivable_edge

        Args:
            poisition_list (List[list]): ploygon的点集
        """
        polygon_token = self.add_polygon(position_list)
        
        # create drivable_area
        nusc_map["road_segment"].append({
            "token": str(uuid.uuid4()),
            "polygon_token": polygon_token,
            "is_intersection": True,
            "drivable_area_token": ""
        })
        return
    
    def draw_road_segment_in_edge(self, polygon_token: str):
        """绘制drivable_area，包括intersection和drivable_edge

        Args:
            poisition_list (List[list]): ploygon的点集
        """
        # create drivable_area
        nusc_map["road_segment"].append({
            "token": str(uuid.uuid4()),
            "polygon_token": polygon_token,
            "is_intersection": False,
            "drivable_area_token": ""
        })
        return
    
    def add_lane(self, polygon_token: str):
        nusc_map["lane"].append({
            "token": str(uuid.uuid4()),
            "polygon_token": polygon_token,
            "lane_type": "CAR",
            "from_edge_line_token": None,
            "to_edge_line_token": None,
            "left_lane_divider_segments": [],
            "right_lane_divider_segments": []
        })
        return
    
    def draw_line(self, line_list: List[list], line_type: str):
        """绘制line，包括road_divider和lane_divider

        Args:
            line_list (List[list]): _description_
            line_type (str): [road_divider, lane_divider]
        """
        node_token_list = []
        for point in line_list:
            # create node
            node_token = str(uuid.uuid4())
            nusc_map["node"].append({
                "token": node_token,
                "x": point[0],
                "y": point[1]
            })
            node_token_list.append(node_token)
    
        # create line
        line_token = str(uuid.uuid4())
        nusc_map["line"].append({
            "token": line_token,
            "node_tokens": node_token_list[::]
        })
        
        # create type_line
        if line_type == "road_divider":
            nusc_map["road_divider"].append({
                "token": str(uuid.uuid4()),
                "line_token": line_token,
                "road_segment_token": None
            })
        else:
            nusc_map["lane_divider"].append({
                "token": str(uuid.uuid4()),
                "line_token": line_token,
                "lane_divider_segments": []
            })

        line_list.append([self.max_x, self.max_y])
        self.max_x, self.max_y = np.max(np.array(line_list), axis=0)
        line_list.pop()
        line_list.append([self.min_x, self.min_y])
        self.min_x, self.min_y = np.min(np.array(line_list), axis=0)
    
        return

    def draw_line2polygon(self):
        """ [stop_line, walkway, crossing]
        """
        # for stop_line in self.stop_lines:
        #     bound = stop_line.left_bound[::]
        #     bound.extend(stop_line.right_bound[::-1])
        #     ploygon_token = self.add_polygon(bound)
        #     nusc_map["stop_line"].append({
        #         "token": str(uuid.uuid4()),
        #         "polygon_token": ploygon_token,
        #         "stop_line_type": "STOP_SIGN",
        #         "ped_crossing_tokens": [],
        #         "traffic_light_tokens": [],
        #         "road_block_token": None
        #     })
        
        for walkway in self.walkways:
            bound = walkway.left_bound[::]
            bound.extend(walkway.right_bound[::-1])
            ploygon_token = self.add_polygon(bound)
            walk_token = str(uuid.uuid4())
            nusc_map["walkway"].append({
                "token": walk_token,
                "polygon_token": ploygon_token,
            })
                

        for crossing in self.crossings:
            bound = crossing.left_bound[::]
            bound.extend(crossing.right_bound[::-1])
            ploygon_token = self.add_polygon(bound)
            nusc_map["ped_crossing"].append({
                "token": str(uuid.uuid4()),
                "polygon_token": ploygon_token,
                "road_segment_token": None
            })
        return

    def draw_junction(self, junction: ET.Element):
        """ 判断junction是否drivable，并且调用draw_drivable_area
            同时，根据incLanes判断是否需要画出stop_line

        Args:
            junction (ET.Element): _description_
        """
        # 在draw junction的时候，应该根据incline停止线画出来
        # 如果incline是edge，可以去找edge的左右边界
        drivable = False
        incline_list = junction.attrib["incLanes"].split(" ")
        for incline in incline_list:
            if incline[0] == ":":
                if self.junctionLanes[incline]:
                    drivable = True
            else:
                edge_id = "".join(id for id in incline.split("_")[:-1])
                if edge_id in self.edges:
                    drivable = True
                    # 找到stop_line，构造stop_line的ET
                    left_point = self.edges[edge_id].left_bound[-1]
                    right_point = self.edges[edge_id].right_bound[-1]
                    if left_point[0] == right_point[0] and left_point[1] == right_point[1]:
                        print("edge {} is not drivable".format(edge_id))
                        continue
                    shape = str(left_point[0])+","+str(left_point[1])+" "+str(right_point[0])+","+str(right_point[1])
                    stop_line = ET.Element('lane', attrib={'id': str(uuid.uuid4())+"_0", 'shape': shape, 'width': '0.8'})
                    self.stop_lines.append(NormalLine(stop_line))


        shape = junction.attrib["shape"].split(" ")
        position_list = []
        for i in range(len(shape)):
            position_list.append([float(shape[i].split(",")[0]), float(shape[i].split(",")[1])])
        alpha_shape = alphashape.alphashape(position_list, 0)
        if alpha_shape.geom_type == "Polygon":
            position_list = alpha_shape.exterior.coords[::]
        else:
            return 
        if drivable:
            self.draw_road_segment_in_junction_area(position_list)
        else:
            # 将其绘制为walkway
            ploygon_token = self.add_polygon(position_list)
            nusc_map["walkway"].append({
                "token": str(uuid.uuid4()),
                "polygon_token": ploygon_token,
            })

        return 


    def draw_edge(self):
        """ 从Edge中合并同一个大的Edge，增加road_divider
        根据边界是否重合来判定是否需要合并
        """
        incorporation_id = []
        for edge_id, edge in self.edges.items():
            if edge_id in incorporation_id:
                continue
            for compare_edge_id, compare_edge in self.edges.items():
                if compare_edge_id == edge_id or compare_edge_id in incorporation_id:
                    continue
                if compare_edge.left_line.hausdorff_distance(edge.left_line) < 2.0:
                    incorporation_id.append(compare_edge_id)
                    self.edges[edge_id].road_divider = edge.left_bound[::]
                    self.edges[edge_id].left_bound = compare_edge.right_bound[::-1]
                    self.edges[edge_id].lane_divider.extend(compare_edge.lane_divider[::])
                    break
                elif compare_edge.right_line.hausdorff_distance(edge.right_line) < 2.0:
                    incorporation_id.append(compare_edge_id)
                    self.edges[edge_id].road_divider = edge.right_bound[::]
                    self.edges[edge_id].right_bound = compare_edge.left_bound[::-1]
                    self.edges[edge_id].lane_divider.extend(compare_edge.lane_divider[::])
                    break
                else:
                    continue
        
        # 将合并过的另一半edge删除
        for edge_id in incorporation_id:
            self.edges.pop(edge_id)

        # 画出edge的所有元素
        for edge_id, edge in self.edges.items():
            shape = edge.right_bound[::]
            shape.extend(edge.left_bound[::-1])
            polygon_token = self.add_polygon(shape)
            self.add_lane(polygon_token)
            self.draw_road_segment_in_edge(polygon_token)
            # self.draw_drivable_area(shape)
            for lane_divider in edge.lane_divider:
                self.draw_line(lane_divider, "lane_divider")
            if edge.road_divider:
                self.draw_line(edge.road_divider, "road_divider")
        
        return 
        
    def get_edge(self, edge: ET.Element):
        lane_dict = {}
        for lane in edge.iter("lane"):
            # 这里需要判断是否是drivable
            drivable = False
            if "allow" in lane.attrib:
                for v in ["delivery", "evehicle", "all"]:
                    if v in lane.attrib["allow"]:
                        drivable = True
                        break
            else:
                if "all" not in lane.attrib["disallow"]:
                    drivable = True


            if not drivable:
                if "allow" in lane.attrib and "pedestrian" in lane.attrib["allow"]:
                    self.walkways.append(NormalLine(lane))
                continue

            current_lane = NormalLine(lane)
            lane_dict[current_lane.index] = current_lane
        if(lane_dict):
            key_list = sorted(lane_dict.keys())
            if self.rule == LEFT:
                left_bound = lane_dict[key_list[0]].left_bound
                right_bound = lane_dict[key_list[-1]].right_bound
                lane_divider = []
                for i in range(0, len(key_list)-1):
                    lane_divider.append(lane_dict[key_list[i]].right_bound)
            else:
                left_bound = lane_dict[key_list[-1]].left_bound
                right_bound = lane_dict[key_list[0]].right_bound
                lane_divider = []
                for i in range(0, len(key_list)-1):
                    lane_divider.append(lane_dict[key_list[i]].left_bound)

            if self.rule == LEFT:
                road_divider = right_bound[::]
            else:
                road_divider = left_bound[::]
            current_edge = Edge(edge.attrib["id"], left_bound, right_bound, lane_divider)
            self.edges[edge.attrib["id"]] = current_edge
        return

    def get_data(self):
        for edge in self.root.iter("edge"):
            # 处理junction lane
            if edge.attrib["id"][0] == ":":
                for lane in edge.iter("lane"):
                    if edge.attrib["function"] == "crossing":
                        self.crossings.append(NormalLine(lane))
                    drivable = False
                    if "allow" in lane.attrib:
                        for v in ["delivery", "evehicle", "all"]:
                            if v in lane.attrib["allow"]:
                                drivable = True
                                break
                    else:
                        if "all" not in lane.attrib["disallow"]:
                            drivable = True

                    if not drivable:
                        self.junctionLanes.update({lane.attrib["id"]: False})
                    else:
                        self.junctionLanes.update({lane.attrib["id"]: True})
            
            # 处理edge
            else:
                self.get_edge(edge)
        
        # 处理junction
        for junction in self.root.iter("junction"):
            if junction.attrib["id"][0] != ":" and len(junction.attrib["shape"].split(" ")) > 2 and junction.attrib["incLanes"].strip() != "":
                self.draw_junction(junction)
        
        # 处理edge
        self.draw_edge()
        # 处理line
        self.draw_line2polygon()
        return
    
    def save_json(self, file_path: str):
        # 处理max_x, max_y, min_x, min_y
        self.min_x = (self.min_x-10)//10*10
        self.min_y = (self.min_y-10)//10*10
        nusc_map["canvas_edge"] = [self.max_x, self.max_y]

        # 如果需要将node的坐标转换到(0, 0)
        # nusc_map["canvas_edge"] = [self.max_x-self.min_x, self.max_y-self.min_y]
        # for node in nusc_map["node"]:
        #     node["x"] -= self.min_x
        #     node["y"] -= self.min_y

        with open(file_path, 'w') as f:
            json.dump(nusc_map, f, indent=4)
        return


if __name__ == "__main__":
    map_name = "CarlaTown05" #"boston-thomaspark" #"singapore-onenorth"
    root_dir = "networkFiles/"+map_name+"/"
    networkFile = 'osm.net.xml'
    xml2json = XML2JSON(root_dir+networkFile)
    xml2json.get_data()
    xml2json.save_json(root_dir+map_name+'.json')
