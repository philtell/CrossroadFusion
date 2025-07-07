# radar_fusion_with_geo_optimized.py

import socket
import struct
import threading
import math
import time
import math
from collections import defaultdict
from scipy.spatial import KDTree
import numpy as np
import asyncio
import json
import websockets
from collections import defaultdict
from pyproj import Transformer
import numpy as np
from scipy.spatial import cKDTree
from collections import defaultdict
from scipy.spatial import cKDTree
from collections import defaultdict
from scipy.spatial.transform import Rotation as R
from TrackManager import TrackerInterface
from visualizer import TrajectoryVisualizer
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from collections import defaultdict

# =====================================
# 一、全局配置与初始化
# =====================================

# 可视化与跟踪器
visualizer = TrajectoryVisualizer()
tracker = TrackerInterface()

# PyProj 定义：WGS84<->ECEF 转换
# （注意：Transformer 的输入顺序通常是 lon, lat, height）
transformer_wgs84_to_ecef = Transformer.from_crs(
    "epsg:4979",  # WGS84 三维：经度、纬度、高程
    {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
    always_xy=True
)

transformer_ecef_to_wgs84 = Transformer.from_crs(
    {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
    "epsg:4979",
    always_xy=True
)

radar_configs = {
    "radar1": {"ip": "192.168.1.61", "yaw": 180.0, "pitch": 0.0, "roll": -1.0,
               "lat": 31.84217867, "lon": 117.18892217, "alt": 7.5, "source": "radar1"},
    "radar2": {"ip": "192.168.1.62", "yaw": 90.0, "pitch": -0.5, "roll": -2.5,
               "lat": 31.84174900, "lon": 117.18923450, "alt": 6.55, "source": "radar2"},
    "radar3": {"ip": "192.168.1.63", "yaw": 360.0, "pitch": -1.0, "roll": -0.7,
               "lat": 31.84202567, "lon": 117.18953600, "alt": 7.9, "source": "radar3"},
    "radar4": {"ip": "192.168.1.64", "yaw": 270.0, "pitch": 1.5, "roll": 4.0,
               "lat": 31.84244650, "lon": 117.18932617, "alt": 6.3, "source": "radar4"},
}


class TrackHistory3:
    def __init__(self):
        # track_id -> list of positions (max 3)
        self.positions = {}

    def update(self, track_id, x, y):
        """
        更新该track_id的历史位置，返回更新后的list
        """
        pos_list = self.positions.get(track_id, [])
        pos_list.append((x, y))
        if len(pos_list) > 10:
            pos_list.pop(0)
        self.positions[track_id] = pos_list
        return pos_list


history = TrackHistory3()


class TrackCounter:
    def __init__(self):
        # 存储 track_id -> 次数
        self.counter = {}

    def update(self, current_ids):
        """
        current_ids: set或list, 本帧出现的track_id
        """
        current_ids = set(current_ids)

        # Step 1: 对本帧出现的track_id做 +1
        for tid in current_ids:
            if tid in self.counter:
                self.counter[tid] = min(self.counter[tid] + 1, 6)
            else:
                self.counter[tid] = 1

        # Step 2: 对所有已知track_id里本帧没出现的做 -1
        for tid in list(self.counter.keys()):
            if tid not in current_ids:
                if self.counter[tid] == 1:
                    self.counter[tid] = 0
                else:
                    self.counter[tid] = max(self.counter[tid] - 1, 0)

        # 返回当前次数表
        return dict(self.counter)


id_tracker = TrackCounter()

transformer = Transformer.from_crs(
    "epsg:4326",
    f"+proj=tmerc +lat_0={radar_configs['radar2']['lat']} "
    f"+lon_0={radar_configs['radar2']['lon']} +k=1 +x_0=0 +y_0=0 +ellps=WGS84",
    always_xy=True
)

local_transforms = {}
local_inv = {}

for name, cfg in radar_configs.items():
    proj_str = (
        f"+proj=tmerc +lat_0={cfg['lat']} +lon_0={cfg['lon']}"
        " +k=1 +x_0=0 +y_0=0 +ellps=WGS84"
    )
    # 从经纬度到该雷达局部（单位：米）
    local_transforms[name] = Transformer.from_crs(
        "epsg:4326", proj_str, always_xy=True
    )
    # 反向：局部坐标系 → WGS84 经纬度
    local_inv[name] = Transformer.from_crs(
        proj_str, "epsg:4326", always_xy=True
    )

visualizer.set_radar_config(radar_configs)

# 匹配优先规则映射
priority_rules = {
    ("radar1", "radar3"): "radar1",
    ("radar2", "radar4"): "radar2",
    ("radar1", "radar2"): "radar2",
    ("radar2", "radar3"): "radar2",
    ("radar3", "radar4"): "radar4",
}
static_points = [
    (117.18935283, 31.84243650),
    (117.18919867, 31.84243400),
    (117.18900750, 31.84223783),
    (117.18900133, 31.84190817),
    (117.18917383, 31.84175217),
    (117.18932600, 31.84175283),
    (117.18953533, 31.84194550),
    (117.18955583, 31.84227133)
]
center_point1 = (static_points[0][0] + static_points[1][0]) / 2, (static_points[0][1] + static_points[1][1]) / 2
center_point2 = (static_points[2][0] + static_points[3][0]) / 2, (static_points[2][1] + static_points[3][1]) / 2
center_point3 = (static_points[4][0] + static_points[5][0]) / 2, (static_points[4][1] + static_points[5][1]) / 2
center_point4 = (static_points[6][0] + static_points[7][0]) / 2, (static_points[6][1] + static_points[7][1]) / 2


# 并查集
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while x != self.parent[x]:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        fx, fy = self.find(x), self.find(y)
        if fx != fy:
            self.parent[fx] = fy


# 源优先级定义
source_priority = ['radar2', 'radar4', 'radar3']


def merge_obstacles(data_list, max_distance=5.0):
    n = len(data_list)
    uf = UnionFind(n)
    positions = [(item['x'], item['y']) for item in data_list]
    tree = KDTree(positions)

    # 合并不同 source 的、距离小于 max_distance 的障碍物
    for i, item in enumerate(data_list):
        nearby_indices = tree.query_ball_point([item['x'], item['y']], r=max_distance)
        for j in nearby_indices:
            if i != j and data_list[i]['source'] != data_list[j]['source']:
                uf.union(i, j)

    clusters = defaultdict(list)
    for i in range(n):
        root = uf.find(i)
        clusters[root].append(i)

    merged_result = []

    for indices in clusters.values():
        if len(indices) == 1:
            merged_result.append(data_list[indices[0]])
            continue

        # 合并多个障碍物
        merged_sources = set()
        best_item = None

        for source_name in source_priority:
            for idx in indices:
                if data_list[idx]['source'] == source_name:
                    best_item = data_list[idx]
                    break
            if best_item:
                break

        if not best_item:
            best_item = data_list[indices[0]]  # 默认选择第一个

        for idx in indices:
            merged_sources.add(data_list[idx]['source'])

        merged_obj = best_item.copy()
        merged_obj['source'] = list(merged_sources)
        merged_result.append(merged_obj)

    return merged_result


lock = threading.Lock()

# 网络端口
TCP_PORT = 7160  # 各雷达 TCP 数据端口
WS_PORT = 8765  # Websocket 转发端口

# 存储状态
connected_clients = set()  # WebSocket 客户端集合
frame_data = {}  # 各雷达当帧数据
frame_ready = {}  # 各雷达当帧就绪状态
frame_cond = threading.Condition()  # 条件变量，用于多线程同步
track_history = defaultdict(lambda: {"pos": [], "ts": []})  # 用于轨迹历史、航向计算


def parse_targets(data: bytes, radar_name: str) -> list:
    """
    从二进制 data 中解析出若干目标点 (x, y, speed)，并转换为地理坐标。
    data 帧格式假设：第 5 字节是目标数量 count；每个目标占 25 字节，其中：
      - offset+1:2 字节为目标序列号（小端 uint16）
      - offset+5:4 字节为 x(float32)
      - offset+9:4 字节为 y(float32)
      - offset+21:4 字节为 speed(float32)
    只保留 |x|、|y| <= 100 米的点。最终返回列表，每项包含：
      x, y, speed, lat, lon, alt, target_id, track_id, timestamp, source。
    """
    cfg = radar_configs[radar_name]
    results = []
    if len(data) < 6:
        return results

    count = data[5]
    for i in range(count):
        offset = 6 + i * 25
        if offset + 25 > len(data):
            break
        try:
            target_serial = struct.unpack_from("<H", data, offset + 1)[0]
            x = struct.unpack_from("<f", data, offset + 5)[0]
            y = struct.unpack_from("<f", data, offset + 9)[0]
            speed = struct.unpack_from("<f", data, offset + 21)[0]
        except struct.error:
            continue

        # 只处理合理范围内的点
        if abs(x) > 100 or abs(y) > 100:
            continue

        # 把雷达局部 (x,y) 转到地理坐标
        lat, lon = radar_to_gps(x, y, cfg)

        # x_r_out, y_r_out = gps_to_radar(lat, lon, cfg)

        # print(f"lat:{lat},lon:{lon} x:{x},y:{y} x_r_out:{x_r_out} , y_r_out:{y_r_out}")
        # lat, lon, alt = radar_local_to_geodetic(x, y, cfg)

        results.append({
            "x": x, "y": y, "speed": speed,
            "lat": lat, "lon": lon, "alt": cfg["alt"],
            "target_id": data[offset],  # 1 字节 ID
            "track_id": target_serial,  # 序列号当 track_id
            "timestamp": time.time(),
            "source": radar_name
        })
    return results


# =====================================
# 五、雷达数据监听线程：radar_listener
# =====================================
def radar_listener(radar_name: str, cfg: dict):
    """
    与指定雷达通过 TCP 连接，循环接收原始字节流，解析为目标列表，
    并将结果写入全局 frame_data。使用条件变量通知融合线程。
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((cfg["ip"], TCP_PORT))
        while True:
            data = sock.recv(4096)  # 根据实际协议包大小调整缓冲区
            if not data:
                break
            # 通常 data[4] == 0x70 表示一个有效帧
            if len(data) >= 6 and data[4] == 0x70:
                targets = parse_targets(data, radar_name)
                with frame_cond:
                    frame_data[radar_name] = targets
                    frame_ready[radar_name] = True
                    frame_cond.notify_all()
    except Exception as ex:
        print(f"[{radar_name}] 监听出错：{ex}")
    finally:
        sock.close()


# =====================================
# 九、启动 WebSocket 服务端
# =====================================
def start_websocket_server():
    """
    在新线程中启动 asyncio 的 WebSocket 服务，用于广播轨迹数据。
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    ws_srv = websockets.serve(websocket_handler, "0.0.0.0", WS_PORT)
    loop.run_until_complete(ws_srv)
    loop.run_forever()


# =====================================
# 十、主函数：启动多线程和融合循环
# =====================================
# -------------------------------
# 雷达坐标系 → GPS 坐标
# -------------------------------
def radar_to_gps(x_r, y_r, radar_cfg):
    """
    将雷达坐标系下的点 (x_r, y_r)（单位：米）转换为GPS经纬度 (lat, lon)。
    同时考虑安装高度和俯仰角。

    radar_cfg: dict {
        "yaw": 雷达正前方相对于正北的角度（度，顺时针为正）
        "pitch": 俯仰角（度，向下为正）
        "lat": 安装位置纬度
        "lon": 安装位置经度
        "height": 安装高度（m）
    }
    """
    # 1. 修正俯仰角，将x_r投影到水平面
    pitch_rad = math.radians(radar_cfg.get("pitch", 0.0))
    x_horiz = x_r * math.cos(pitch_rad)
    z_r = -x_r * math.sin(pitch_rad)

    # 2. 如果要考虑地面交点，再缩放
    if radar_cfg.get("alt") is not None:
        h = radar_cfg["alt"]
        scale = h / (h + z_r) if (h + z_r) != 0 else 1.0
        # print(h,scale)
        x_horiz *= scale
        y_r *= scale

    # 3. 旋转yaw
    yaw_rad = math.radians(radar_cfg["yaw"] - 180)

    delta_x = 0
    delta_y = 0
    #
    if radar_cfg["source"] == "radar1":
        delta_x = 6
        delta_y = -9.5
    elif radar_cfg["source"] == "radar2":
        delta_x = 2
        delta_y = -12
    elif radar_cfg["source"] == "radar3":
        delta_x = 0
        delta_y = -12
    elif radar_cfg["source"] == "radar4":
        delta_x = -1
        delta_y = -11

    x_horiz = x_horiz + delta_x
    y_r = y_r + delta_y
    north = x_horiz * math.cos(yaw_rad) - y_r * math.sin(yaw_rad)
    east = x_horiz * math.sin(yaw_rad) + y_r * math.cos(yaw_rad)

    # 4. 米转经纬度
    lat0 = radar_cfg["lat"]
    lon0 = radar_cfg["lon"]
    lat0_rad = math.radians(lat0)
    delta_lat = north / 111700.0
    delta_lon = east / (111700.0 * math.cos(lat0_rad))

    lat = lat0 + delta_lat
    lon = lon0 + delta_lon

    return lat, lon


def getAngle(lat, lon):
    # 中心点
    center_lon = sum(p[0] for p in static_points) / 8
    center_lat = sum(p[1] for p in static_points) / 8

    d_lon = lon - center_lon
    d_lat = lat - center_lat

    # 判断偏移更大的方向
    if abs(d_lon) > abs(d_lat):
        if d_lon > 0:
            return 180  # 东
        else:
            return 0  # 西
    else:
        if d_lat > 0:
            return 270  # 北
        else:
            return 90  # 南


# -------------------------------
# GPS 经度纬度 → 雷达坐标系
# -------------------------------
def gps_to_radar(lat, lon, radar_cfg):
    """
    将 GPS 经纬度 (lat, lon) 转换为雷达坐标系下的 (x_r, y_r)，单位：米
    - 先算出目标点相对于雷达原点的 north/east（米），
    - 再将 (north, east) 逆时针旋转 -yaw 度，回到雷达坐标系 (x_r, y_r)。
    """
    lat0 = radar_cfg["lat"]
    lon0 = radar_cfg["lon"]
    lat0_rad = math.radians(lat0)

    # 1. 先把经纬度差换算成 north/east（米）
    # north = (lat - lat0) * 111700
    north = (lat - lat0) * 111700.0
    # east  = (lon - lon0) * 111700 * cos(lat0)
    east = (lon - lon0) * 111700.0 * math.cos(lat0_rad)

    # 2. 将 (north, east) 绕垂直轴逆时针旋转 -yaw，得到 (x_r, y_r)
    yaw_rad = math.radians(radar_cfg["yaw"])
    # 旋转矩阵 R = [[ cos(yaw), -sin(yaw)],
    #               [ sin(yaw),  cos(yaw)]]
    # 逆向：R^T = [[ cos(yaw), sin(yaw)],
    #             [-sin(yaw), cos(yaw)]]
    x_r = north * math.cos(yaw_rad) + east * math.sin(yaw_rad)
    y_r = -north * math.sin(yaw_rad) + east * math.cos(yaw_rad)

    return x_r, y_r


def latlon_to_xy(lat1, lon1, lat2, lon2):
    R = 6378137  # WGS-84赤道半径
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    mean_lat = math.radians((lat1 + lat2) / 2)
    x = R * d_lon * math.cos(mean_lat)  # 东向
    y = R * d_lat  # 北向
    return x, y


def rotate_point(x, y, angle_degrees, clockwise=False):
    theta = math.radians(angle_degrees)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    if clockwise:
        x_new = x * cos_theta + y * sin_theta
        y_new = -x * sin_theta + y * cos_theta
    else:
        x_new = x * cos_theta - y * sin_theta
        y_new = x * sin_theta + y * cos_theta

    return x_new, y_new


async def websocket_handler(websocket, path):
    """
    用于接收客户端连接，将 socket 对象加入 connected_clients。
    本示例不处理客户端传入消息，仅用于广播数据。
    """
    connected_clients.add(websocket)
    try:
        async for _ in websocket:
            # 不接收任何下行，只保持连接
            pass
    finally:
        connected_clients.remove(websocket)


async def broadcast_tracks(tracks: list):
    """
    将融合后的轨迹列表序列化后广播给所有 WebSocket 客户端。
    """
    if not connected_clients:
        return
    msg = json.dumps(tracks)
    # 并行发送，如果有异常不影响其他客户端
    await asyncio.gather(
        *[ws.send(msg) for ws in connected_clients],
        return_exceptions=True
    )


def prepare_websocket_output(tracks: list) -> list:
    """
    把 tracker 返回的轨迹（字典列表）转换为前端需要的简化格式。
    """
    ws_list = []
    for t in tracks:
        angle_value = t.get("angle")
        angle_str = f"{angle_value:.2f}" if angle_value is not None else "None"

        ws_list.append({
            "id": t["track_id"],
            "target_id": t["target_id"],
            "type": 2,  # 固定值示例
            "x": f"{t['x']:.2f}",
            "y": f"{t['y']:.2f}",
            "angle": angle_str,
            "speed": f"{t['speed']:.2f}",
            "source": t['source'],
            "lat": f"{t.get('lat', 0):.7f}",
            "lon": f"{t.get('lon', 0):.7f}",
            "timestamp": f"{t['timestamp']:.2f}"
        })
    return ws_list


async def frame_fusion_timer(interval=0.5):
    while True:
        time.sleep(interval)
        with lock:
            all_targets = []
            for targets in frame_data.values():
                all_targets.extend(targets)
        visualizer.set_origin(radar_configs['radar2'])
        tr2 = transformer
        if all_targets:
            print(f"\n=== Frame at {time.strftime('%H:%M:%S')} ===")
            id_list = []
            for t in all_targets:
                ref_lat = radar_configs['radar2']['lat']
                ref_lon = radar_configs['radar2']['lon']
                x, y = latlon_to_xy(ref_lat, ref_lon, t['lat'], t['lon'])
                t['x'] = x
                t['y'] = y
                id_list.append(t['track_id'])
            track_set = id_tracker.update(id_list)
            merged = merge_obstacles(all_targets)
            if len(all_targets) > len(merged):
                fused_num = len(all_targets) - len(merged)
                print(f"fused {fused_num} obj")
            print(f"fusion before size:{len(all_targets)} after size:{(len(merged))}")

            filtered_targets = []
            for t in merged:
                count = track_set.get(t['track_id'], 0)
                if count > 5:
                    pos_list = history.update(t['track_id'], t['x'], t['y'])

                    if len(pos_list) >= 5:
                        # 如果你要3帧，判断是否真的有3帧
                        p_start = pos_list[0]
                        p_end = pos_list[-1]

                        dx = p_end[0] - p_start[0]
                        dy = p_end[1] - p_start[1]

                        if dy == 0 or dx == 0:
                            t['angle'] = getAngle(t['lat'], t['lon'])
                        else:
                            angle_rad = math.atan2(dy, dx)
                            angle_deg = math.degrees(angle_rad)
                            # angle_deg = 90 - angle_deg
                            # angle_deg = (360-angle_deg) % 360
                            t['angle'] = angle_deg
                        # t['angle'] = ( t['angle'] + 90) % 360
                    else:
                        t['angle'] = None
                    filtered_targets.append(t)
            # for t in filtered_targets:
            #     track_id = t['track_id']
            #     angle = t['angle']
            #     print(f"track_id")
            # for t in filtered_targets:
            #     print(t)
            ws_data = prepare_websocket_output(filtered_targets)
            await broadcast_tracks(ws_data)
            visualizer.update_tracks(filtered_targets)
            visualizer.draw()
        else:
            print(f"[{time.strftime('%H:%M:%S')}] No data received")


def main():
    threading.Thread(target=start_websocket_server, daemon=True).start()

    for name, cfg in radar_configs.items():
        threading.Thread(target=radar_listener, args=(name, cfg), daemon=True).start()
    asyncio.run(frame_fusion_timer(0.1))


if __name__ == "__main__":
    main()