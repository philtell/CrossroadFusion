import cv2
import numpy as np
import random
import math


class TrajectoryVisualizer:
    def __init__(self, width=1020, height=800, scale=7.0, origin=(31.84174900, 117.18923450)):
        self.width = width
        self.height = height
        self.scale = scale
        self.center = (width // 2, height // 1.2)
        self.track_history = {}  # track_id -> [(x, y)]
        self.track_sources = {}  # track_id -> list of sources
        self.color_map = {}  # track_id -> (B, G, R)
        self.last_seen_frame = {}  # track_id -> last_seen_frame_index
        self.track_angles = {}
        self.frame_count = 0
        self.max_history = 1
        self.max_missing_frames = 5
        self.origin_latlon = origin  # 用于经纬度转局部坐标系的原点
        self.radar_configs = {},
        self.static_points = [
            (117.18935283, 31.84243650),
            (117.18919867, 31.84243400),
            (117.18900750, 31.84223783),
            (117.18900133, 31.84190817),
            (117.18917383, 31.84175217),
            (117.18932600, 31.84175283),
            (117.18953533, 31.84194550),
            (117.18955583, 31.84227133)
        ]

    def set_origin(self, radar_configs):
        origin = radar_configs['lat'], radar_configs['lon']
        self.origin_latlon = origin

    def set_radar_config(self, radar_config):
        self.radar_configs = radar_config

    def latlon_to_xy(self, lat2, lon2):
        lat1, lon1 = self.origin_latlon
        R = 6378137  # WGS-84赤道半径
        d_lat = math.radians(lat2 - lat1)
        d_lon = math.radians(lon2 - lon1)
        mean_lat = math.radians((lat1 + lat2) / 2)
        x = R * d_lon * math.cos(mean_lat)  # 东向
        y = R * d_lat  # 北向
        return x, y

    def world_to_image(self, x, y):
        if x is None or y is None or np.isnan(x) or np.isnan(y):
            return None
        ix = int(self.center[0] + x * self.scale)
        iy = int(self.center[1] - y * self.scale)
        if 0 <= ix < self.width and 0 <= iy < self.height:
            return ix, iy
        return None

    def get_color(self, track_id):
        if track_id not in self.color_map:
            self.color_map[track_id] = tuple(random.randint(0, 255) for _ in range(3))
        return self.color_map[track_id]

    def update_tracks(self, fused_targets):
        """
        fused_targets: list of dicts, each must have:
            'track_id', 'lat', 'lon', and optionally 'source'
        """
        self.frame_count += 1
        current_ids = set()

        for t in fused_targets:
            tid = t['track_id']
            # lat, lon = t['lat'], t['lon']
            # x, y = self.latlon_to_xy(lat, lon)
            x = t['x']
            y = t['y']
            srcs = t.get('source', [])

            current_ids.add(tid)
            self.last_seen_frame[tid] = self.frame_count

            self.track_history.setdefault(tid, []).append((x, y))
            if len(self.track_history[tid]) > self.max_history:
                self.track_history[tid].pop(0)

            self.track_sources[tid] = srcs
            self.track_angles[tid] = t['angle']

        # 移除长期未更新的目标
        to_delete = [tid for tid, last in self.last_seen_frame.items()
                     if self.frame_count - last >= self.max_missing_frames]
        for tid in to_delete:
            self.track_history.pop(tid, None)
            self.track_sources.pop(tid, None)
            self.last_seen_frame.pop(tid, None)
            self.color_map.pop(tid, None)
            self.track_angles.pop(tid, None)
        self.latest_targets = fused_targets

    def valid_pixel(self, pt):
        if not isinstance(pt, tuple) or len(pt) != 2:
            return False
        x, y = pt
        return 0 <= x < self.width and 0 <= y < self.height

    def safe_angle(self, angle):
        """
        将angle安全转成float，若无效则返回None
        """
        try:
            angle_f = float(angle)
            if math.isnan(angle_f):
                return None
            return angle_f
        except (TypeError, ValueError):
            return None

    def draw(self):
        img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

        if not hasattr(self, 'radar_configs'):
            return
        # 1. 绘制静态点（蓝色）
        if hasattr(self, 'static_points'):
            for i, (lon, lat) in enumerate(self.static_points):
                x, y = self.latlon_to_xy(lat, lon)  # 注意参数顺序：纬度, 经度
                pt = self.world_to_image(x, y)
                if pt is None or not self.valid_pixel(pt):
                    continue

                px, py = pt
                # print(px,py)
                # 绘制蓝色圆点
                cv2.circle(img, (px, py), 5, (255, 255, 0), -1)
                # 标注点序号
                cv2.putText(img, str(i + 1), (px + 8, py - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        axis_len = 50  # 雷达坐标系的轴长度（像素）
        for name, cfg in self.radar_configs.items():
            x, y = self.latlon_to_xy(cfg['lat'], cfg['lon'])
            pt = self.world_to_image(x, y)
            if pt is None or not self.valid_pixel(pt):
                continue

            px, py = pt
            cv2.circle(img, (px, py), 4, (0, 0, 0), -1)
            cv2.putText(img, name, (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            yaw_rad = math.radians((90 - cfg['yaw']))

            # X轴（前方向，绿色）
            dx = math.cos(yaw_rad)
            dy = math.sin(yaw_rad)
            x_end = (int(px + axis_len * dx), int(py - axis_len * dy))  # 注意图像y轴向下
            cv2.arrowedLine(img, (px, py), x_end, (0, 255, 0), 2, tipLength=0.3)
            cv2.putText(img, 'X', (x_end[0] + 2, x_end[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 128, 0), 1)

            # Y轴（左方向，红色） —— 与X轴垂直，顺时针旋转90度
            yaw_y = yaw_rad + math.pi / 2
            dx_y = math.cos(yaw_y)
            dy_y = math.sin(yaw_y)
            y_end = (int(px + axis_len * dx_y), int(py - axis_len * dy_y))
            cv2.arrowedLine(img, (px, py), y_end, (0, 0, 255), 2, tipLength=0.3)
            cv2.putText(img, 'Y', (y_end[0] + 2, y_end[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 128), 1)
        # 绘制轨迹
        for tid, path in self.track_history.items():
            color = self.get_color(tid)
            for i in range(1, len(path)):
                pt1 = self.world_to_image(*path[i - 1])
                pt2 = self.world_to_image(*path[i])
                if pt1 and pt2 and self.valid_pixel(pt1) and self.valid_pixel(pt2):
                    cv2.line(img, pt1, pt2, color, 2)

            latest = path[-1]
            pixel = self.world_to_image(*latest)
            if pixel and self.valid_pixel(pixel):
                px, py = pixel
                cv2.circle(img, (px, py), 5, color, -1)

                srcs = self.track_sources.get(tid, [])
                if isinstance(srcs, str):
                    srcs = [srcs]

                angle_raw = self.track_angles.get(tid, None)
                angle = self.safe_angle(angle_raw)
                label = f"{tid} "
                if angle is not None:
                    # print(angle)
                    angle_rad = math.radians(angle)
                    dx = int(20 * math.cos(angle_rad))
                    dy = int(20 * math.sin(angle_rad))
                    end_point = (px + dx, py - dy)
                    cv2.arrowedLine(img, (px, py), end_point, color, 2, tipLength=0.3)
                    # label = f"{tid} angle:{angle:.1f}"

                # label = f"{tid} [{', '.join(srcs)}] ({latest[0]:.1f}, {latest[1]:.1f})"
                label = f"{tid} [{', '.join(srcs)}]"

                cv2.putText(img, label, (px + 5, py - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow("Fused Trajectories", img)
        cv2.waitKey(1)
