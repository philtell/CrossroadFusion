import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

# ============ KalmanBoxTracker 和关联工具 ============

class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.eye(7)
        self.kf.F[0,4] = 1
        self.kf.F[1,5] = 1
        self.kf.F[2,6] = 1

        self.kf.H = np.zeros((4,7))
        self.kf.H[0,0] = 1
        self.kf.H[1,1] = 1
        self.kf.H[2,2] = 1
        self.kf.H[3,3] = 1

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = self.convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))

    def predict(self):
        if (self.kf.x[6]+self.kf.x[2]) <=0:
            self.kf.x[6] *=0.0
        self.kf.predict()
        self.age +=1
        if self.time_since_update >0:
            self.hit_streak =0
        self.time_since_update +=1
        return self.convert_x_to_bbox(self.kf.x)

    def get_state(self):
        return self.convert_x_to_bbox(self.kf.x)

    @staticmethod
    def convert_bbox_to_z(bbox):
        w = bbox[2]-bbox[0]
        h = bbox[3]-bbox[1]
        x = bbox[0]+w/2.
        y = bbox[1]+h/2.
        s = w*h
        r = w/float(h)
        return np.array([x,y,s,r]).reshape((4,1))

    @staticmethod
    def convert_x_to_bbox(x,score=None):
        w = np.sqrt(x[2]*x[3])
        h = x[2]/w
        x1 = x[0]-w/2.
        y1 = x[1]-h/2.
        x2 = x[0]+w/2.
        y2 = x[1]+h/2.
        if score is None:
            return np.array([x1,y1,x2,y2]).reshape((1,4))
        else:
            return np.array([x1,y1,x2,y2,score]).reshape((1,5))

def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w*h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
              + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return o

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers)==0:
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0),dtype=int)

    iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

    for d,det in enumerate(detections):
        for t,trk in enumerate(trackers):
            iou_matrix[d,t] = iou(det, trk)

    matched_indices = linear_sum_assignment(-iou_matrix)
    matched_indices = np.array(list(zip(*matched_indices)))

    unmatched_detections = []
    for d in range(len(detections)):
        if d not in matched_indices[:,0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t in range(len(trackers)):
        if t not in matched_indices[:,1]:
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]]<iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if len(matches)==0:
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class Sort:
    def __init__(self,max_age=3,min_hits=1,iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self,dets):
        if len(dets)==0:
            # 没有观测，直接更新预测
            tracks = self.sort.update(np.empty((0,4)))
            return []

        self.frame_count +=1
        trks = np.zeros((len(self.trackers),5))
        to_del = []
        ret = []

        for t,trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:4] = pos
            trk[4] = 0
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks,self.iou_threshold)

        for m in matched:
            self.trackers[m[1]].update(dets[m[0]])

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i])
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1))
            i -=1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret)>0:
            return np.concatenate(ret)
        return np.empty((0,5))

# ============ 封装接口 ============

class TrackerInterface:
    def __init__(self):
        self.sort = Sort(max_age=3, min_hits=1, iou_threshold=0.3)
        self.hit_streaks = {}  # track_id -> hit count

    def update(self, detections):
        """
        detections: list of dict, e.g.
        [
            {"x": "-1.59", "y": "-31.11"},
            {"x": "2.3", "y": "-20.5"},
            ...
        ]
        """
        dets = []
        for det in detections:
            x = float(det["x"])
            y = float(det["y"])
            w = 2.0
            h = 2.0
            dets.append([x - w/2, y - h/2, x + w/2, y + h/2])

        dets_np = np.array(dets)
        tracks = self.sort.update(dets_np)

        current_frame_outputs = []

        for t in tracks:
            track_id = int(t[4])

            # 更新hit streak
            if track_id not in self.hit_streaks:
                self.hit_streaks[track_id] = 1
            else:
                self.hit_streaks[track_id] += 1

            # 如果hit streak恰好达到5，表示本帧是第5次连续命中
            if self.hit_streaks[track_id] == 5:
                bbox = [float(t[0]), float(t[1]), float(t[2]), float(t[3])]
                current_frame_outputs.append({
                    "track_id": track_id,
                    "bbox": bbox
                })

        # 如果没有任何连续≥5的，则返回空list
        return current_frame_outputs

    def get_tracked(self):
        """
        Returns a list of tracked targets whose streak >=5
        """
        results = []
        for trk in self.sort.trackers:
            track_id = int(trk.id+1)
            if self.hit_streaks.get(track_id,0)>=5:
                bbox = trk.get_state()[0]
                results.append({
                    "track_id": track_id,
                    "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
                })
        return results


