#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraspServiceNode — 统一输出带“物品类别”标注的抓取姿态
───────────────────────────────────────────────
· 依赖自定义服务 grasp_msgs/srv/GetGrasp：
    string   input            # "start_grasp"
    string[] target_classes   # 可为空；不为空时只保留这些类别
    ---
    bool                       success
    visualization_msgs/MarkerArray output

· 功能
  1. 接收 RGB + Depth，调用 AnchorGraspNet & PatchMultiGraspNet 生成 2‑D grasp 候选
  2. 订阅最近一帧 YOLO 检测 (含 interior_mask)；根据 grasp 像素中心落在哪个 mask → 判定类别
  3. 若 req.target_classes 非空，只保留匹配类别的 grasp
  4. 在 Marker.ns 写入 class_name；同时可选发布文字标签
"""

import argparse, os, time
from typing import List

import cv2, message_filters, numpy as np, torch, rclpy
from cv_bridge import CvBridge
from rclpy.node     import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.duration  import Duration

from sensor_msgs.msg        import Image
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg      import PointStamped
from yolo_seg_msgs.msg      import DetectionArray
from grasp_msgs.srv         import GetGrasp          # 自定义服务

from demo import PointCloudHelper, inference, AnchorGraspNet, PatchMultiGraspNet
import demo as _d

bridge = CvBridge()

# ------------------------------------------------------------
# 相机内参（出自你刚才那条 CameraInfo 消息的 K 矩阵）
FX = 906.2640991210938
FY = 906.7696533203125
CX = 651.044921875
CY = 357.40350341796875
# ------------------------------------------------------------


def project_points_xyz_to_uv(points_xyz: np.ndarray) -> np.ndarray:
    """
    points_xyz : (N, 3)  in **camera optical frame**, 单位 (m)
    return      : (N, 2)  像素坐标 (u, v)
    """
    x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
    z = np.clip(z, 1e-6, None)                       # 防 0 除
    u = (x * FX / z) + CX
    v = (y * FY / z) + CY
    return np.stack([u, v], axis=1)


# 例子：把 Grasp 3‑D 中心投到像素平面
def annotate_grasp_with_pixel(grasp):
    """
    grasp.translation 是 (3,)  以 camera_optical_frame 为坐标系
    在 grasp 对象里补充 center_pix 属性 (u, v)
    """
    uv = project_points_xyz_to_uv(np.asarray(grasp.translation)[None, :])[0]
    grasp.center_pix = uv.astype(np.int32)
    return grasp

def depth_to_mm(d: np.ndarray) -> np.ndarray:
    if d.dtype == np.uint16:
        return d.astype(np.float32)
    if d.dtype == np.float32:
        return d * 1000.0
    raise ValueError(f"Unsupported depth dtype: {d.dtype}")

# ======================================================================
#   DetectionBuffer — 仅保存最近一帧 YOLO 检测并做 mask hit 判定
# ======================================================================
class DetectionBuffer:
    def __init__(self, node: Node, topic="/yolo/detections"):
        self.node = node
        self.detections: List[dict] = []
        self.frame_id = ""
        node.create_subscription(DetectionArray, topic, self.cb, 10)

    def cb(self, msg: DetectionArray):
        self.frame_id = msg.header.frame_id
        self.detections.clear()
        for det in msg.detections:
            if not det.interior_mask.data:
                continue
            h, w = det.interior_mask.height, det.interior_mask.width
            mask = np.frombuffer(det.interior_mask.data, np.uint8).reshape(h, w)
            self.detections.append({
                "mask": mask,
                "h": h, "w": w,
                "class": det.class_name,
                "conf": det.confidence,
            })

    # --------------------------------------------------
    def class_of_pixel(self, u: int, v: int) -> str | None:
        """返回像素(u,v)命中的类别；无命中→None"""
        best = None
        for det in self.detections:
            if 0 <= v < det["h"] and 0 <= u < det["w"]:
                if det["mask"][v, u]:
                    if best is None or det["conf"] > best["conf"]:
                        best = det
        return best["class"] if best else None

# ======================================================================
#                         主   节   点
# ======================================================================
class GraspServiceNode(Node):
    def __init__(self, a: argparse.Namespace):
        super().__init__("grasp_service_node")

        # ----- callback groups -----
        self.sub_group = ReentrantCallbackGroup()
        self.srv_group = ReentrantCallbackGroup()

        # ----- 订阅 RGB / Depth -----
        qos = rclpy.qos.QoSProfile(depth=5)
        self.sub_rgb = message_filters.Subscriber(self, Image, a.rgb_topic, qos_profile=qos,
                                                   callback_group=self.sub_group)
        self.sub_dep = message_filters.Subscriber(self, Image, a.depth_topic, qos_profile=qos,
                                                   callback_group=self.sub_group)
        sync = message_filters.ApproximateTimeSynchronizer([self.sub_rgb, self.sub_dep], 10, 0.05)
        sync.registerCallback(self.image_cb)

        self.latest_rgb = None  # H×W×3 float32 0‑1
        self.latest_depth = None

        # ----- YOLO 检测缓冲 -----
        self.det_buf = DetectionBuffer(self, "/yolo/detections")

        # ----- Marker 发布 -----
        self.mark_pub = self.create_publisher(MarkerArray, "grasp_markers", 10)

        # ----- 加载网络 -----
        self._load_network(a)
        self.pc_helper = PointCloudHelper(all_points_num=a.points_num)

        # ----- 服务 -----
        self.srv = self.create_service(GetGrasp, "get_grasps", self.handle_get_grasps,
                                        callback_group=self.srv_group)
        self.get_logger().info("GraspServiceNode ready ✔")

    # --------------------------------------------------
    def _load_network(self, a):
        self.get_logger().info(f"Loading checkpoint: {a.checkpoint}")
        ckpt = torch.load(a.checkpoint, map_location="cuda")
        self.anchornet = AnchorGraspNet(in_dim=4, ratio=8, anchor_k=6).cuda().eval()
        self.localnet  = PatchMultiGraspNet(49, theta_k_cls=6, feat_dim=a.embed_dim,
                                            anchor_w=a.anchor_w).cuda().eval()
        def clean(sd):
            return {k:v for k,v in sd.items() if not (k.endswith("total_ops") or k.endswith("total_params"))}
        self.anchornet.load_state_dict(clean(ckpt["anchor"]))
        self.localnet .load_state_dict(clean(ckpt["local"]))
        _d.anchornet, _d.localnet = self.anchornet, self.localnet
        _d.anchors = {"gamma": ckpt["gamma"].cuda(), "beta": ckpt["beta"].cuda()}
        _d.args.center_num = a.center_num

    # ==================================================
    #                     订   阅
    # ==================================================
    def image_cb(self, rgb_msg: Image, dep_msg: Image):
        self.latest_rgb = cv2.cvtColor(bridge.imgmsg_to_cv2(rgb_msg, "bgr8"), cv2.COLOR_BGR2RGB)
        self.latest_rgb = self.latest_rgb.astype(np.float32) / 255.0
        self.latest_depth = depth_to_mm(bridge.imgmsg_to_cv2(dep_msg, "passthrough")).astype(np.float32)
        self.latest_depth = np.clip(self.latest_depth, 0, 2500)

    # ==================================================
    #                 Service 回调 → 推理
    # ==================================================
    def handle_get_grasps(self, req: GetGrasp.Request, res: GetGrasp.Response):
        if req.input != "start_grasp":
            res.success = False; return res
        if self.latest_rgb is None:
            self.get_logger().warn("No image buffered yet"); res.success=False; return res
        target_set = set(req.target_classes)

        tic = time.time()
        dep_t = torch.from_numpy(self.latest_depth).T[None].cuda()
        dep_4d = dep_t.unsqueeze(1)
        rgb_t  = torch.from_numpy(self.latest_rgb).permute(2,1,0)[None].cuda()

        # 点云 & xyz
        pts, *_ = self.pc_helper.to_scene_points(rgb_t, dep_t, include_rgb=True)
        xyz     = self.pc_helper.to_xyz_maps(dep_t)
        rgbd    = torch.cat([rgb_t.squeeze(), xyz.squeeze()], 0)

        # 缩放输入
        rgb_ds = torch.nn.functional.interpolate(rgb_t, size=(640,360))
        dep_ds = torch.nn.functional.interpolate(dep_4d, size=(640,360))[0]
        dep_ds = torch.clip(dep_ds/1000.0 - dep_ds.mean()/1000.0, -1, 1)
        x_in   = torch.cat([dep_ds[None], rgb_ds], 1).float()

        # 推理 grasps
        grasps = inference(pts, rgbd, x_in, rgb_t, dep_t,
                           use_heatmap=True, vis_heatmap=False, vis_grasp=False)
        if not grasps:
            res.success=False; return res

        selected = []
        for g in grasps.nms():
            # 取像素中心 (g.center_pix 在 demo 内部生成；若没有请替换为 bbox center)
            annotate_grasp_with_pixel(g)          # ← 新增
            cx, cy = map(int, g.center_pix)
            cls = self.det_buf.class_of_pixel(cx, cy)
            g.class_name = cls or "unknown"
            if target_set and g.class_name not in target_set:
                continue
            selected.append(g)

        if not selected:
            self.get_logger().warn("No grasp pose matches target_classes")
            res.success = False; return res

        # 生成 MarkerArray
        ma = MarkerArray()
        for i, g in enumerate(selected[:50]):
            ma.markers.append(self.make_marker(g, i))
        self.mark_pub.publish(ma)

        res.success, res.output = True, ma
        fps = 1/(time.time()-tic)
        self.get_logger().info(f"Grasp OK — {len(selected)} poses | FPS≈{fps:.1f}")
        return res

    # --------------------------------------------------
    def make_marker(self, g, idx: int) -> Marker:
        pos = np.asarray(g.translation).flatten()
        R = np.asarray(g.rotation).reshape(3,3)
        qw = np.sqrt(1.0 + np.trace(R)) / 2.0
        qx = (R[2,1]-R[1,2]) / (4*qw); qy = (R[0,2]-R[2,0]) / (4*qw); qz = (R[1,0]-R[0,1]) / (4*qw)
        m = Marker()
        m.header.frame_id = self.det_buf.frame_id or "d435_color_optical_frame"
        m.ns  = g.class_name              # 类别写进 ns
        m.id  = idx
        m.type, m.action = Marker.ARROW, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = pos.tolist()
        m.pose.orientation.w, m.pose.orientation.x = qw, qx
        m.pose.orientation.y, m.pose.orientation.z = qy, qz
        m.scale.x, m.scale.y, m.scale.z = 0.1, 0.01, 0.02
        m.color.r, m.color.g, m.color.b, m.color.a = 0., 1., 0., g.score
        return m

# ---------------------- main ----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--rgb-topic", default="/camera/d435/color/image_raw")
    p.add_argument("--depth-topic", default="/camera/d435/aligned_depth_to_color/image_raw")
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--anchor-w", type=float, default=60.)
    p.add_argument("--points-num", type=int, default=25600)
    p.add_argument("--center-num", type=int, default=320)
    return p.parse_args()

def main():
    a = parse_args()
    if not os.path.isfile(a.checkpoint):
        print("❌ checkpoint 不存在"); return
    rclpy.init()
    node = GraspServiceNode(a)
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node(); rclpy.shutdown()

if __name__ == "__main__":
    main()
