#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grasp_service_node.py  (基于 grasp_realtime_node.py 修改)

运行示例：
  python grasp_service_node.py \
      --checkpoint ~/checkpoints/RNGNet_realsense_checkpoint
"""

import argparse, os, time
from typing import Tuple

import cv2, message_filters, numpy as np, torch, rclpy
from cv_bridge import CvBridge
from rclpy.node     import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg        import Image
from visualization_msgs.msg import Marker, MarkerArray
from grasp_msgs.srv         import GetGrasp          # 自定义服务
from geometry_msgs.msg      import Pose

# ---------- 你的原始推理脚本中的核心类 / 函数 ----------
from demo import PointCloudHelper, inference,\
                  AnchorGraspNet, PatchMultiGraspNet
import demo as _d
# ---------------------------------------------------------

bridge = CvBridge()

def depth_to_mm(d: np.ndarray) -> np.ndarray:
    if d.dtype == np.uint16:  return d.astype(np.float32)
    if d.dtype == np.float32: return d * 1000.0
    raise ValueError(f"Unsupported depth dtype: {d.dtype}")


class GraspServiceNode(Node):
    def __init__(self, a: argparse.Namespace):
        super().__init__("grasp_service_node")

        # -------- 两个 callback group：订阅 & 服务 ----------
        self.sub_group = ReentrantCallbackGroup()
        self.srv_group = ReentrantCallbackGroup()

        # -------- 订阅话题，缓存最新帧 ----------
        qos = rclpy.qos.QoSProfile(depth=5)
        self.sub_rgb = message_filters.Subscriber(
            self, Image, a.rgb_topic, qos_profile=qos,
            callback_group=self.sub_group)
        self.sub_dep = message_filters.Subscriber(
            self, Image, a.depth_topic, qos_profile=qos,
            callback_group=self.sub_group)
        sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_dep], queue_size=10, slop=0.05)
        sync.registerCallback(self.image_cb)

        self.latest_rgb   = None  # (H,W,3) float32 0-1
        self.latest_depth = None  # (H,W)   float32 mm

        # -------- Marker 发布器 ----------
        self.mark_pub = self.create_publisher(
            MarkerArray, "grasp_markers", 10)

        # -------- 加载网络 ----------
        self.get_logger().info(f"Loading checkpoint: {a.checkpoint}")
        ckpt = torch.load(a.checkpoint, map_location="cuda")
        self.anchornet = AnchorGraspNet(in_dim=4, ratio=8, anchor_k=6).cuda().eval()
        self.localnet  = PatchMultiGraspNet(
            49, theta_k_cls=6, feat_dim=a.embed_dim, anchor_w=a.anchor_w).cuda().eval()

        def clean(sd):  # 删掉 thop 统计键
            return {k:v for k,v in sd.items()
                    if not (k.endswith("total_ops") or k.endswith("total_params"))}
        self.anchornet.load_state_dict(clean(ckpt["anchor"]), strict=True)
        self.localnet .load_state_dict(clean(ckpt["local"]),  strict=True)

        self.anchors = {"gamma": ckpt["gamma"].cuda(), "beta": ckpt["beta"].cuda()}
        _d.anchornet, _d.localnet, _d.anchors = self.anchornet, self.localnet, self.anchors
        _d.args.center_num = a.center_num 
        # -------- 点云工具 ----------
        self.pc_helper = PointCloudHelper(all_points_num=a.points_num)

        # -------- 服务 ----------
        self.srv = self.create_service(
            GetGrasp, "get_grasps",
            self.handle_get_grasps,
            callback_group=self.srv_group)

        self.args = a
        self.get_logger().info("Node initialised ✅")

    # =====================================================
    #                       订阅
    # =====================================================
    def image_cb(self, rgb_msg: Image, dep_msg: Image):
        self.latest_rgb = cv2.cvtColor(
            bridge.imgmsg_to_cv2(rgb_msg, "bgr8"), cv2.COLOR_BGR2RGB
        ).astype(np.float32) / 255.0
        self.latest_depth = depth_to_mm(
            bridge.imgmsg_to_cv2(dep_msg, "passthrough")).astype(np.float32)

    # =====================================================
    #               GetGrasp 服务回调 =→ 推理
    # =====================================================
    def handle_get_grasps(self, req: GetGrasp.Request,
                          res: GetGrasp.Response) -> GetGrasp.Response:
        if req.input != "start_grasp":
            res.success = False
            return res

        if self.latest_rgb is None:
            self.get_logger().warn("No image buffered yet")
            res.success = False
            return res

        tic = time.time()
        # ---- numpy → torch ----
        rgb_t = torch.from_numpy(self.latest_rgb).permute(2,1,0)[None].cuda()
        dep_t = torch.from_numpy(self.latest_depth).T[None].cuda()

        # ---- 点云 & xyz ----
        pts, *_ = self.pc_helper.to_scene_points(rgb_t, dep_t, include_rgb=True)
        xyz = self.pc_helper.to_xyz_maps(dep_t)
        rgbd = torch.cat([rgb_t.squeeze(), xyz.squeeze()], 0)

        # ---- 2-D 预处理 ----
        rgb_ds = torch.nn.functional.interpolate(rgb_t, (640,360))
        dep_ds = torch.nn.functional.interpolate(dep_t[None], (640,360))[0]
        dep_ds = torch.clip(dep_ds/1000.0 - dep_ds.mean()/1000.0, -1, 1)
        x_in = torch.cat([dep_ds[None], rgb_ds], 1).float()

        # ---- 推理 ----
        gg = inference(
            pts, rgbd, x_in, rgb_t, dep_t,
            use_heatmap=True, vis_heatmap=False, vis_grasp=False)

        if gg is None or len(gg) == 0:
            res.success = False
            return res

        # ---- MarkerArray ----
        ma = MarkerArray()
        for i,g in enumerate(gg.nms()[:50]):
            ma.markers.append(self.make_marker(g, i))
        self.mark_pub.publish(ma)

        res.success = True
        res.output = ma
        fps = 1/(time.time()-tic)
        self.get_logger().info(f"Grasp OK — {len(ma.markers)}  |  FPS≈{fps:.1f}")
        return res

    # =====================================================
    #            Grasp → Marker (同你之前逻辑)
    # =====================================================
    def make_marker(self, g, idx: int) -> Marker:
        pos = np.asarray(g.translation).flatten()
        R = np.asarray(g.rotation).reshape(3,3)
        qw = np.sqrt(1.0 + np.trace(R)) / 2.0
        qx = (R[2,1]-R[1,2]) / (4*qw)
        qy = (R[0,2]-R[2,0]) / (4*qw)
        qz = (R[1,0]-R[0,1]) / (4*qw)

        m = Marker()
        m.header.frame_id = "d435_color_optical_frame"
        m.ns, m.id = "grasps", idx
        m.type, m.action = Marker.ARROW, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = pos.tolist()
        m.pose.orientation.w = qw; m.pose.orientation.x = qx
        m.pose.orientation.y = qy; m.pose.orientation.z = qz
        m.scale.x, m.scale.y, m.scale.z = 0.1, 0.01, 0.02
        m.color.r, m.color.g, m.color.b, m.color.a = 0., 1., 0., g.score
        return m


# ---------------------- main ----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--rgb-topic",   default="/camera/d435/color/image_raw")
    p.add_argument("--depth-topic", default="/camera/d435/aligned_depth_to_color/image_raw")
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--anchor-w",  type=float, default=60.)
    p.add_argument("--points-num",type=int, default=25600)
    p.add_argument("--center-num", type=int, default=48)

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
