#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32
import sys
import time

try:
    from lane_follower.msg import LaneData, TurnDetect
except ImportError:
    rospy.logerr("Cannot import LaneData or TurnDetect! Please ensure you have run 'catkin_make' and 'source devel/setup.bash' after creating the custom message.")
    sys.exit(1)


# ---- 第 3 次 visual 轉彎的新流程狀態 ----
# T3_INACTIVE -> (偵測到 3rd sign 且 px >= turn_pixel_threshold_3)
# T3_APPROACH -> (ultrasonic <= t3_ultrasonic_threshold)
# T3_TURN     -> (odom 右轉 90 度完成)
# T3_ALIGN    -> (LaneData.angle 對正 / timeout)
# T3_FORWARD  -> (odom 直走 t3_forward_dist) -> 設 handoff_started 進入紅綠燈流程
T3_INACTIVE = 0
T3_APPROACH = 1
T3_TURN     = 2
T3_ALIGN    = 3
T3_FORWARD  = 4

class FuzzyLogicController:
    """
    Lightweight, dependency-free Fuzzy Controller using Sugeno inference.
    """
    def __init__(self):
        # Define the center points of the fuzzy sets (NL, NM, Z, PM, PL)
        self.offset_centers = [-100.0, -50.0, 0.0, 50.0, 100.0]
        self.angle_centers = [-50.0, -25.0, 0.0, 25.0, 50.0]
        
        # Create a 5x5 Rule Base (Sugeno singletons, range -1.0 to 1.0)
        # Column: Offset (NL, NM, Z, PM, PL)
        # Row: Angle (NL, NM, Z, PM, PL)
        # Logic: offset > 0 or angle > 0 means the car is drifting right, requires left turn (positive output)
        self.rule_matrix = [
            [-1.0, -1.0, -0.8, -0.4,  0.0],
            [-1.0, -0.6, -0.4,  0.0,  0.4],
            [-0.8, -0.4,  0.0,  0.4,  0.8],
            [-0.4,  0.0,  0.4,  0.6,  1.0],
            [ 0.0,  0.4,  0.8,  1.0,  1.0]
        ]
        
    def fuzzify(self, val, centers):
        """
        Fuzzify the input value, returning a dictionary of adjacent set memberships {index: weight}
        """
        if val <= centers[0]:
            return {0: 1.0}
        if val >= centers[-1]:
            return {len(centers)-1: 1.0}
            
        for i in range(len(centers) - 1):
            if centers[i] <= val <= centers[i+1]:
                # Simple linear interpolation (triangular/trapezoidal membership functions)
                ratio = (val - centers[i]) / float(centers[i+1] - centers[i])
                return {i: 1.0 - ratio, i+1: ratio}
        return {2: 1.0}
        
    def compute(self, offset, angle):
        offset_memberships = self.fuzzify(offset, self.offset_centers)
        angle_memberships = self.fuzzify(angle, self.angle_centers)
        
        num = 0.0
        den = 0.0
        
        # Rule Evaluation - using Product Inference
        for o_idx, o_weight in offset_memberships.items():
            for a_idx, a_weight in angle_memberships.items():
                rule_weight = o_weight * a_weight
                out_val = self.rule_matrix[a_idx][o_idx]
                
                num += rule_weight * out_val
                den += rule_weight
                
        if den == 0:
            return 0.0
        return num / den

class LaneControllerFuzzy:
    def __init__(self):
        rospy.init_node('lane_controller_fuzzy', anonymous=True)
        
        # Read parameters
        self.base_speed = rospy.get_param('~base_speed', 0.5)
        self.max_angular = rospy.get_param('~max_angular', 1.0) # Max angular velocity is +/- 1.0 rad/s
        
        # Params for Turn 1
        self.turn_pixel_threshold_1 = rospy.get_param('~turn_pixel_threshold_1', 1000.0)
        self.hard_turn_angular_1 = rospy.get_param('~hard_turn_angular_1', 2.0)
        self.hard_turn_duration_1 = rospy.get_param('~hard_turn_duration_1', 1.0)
        
        # Params for Turn 2
        self.turn_pixel_threshold_2 = rospy.get_param('~turn_pixel_threshold_2', 1000.0)
        self.hard_turn_angular_2 = rospy.get_param('~hard_turn_angular_2', 2.0)
        self.hard_turn_duration_2 = rospy.get_param('~hard_turn_duration_2', 1.0)
        
        # Cooldown after a hard turn to ignore signs and resume lane following
        self.hard_turn_cooldown = rospy.get_param('~hard_turn_cooldown', 2.0)

        # Params for the scheduled turn triggered X seconds after the first hard turn
        # 第一次大轉彎完成後，經過 scheduled_turn_delay 秒，再執行一次同方向的硬轉
        self.scheduled_turn_delay = rospy.get_param('~scheduled_turn_delay', 3.0)
        self.scheduled_turn_angular = rospy.get_param('~scheduled_turn_angular', 2.0)
        self.scheduled_turn_duration = rospy.get_param('~scheduled_turn_duration', 1.0)

        # Params for the scheduled turn triggered X seconds after the SECOND vision hard turn
        # 第二次大轉彎完成後，經過 scheduled_turn_delay_2 秒，再執行一次同方向的硬轉
        self.scheduled_turn_delay_2 = rospy.get_param('~scheduled_turn_delay_2', 3.0)
        self.scheduled_turn_angular_2 = rospy.get_param('~scheduled_turn_angular_2', 2.0)
        self.scheduled_turn_duration_2 = rospy.get_param('~scheduled_turn_duration_2', 1.0)
        
        # Sign alignment parameters
        self.sign_detect_pixel_threshold = rospy.get_param('~sign_detect_pixel_threshold', 5000.0)
        self.sign_offset_threshold = rospy.get_param('~sign_offset_threshold', 50.0)
        self.sign_align_angular = rospy.get_param('~sign_align_angular', 0.5)
        self.scan_angular_z = rospy.get_param('~scan_angular_z', 0.5)
        
        # Turn state
        self.hard_turn_count = 0  # 紀錄大轉彎次數
        self.hard_turn_end_time = 0.0
        self.ignore_sign_end_time = 0.0
        self.active_hard_turn_dir = None
        self.active_hard_turn_angular = 0.0
        self.last_sign_time = 0.0
        self.approaching_sign = False
        self.aligning_sign = False
        self.align_angular_z = 0.0
        self.is_scanning = False
        self.scan_start_time = 0.0

        # Scheduled-turn state (armed after a vision hard turn)
        # 同一時間只會 arm 一個 pending（第一段在第 1 次硬轉後 arm，第二段在第 2 次硬轉後 arm，
        # 兩段之間中間還會夾一次 vision 硬轉，pending 已先 fire 完不會被覆蓋）
        self.scheduled_turn_pending = False
        self.scheduled_turn_trigger_time = 0.0
        self.scheduled_turn_dir = None
        self.scheduled_turn_active_angular = 0.0
        self.scheduled_turn_active_duration = 0.0

        # ---- Mission handoff (lane -> lidar_avoid) ----
        # T3_FORWARD 完成後直接設 handoff_started=True，停車 handoff_stop_duration 秒做緩衝，
        # 然後進紅綠燈等待狀態，放行條件達到後 publish /mission/phase = "lidar_avoid"。
        # 之後本節點不再發 cmd_vel，由 lidar_odom_nav_node 接管底盤。
        self.handoff_stop_duration = rospy.get_param('~handoff_stop_duration', 1.0)
        self.handoff_started = False
        self.handoff_stop_end_time = 0.0
        self.handed_off = False

        # ---- Traffic-light wait（緩衝停車結束後、交棒前） ----
        # 緩衝停車視窗結束 -> 進入 traffic-light 等待狀態（cmd_vel 維持全 0）：
        #   - 看到 'green'                                -> 立即放行 -> phase=lidar_avoid
        #   - 看到 'red' / 'yellow'                       -> 繼續停車並 reset no-detect 計時
        #   - 連續 'none' 累積 >= no_detect_timeout       -> 放行 -> phase=lidar_avoid
        # 「沒偵測到」= /traffic_light 收到 'none'；收到 red/yellow/green 都算有偵測到。
        self.traffic_light_topic = rospy.get_param('~traffic_light_topic', '/traffic_light')
        self.traffic_light_no_detect_timeout = rospy.get_param('~traffic_light_no_detect_timeout', 3.0)
        self.in_traffic_light_state = False
        self.tl_pass_green = False
        self.tl_last_detect_time = 0.0

        # ---- 第 3 次 visual 轉彎的新流程（取代寫死 hard turn）----
        # 1. 偵測 3rd sign 且 px >= turn_pixel_threshold_3 -> T3_APPROACH
        # 2. T3_APPROACH: 用 t3_approach_speed 慢速靠近，有 sign 時繼續走 offset 對齊，
        #    sign 丟失就直走。等 /ultrasonic <= t3_ultrasonic_threshold -> T3_TURN
        # 3. T3_TURN: 原地右轉 90 度（odom yaw 累積差量判斷）-> T3_ALIGN
        # 4. T3_ALIGN: 用 LaneData.angle 原地對正（容差 t3_align_tol_deg），timeout 直接放行
        # 5. T3_FORWARD: odom 直走 t3_forward_dist 公尺 -> 設 handoff_started 進入緩衝停車 + 紅綠燈
        self.turn_pixel_threshold_3   = rospy.get_param('~turn_pixel_threshold_3', 25000.0)
        self.t3_ultrasonic_threshold  = rospy.get_param('~t3_ultrasonic_threshold', 8.0)   # cm
        self.t3_approach_speed        = rospy.get_param('~t3_approach_speed', 0.1)         # m/s
        self.t3_odom_turn_angular     = rospy.get_param('~t3_odom_turn_angular', 1.0)      # rad/s
        self.t3_odom_turn_tol_deg     = rospy.get_param('~t3_odom_turn_tol_deg', 2.0)      # deg
        self.t3_align_angular         = rospy.get_param('~t3_align_angular', 0.4)          # rad/s
        self.t3_align_tol_deg         = rospy.get_param('~t3_align_tol_deg', 3.0)          # deg
        self.t3_align_timeout         = rospy.get_param('~t3_align_timeout', 2.0)          # s
        self.t3_forward_speed         = rospy.get_param('~t3_forward_speed', 0.15)         # m/s
        self.t3_forward_dist          = rospy.get_param('~t3_forward_dist', 0.1)           # m
        self.t3_state = T3_INACTIVE
        self.t3_turn_accum = 0.0       # 由 odom_callback 累積（進 T3_TURN 時歸零）
        self.t3_align_entry_time = 0.0
        self.t3_forward_start_x = 0.0
        self.t3_forward_start_y = 0.0
        # 緩存最新 LaneData（給 T3_ALIGN 用，因為 T3 期間 lane_callback 早 return）
        self.last_lane_angle = None
        self.last_lane_offset = None
        self.last_lane_data_time = 0.0
        # /odometry 追蹤
        self.have_odom = False
        self.cur_yaw = 0.0
        self.prev_yaw_for_accum = 0.0
        self.cur_x = 0.0
        self.cur_y = 0.0

        # ---- Ultrasonic stop（走線最前 X 秒內偵測停止標示）----
        # 從第一次收到 lane_detect 起算 ultrasonic_watch_duration 秒內，
        # 若 /ultrasonic (Float32, cm) < stop_threshold 視為遇到停止標示，停車（發全 0 cmd）。
        # 等到值回升 >= resume_threshold 視為標示被移走，恢復走線，並從此關閉偵測（即使視窗未過）。
        # 視窗過期且尚未觸發停車 -> 直接關閉偵測。
        self.ultrasonic_topic = rospy.get_param('~ultrasonic_topic', '/ultrasonic')
        self.ultrasonic_watch_duration = rospy.get_param('~ultrasonic_watch_duration', 10.0)
        self.ultrasonic_stop_threshold = rospy.get_param('~ultrasonic_stop_threshold', 20.0)
        self.ultrasonic_resume_threshold = rospy.get_param('~ultrasonic_resume_threshold', 25.0)
        self.lane_start_time = None
        self.ultrasonic_enabled = True
        self.ultrasonic_stopping = False
        self.last_ultrasonic_cm = None

        # Initialize Fuzzy Controller
        self.fuzzy_controller = FuzzyLogicController()

        # Publisher
        self.cmd_pub = rospy.Publisher('arduino_vel', Twist, queue_size=10)
        # latched 任務階段，後啟動的 lidar_odom_nav 也能讀到當前值
        self.phase_pub = rospy.Publisher('/mission/phase', String, queue_size=1, latch=True)
        self.phase_pub.publish(String(data="lane"))

        # Subscriber: Subscribe to the custom message containing offset and angle
        self.lane_sub = rospy.Subscriber('lane_detect', LaneData, self.lane_callback)
        self.turn_sub = rospy.Subscriber('turn_detect', TurnDetect, self.turn_callback)
        self.ultrasonic_sub = rospy.Subscriber(self.ultrasonic_topic, Float32, self.ultrasonic_callback)
        self.traffic_light_sub = rospy.Subscriber(self.traffic_light_topic, String, self.traffic_light_callback)
        self.odom_sub = rospy.Subscriber('/odometry', Odometry, self.odom_callback)

        # T3 流程的 50 Hz timer：T3 啟動後由 timer 全權控制 cmd_vel（lane_callback 同期 early return）
        self.t3_timer = rospy.Timer(rospy.Duration(0.05), self._t3_timer_cb)
        
        # 註冊關閉時的回調函數，讓車子可以安全煞停
        rospy.on_shutdown(self.shutdown_hook)
        
        rospy.loginfo("Fuzzy Lane Controller Started.")
        rospy.loginfo("Max Angular Speed: %.2f rad/s, Base Speed: %.2f m/s", self.max_angular, self.base_speed)

    def shutdown_hook(self):
        rospy.loginfo("Shutting down... Stopping the car.")
        twist = Twist()  # Twist() zero-initializes every field → a full stop command
        # 大量發送停機指令，確保信號送到 Arduino
        # 注意：shutdown 期間 rospy.sleep() 會立即拋出 ROSInterruptException，
        # 整個迴圈會在不到 1 ms 內跑完，rosserial 可能還沒把訊息送出去就關掉串列埠，
        # 導致車子停不下來。改用 time.sleep() 確保 publish 之間有真實間隔，
        # 讓 rosserial 有充分時間把至少一筆停車訊息送到 Arduino。
        for _ in range(20):
            try:
                self.cmd_pub.publish(twist)
            except Exception:
                pass
            time.sleep(0.05)

    def ultrasonic_callback(self, msg):
        self.last_ultrasonic_cm = msg.data

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        if self.have_odom:
            d = yaw - self.prev_yaw_for_accum
            # 收斂到 (-pi, pi] 避免 wrap-around 把 90 度誤判
            d = math.atan2(math.sin(d), math.cos(d))
            self.t3_turn_accum += d
        self.prev_yaw_for_accum = yaw
        self.cur_yaw = yaw
        self.cur_x = msg.pose.pose.position.x
        self.cur_y = msg.pose.pose.position.y
        self.have_odom = True

    def traffic_light_callback(self, msg):
        # 只在 traffic-light 等待狀態內才考慮，避免任何階段外的訊息誤觸發。
        if not self.in_traffic_light_state:
            return
        color = msg.data
        now = rospy.Time.now().to_sec()
        if color in ('red', 'yellow', 'green'):
            self.tl_last_detect_time = now
        if color == 'green':
            self.tl_pass_green = True

    def turn_callback(self, msg):
        now = rospy.Time.now().to_sec()

        # T3 流程一旦啟動，所有 turn_detect 訊號都忽略
        if self.t3_state != T3_INACTIVE:
            return

        # 如果目前正在大轉彎或是處於轉彎後的冷卻期，先忽略新的標誌避免重複觸發或影響循線
        if now < self.ignore_sign_end_time:
            return

        if msg.turn_direction in ['left', 'right']:
            # 如果路標太小，視為還沒真正到達需要考慮路標的距離，直接忽略讓系統維持正常循線
            if msg.pixel_size < self.sign_detect_pixel_threshold:
                return

            self.last_sign_time = now
            self.approaching_sign = True

            # 若找到了路標，關閉反轉找標的狀態
            if self.is_scanning:
                self.is_scanning = False

            # === 第三次 visual：用 turn_pixel_threshold_3 commit T3 新流程 ===
            if self.hard_turn_count >= 2:
                if msg.pixel_size >= self.turn_pixel_threshold_3:
                    self.t3_state = T3_APPROACH
                    # T3 期間鎖死，後續 sign 通通忽略（_t3_timer_cb 也會擋）
                    self.ignore_sign_end_time = float('inf')
                    self.aligning_sign = False
                    self.scheduled_turn_pending = False
                    rospy.loginfo(
                        "[T3] 偵測到第 3 個轉彎標示 (px=%.0f, dir=%s) -> T3_APPROACH，等 ultrasonic <= %.1f cm",
                        msg.pixel_size, msg.turn_direction, self.t3_ultrasonic_threshold)
                    return
                # 未達 T3 commit 門檻 -> 走 sub-threshold offset 對齊（跟 turn 1/2 一樣）
                if abs(msg.offset) >= self.sign_offset_threshold:
                    self.aligning_sign = True
                    if msg.offset > 0:
                        self.align_angular_z = -self.sign_align_angular
                    else:
                        self.align_angular_z = self.sign_align_angular
                else:
                    self.aligning_sign = False
                return

            # === Turn 1 / Turn 2 原本邏輯 ===
            # 決定當前要使用的轉彎參數
            if self.hard_turn_count == 0:
                current_pixel_threshold = self.turn_pixel_threshold_1
                current_hard_turn_angular = self.hard_turn_angular_1
                current_hard_turn_duration = self.hard_turn_duration_1
            else:
                # 第二次以後使用 Turn 2 的參數
                current_pixel_threshold = self.turn_pixel_threshold_2
                current_hard_turn_angular = self.hard_turn_angular_2
                current_hard_turn_duration = self.hard_turn_duration_2

            # 當標誌大於門檻，觸發大轉彎
            if msg.pixel_size >= current_pixel_threshold:
                self.active_hard_turn_dir = msg.turn_direction
                self.hard_turn_end_time = now + current_hard_turn_duration
                self.ignore_sign_end_time = self.hard_turn_end_time + self.hard_turn_cooldown
                self.active_hard_turn_angular = current_hard_turn_angular
                self.approaching_sign = False
                self.aligning_sign = False
                self.hard_turn_count += 1
                rospy.loginfo("Executing hard turn #%d (%s) for %.2fs",
                              self.hard_turn_count, msg.turn_direction, current_hard_turn_duration)

                # 第一次大轉彎後，排程 scheduled_turn_delay 秒後再執行一次同方向硬轉
                if self.hard_turn_count == 1:
                    self.scheduled_turn_pending = True
                    self.scheduled_turn_dir = msg.turn_direction
                    self.scheduled_turn_trigger_time = self.hard_turn_end_time + self.scheduled_turn_delay
                    self.scheduled_turn_active_angular = self.scheduled_turn_angular
                    self.scheduled_turn_active_duration = self.scheduled_turn_duration
                    rospy.loginfo("Scheduled follow-up %s turn in %.2fs after first hard turn ends",
                                  self.scheduled_turn_dir, self.scheduled_turn_delay)
                # 第二次大轉彎後，排程 scheduled_turn_delay_2 秒後再執行一次同方向硬轉
                elif self.hard_turn_count == 2:
                    self.scheduled_turn_pending = True
                    self.scheduled_turn_dir = msg.turn_direction
                    self.scheduled_turn_trigger_time = self.hard_turn_end_time + self.scheduled_turn_delay_2
                    self.scheduled_turn_active_angular = self.scheduled_turn_angular_2
                    self.scheduled_turn_active_duration = self.scheduled_turn_duration_2
                    rospy.loginfo("Scheduled follow-up %s turn in %.2fs after second hard turn ends",
                                  self.scheduled_turn_dir, self.scheduled_turn_delay_2)
                return
                
            # 根據 offset 決定是否需要左右轉校正
            if abs(msg.offset) >= self.sign_offset_threshold:
                self.aligning_sign = True
                # 若標誌在右側 (offset > 0)，車子往右偏 (-sign_align_angular) 進行校正
                if msg.offset > 0:
                    self.align_angular_z = -self.sign_align_angular
                else:
                    self.align_angular_z = self.sign_align_angular
            else:
                self.aligning_sign = False

    def _t3_timer_cb(self, event):
        if self.t3_state == T3_INACTIVE:
            return
        if self.handed_off:
            return

        now = rospy.Time.now().to_sec()
        twist = Twist()

        # ---- T3_APPROACH: 慢速靠近，sign 在就 offset 對齊、sign 丟就直走，等 ultrasonic 達標 ----
        if self.t3_state == T3_APPROACH:
            twist.linear.x = self.t3_approach_speed
            # sign 還在最近 0.3 秒且需要對齊 -> 套用 align_angular_z
            if (now - self.last_sign_time) < 0.3 and self.aligning_sign:
                twist.angular.z = self.align_angular_z
            else:
                twist.angular.z = 0.0

            if (self.last_ultrasonic_cm is not None
                    and self.last_ultrasonic_cm <= self.t3_ultrasonic_threshold):
                rospy.loginfo("[T3] APPROACH 完成 (ultra=%.1f cm <= %.1f) -> T3_TURN",
                              self.last_ultrasonic_cm, self.t3_ultrasonic_threshold)
                self.t3_state = T3_TURN
                self.t3_turn_accum = 0.0
                self.cmd_pub.publish(Twist())  # 一筆停車緩衝
                return
            self.cmd_pub.publish(twist)
            return

        # ---- T3_TURN: 原地右轉 90 度（odom yaw 累積差量判斷）----
        if self.t3_state == T3_TURN:
            twist.linear.x = 0.0
            twist.angular.z = -self.t3_odom_turn_angular   # 右轉 = 負角速度
            target = math.radians(90.0)
            tol = math.radians(self.t3_odom_turn_tol_deg)
            if abs(self.t3_turn_accum) >= (target - tol):
                rospy.loginfo("[T3] TURN 完成 (Δyaw=%.1f deg) -> T3_ALIGN",
                              math.degrees(self.t3_turn_accum))
                self.t3_state = T3_ALIGN
                self.t3_align_entry_time = now
                self.cmd_pub.publish(Twist())
                return
            self.cmd_pub.publish(twist)
            return

        # ---- T3_ALIGN: 用 LaneData.angle 原地對正，timeout 直接放行 ----
        if self.t3_state == T3_ALIGN:
            twist.linear.x = 0.0
            elapsed = now - self.t3_align_entry_time

            if elapsed >= self.t3_align_timeout:
                rospy.logwarn("[T3] ALIGN timeout (%.1fs) -> 放行進入 T3_FORWARD",
                              self.t3_align_timeout)
                self._enter_t3_forward()
                return

            # 有近期 LaneData -> 用 angle 對正
            if (self.last_lane_angle is not None
                    and (now - self.last_lane_data_time) < 0.5):
                angle = self.last_lane_angle
                if abs(angle) <= self.t3_align_tol_deg:
                    rospy.loginfo("[T3] ALIGN 完成 (angle=%.1f deg) -> T3_FORWARD", angle)
                    self._enter_t3_forward()
                    return
                # angle > 0 = 車偏右 -> 左轉（正角速度），fuzzy 控制器同慣例
                twist.angular.z = self.t3_align_angular if angle > 0 else -self.t3_align_angular
                self.cmd_pub.publish(twist)
                return

            # 沒近期 LaneData -> 原地停等 timeout 接管
            self.cmd_pub.publish(Twist())
            return

        # ---- T3_FORWARD: odom 直走 t3_forward_dist 公尺 -> 進入紅綠燈 handoff ----
        if self.t3_state == T3_FORWARD:
            dx = self.cur_x - self.t3_forward_start_x
            dy = self.cur_y - self.t3_forward_start_y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist >= self.t3_forward_dist:
                rospy.loginfo("[T3] FORWARD 完成 (dist=%.3f m) -> handoff (緩衝 -> 紅綠燈)", dist)
                self.t3_state = T3_INACTIVE
                self.handoff_started = True
                self.handoff_stop_end_time = now + self.handoff_stop_duration
                self.cmd_pub.publish(Twist())
                return
            twist.linear.x = self.t3_forward_speed
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            return

    def _enter_t3_forward(self):
        self.t3_state = T3_FORWARD
        self.t3_forward_start_x = self.cur_x
        self.t3_forward_start_y = self.cur_y
        self.cmd_pub.publish(Twist())

    def lane_callback(self, msg):
        now = rospy.Time.now().to_sec()

        # 緩存最新 LaneData（T3_ALIGN 由 timer 直接讀，不依賴 lane_callback 觸發）
        self.last_lane_angle = msg.angle
        self.last_lane_offset = msg.offset
        self.last_lane_data_time = now

        # T3 流程啟動後，由 _t3_timer_cb 全權控制 cmd_vel，lane_callback 不再插手
        if self.t3_state != T3_INACTIVE:
            return

        # ---- Mission handoff 檢查（最高優先級） ----
        # 已交棒：完全靜音，由 lidar_odom_nav 接管 /arduino_vel
        if self.handed_off:
            return

        # ---- Ultrasonic stop（僅在走線最前 X 秒內生效，且僅作用一次） ----
        if self.lane_start_time is None:
            self.lane_start_time = now

        if self.ultrasonic_stopping:
            # 標示已移走（且超過 resume 門檻）-> 解除停車，並關閉偵測（即使視窗未過）
            if (self.last_ultrasonic_cm is not None
                    and self.last_ultrasonic_cm >= self.ultrasonic_resume_threshold):
                self.ultrasonic_stopping = False
                self.ultrasonic_enabled = False
                rospy.loginfo("[ultrasonic] %.1f cm >= resume %.1f -> 解除停車，關閉偵測",
                              self.last_ultrasonic_cm, self.ultrasonic_resume_threshold)
                # 不 return，本筆 lane_detect 繼續往下跑正常走線/硬轉邏輯
            else:
                self.cmd_pub.publish(Twist())
                return
        elif self.ultrasonic_enabled:
            if (now - self.lane_start_time) > self.ultrasonic_watch_duration:
                # 視窗過期且尚未觸發停車 -> 關閉偵測
                self.ultrasonic_enabled = False
            elif (self.last_ultrasonic_cm is not None
                  and self.last_ultrasonic_cm < self.ultrasonic_stop_threshold):
                self.ultrasonic_stopping = True
                rospy.loginfo("[ultrasonic] %.1f cm < stop %.1f -> 停車等待標示移除",
                              self.last_ultrasonic_cm, self.ultrasonic_stop_threshold)
                self.cmd_pub.publish(Twist())
                return

        # 交棒中：停車並等待，期間忽略循線
        # （交棒由 T3_FORWARD 完成後在 _t3_timer_cb 內直接設 handoff_started=True 觸發）
        if self.handoff_started:
            # 階段 A：緩衝停車
            if now < self.handoff_stop_end_time:
                self.cmd_pub.publish(Twist())
                return

            # 階段 B：進入 traffic-light 等待狀態（cmd_vel 維持全 0）
            if not self.in_traffic_light_state:
                self.in_traffic_light_state = True
                self.tl_pass_green = False
                # 起始即視為「剛剛沒偵測到」，no-detect 計時從現在算
                self.tl_last_detect_time = now
                rospy.loginfo("[mission] 進入 traffic-light 等待：green 即放行 / 連續 %.1fs 無偵測亦放行",
                              self.traffic_light_no_detect_timeout)

            # 通過條件 1：本階段內看過 'green'
            if self.tl_pass_green:
                self.phase_pub.publish(String(data="lidar_avoid"))
                self.cmd_pub.publish(Twist())
                self.handed_off = True
                self.in_traffic_light_state = False
                rospy.loginfo("[mission] 看到 green -> /mission/phase = lidar_avoid，lane_controller 靜音")
                return

            # 通過條件 2：超過 timeout 都沒看到 red/yellow/green
            if (now - self.tl_last_detect_time) >= self.traffic_light_no_detect_timeout:
                self.phase_pub.publish(String(data="lidar_avoid"))
                self.cmd_pub.publish(Twist())
                self.handed_off = True
                self.in_traffic_light_state = False
                rospy.loginfo("[mission] 連續 %.1fs 無紅綠燈偵測 -> /mission/phase = lidar_avoid",
                              self.traffic_light_no_detect_timeout)
                return

            # 否則：等待中（可能正看著 red/yellow，或還在累計 no-detect）
            self.cmd_pub.publish(Twist())
            return

        twist = Twist()
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0

        # 第一優先級：目前正在大轉彎
        if now < self.hard_turn_end_time:
            twist.linear.x = self.base_speed
            twist.angular.z = self.active_hard_turn_angular if self.active_hard_turn_dir == 'left' else -self.active_hard_turn_angular
            self.cmd_pub.publish(twist)
            return

        # 第一優先級 (排程)：vision 硬轉後 X 秒，無視循線/路標，再執行一次同方向硬轉
        # (第 1 次硬轉後用 scheduled_turn_*，第 2 次硬轉後用 scheduled_turn_*_2，
        #  實際數值在 turn_callback arm 時就已固定在 active_* 裡)
        if self.scheduled_turn_pending and now >= self.scheduled_turn_trigger_time:
            self.active_hard_turn_dir = self.scheduled_turn_dir
            self.active_hard_turn_angular = self.scheduled_turn_active_angular
            self.hard_turn_end_time = now + self.scheduled_turn_active_duration
            self.ignore_sign_end_time = self.hard_turn_end_time + self.hard_turn_cooldown
            self.scheduled_turn_pending = False
            # 清掉其他可能干擾的狀態，硬轉完直接回到循線
            self.approaching_sign = False
            self.aligning_sign = False
            self.is_scanning = False
            rospy.loginfo("Executing scheduled follow-up hard turn (%s) for %.2fs",
                          self.active_hard_turn_dir, self.scheduled_turn_active_duration)
            twist.linear.x = self.base_speed
            twist.angular.z = self.active_hard_turn_angular if self.active_hard_turn_dir == 'left' else -self.active_hard_turn_angular
            self.cmd_pub.publish(twist)
            return

        # 若超過 0.5 秒沒看到標誌，解除靠近狀態並進入尋找路標狀態
        if self.approaching_sign and (now - self.last_sign_time > 0.3):
            self.approaching_sign = False
            self.aligning_sign = False
            self.is_scanning = True
            self.scan_start_time = now
                
        # 第二優先級：原本有看到路標但卻丟失，停止向前，左右小幅掃描找尋
        if self.is_scanning:
            twist.linear.x = 0.0
            
            # 使用週期性切換的方式來左右轉找尋 (左轉1秒 -> 右轉2秒 -> 左轉1秒 -> 不斷循環)
            cycle = (now - self.scan_start_time) % 4.0
            if cycle < 1.0:
                twist.angular.z = self.scan_angular_z
            elif cycle < 3.0:
                twist.angular.z = -self.scan_angular_z
            else:
                twist.angular.z = self.scan_angular_z
                
            self.cmd_pub.publish(twist)
            return

        # 第三優先級：看見路標時，根據 offset 進行對齊校正，或小於 threshold 則直走
        if self.approaching_sign:
            twist.linear.x = self.base_speed-0.2
            if self.aligning_sign:
                twist.angular.z = self.align_angular_z
            else:
                twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            return
        
        # 第四優先級：正常的模糊循線控制 (沒有路標時的日常循線)
        offset = msg.offset
        angle = msg.angle
        
        # Get output from fuzzy inference (range -1.0 to 1.0)
        fuzzy_out = self.fuzzy_controller.compute(offset, angle)
        
        # Scale inference result to the maximum control angular velocity
        angular_z = fuzzy_out * self.max_angular
        
        twist.linear.x = self.base_speed
        twist.angular.z = angular_z
        
        # Publish motor control command
        self.cmd_pub.publish(twist)

if __name__ == '__main__':
    try:
        LaneControllerFuzzy()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
