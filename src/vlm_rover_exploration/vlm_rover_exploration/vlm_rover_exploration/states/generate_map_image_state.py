#!/usr/bin/env python3

# Copyright (C) 2025 Miguel Ángel González Santamarta
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import cv2
import time
import numpy as np

import rclpy
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from nav_msgs.msg import OccupancyGrid
from tf2_ros import Buffer, TransformListener
from tf2_ros import TransformException
from tf_transformations import euler_from_quaternion

import yasmin
from yasmin import Blackboard
from yasmin_ros import MonitorState
from yasmin_ros.basic_outcomes import SUCCEED
from yasmin_ros.yasmin_node import YasminNode


class GenerateMapImageState(MonitorState):

    def __init__(self) -> None:
        node = YasminNode.get_instance()
        self.counter = 0
        self.initial_position = None  # Will be set on first call
        self.position_history = []
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, node)

        map_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        super().__init__(
            OccupancyGrid,
            "/map",
            [SUCCEED],
            self.monitor_handler,
            qos=map_qos,
            msg_queue=1,
            timeout=60, # Aumentado de 10 a 60 para evitar timeouts al arrancar Gazebo/RTAB-Map
        )

    def execute(self, blackboard: Blackboard) -> str:
        # Evita que la imagen de Llava y el grid map se desincronicen
        blackboard["is_map_monitor_active"] = True
        try:
            return super().execute(blackboard)
        finally:
            blackboard["is_map_monitor_active"] = False

    def get_robot_transform(self) -> TransformException | None:
        try:
            transform = self.tf_buffer.lookup_transform(
                "map", "base_link", rclpy.time.Time()
            )
            return transform
        except TransformException as e:
            yasmin.YASMIN_LOG_WARN(f"Could not get transform: {e}")
            return None

    def monitor_handler(self, blackboard: Blackboard, msg: OccupancyGrid) -> str:
        if "is_map_monitor_active" not in blackboard or not blackboard["is_map_monitor_active"]:
            return ""

        transform = self.get_robot_transform()
        while transform is None:
            yasmin.YASMIN_LOG_WARN("Waiting for robot transform...")
            time.sleep(0.5)
            transform = self.get_robot_transform()

        image_width_m = blackboard["image_width_m"]
        image_height_m = blackboard["image_height_m"]
        scale = blackboard["scale_factor"]

        # Add margin in meters
        margin_m = 2  # You can set to 2 or 3 as needed
        alpha = 0.6  # Set your desired alpha value

        # Increase image size by margin
        total_width_m = image_width_m + 2 * margin_m
        total_height_m = image_height_m + 2 * margin_m

        robot_x = transform.transform.translation.x
        robot_y = transform.transform.translation.y
        q = transform.transform.rotation
        _, _, robot_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        blackboard["robot_position"] = (robot_x, robot_y, robot_yaw)

        # Store the initial robot position on the first call
        if self.initial_position is None:
            self.initial_position = (robot_x, robot_y)
            yasmin.YASMIN_LOG_INFO(
                f"Initial robot position set to: ({robot_x:.2f}, {robot_y:.2f})"
            )
        blackboard["initial_position"] = self.initial_position
        
        # Add to history if moved enough (> 0.1 meters)
        if not self.position_history:
            self.position_history.append((robot_x, robot_y))
        else:
            last_x, last_y = self.position_history[-1]
            if np.hypot(robot_x - last_x, robot_y - last_y) > 0.1:
                self.position_history.append((robot_x, robot_y))
                # Keep only last 5 points
                if len(self.position_history) > 5:
                    self.position_history.pop(0)

        init_x, init_y = self.initial_position

        resolution = msg.info.resolution
        blackboard["map_resolution"] = resolution
        map_width = msg.info.width
        map_height = msg.info.height
        map_origin = msg.info.origin

        data = np.array(msg.data, dtype=np.int8).reshape((map_height, map_width))
        
        # Calculate explored area in m2 (cells that are not unknown)
        explored_cells = np.count_nonzero(data != -1)
        blackboard["explored_area_m2"] = float(explored_cells * (resolution * resolution))

        width_px = int(total_width_m / resolution)
        height_px = int(total_height_m / resolution)

        # Center the crop on the robot's initial position (fixed throughout exploration)
        origin_x = map_origin.position.x
        origin_y = map_origin.position.y

        # Convert initial position from world coords to map pixel coords
        init_px = int((init_x - origin_x) / resolution)
        init_py = int((init_y - origin_y) / resolution)

        x_start = max(init_px - width_px // 2, 0)
        y_start = max(init_py - height_px // 2, 0)
        x_end = min(x_start + width_px, map_width)
        y_end = min(y_start + height_px, map_height)

        cropped = data[y_start:y_end, x_start:x_end]
        cropped_height, cropped_width = cropped.shape

        img = np.zeros((cropped_height, cropped_width), dtype=np.uint8)
        free_color = 255
        unknown_color = 100
        occupied_color = 50

        img[cropped == -1] = unknown_color # Unknown -> Grayish
        img[cropped == 0] = free_color # Free -> White
        img[cropped == 100] = occupied_color # Obstacles -> Yellow

        img = cv2.flip(img, 0)
        scaled_img = cv2.resize(
            img,
            (cropped_width * scale, cropped_height * scale),
            interpolation=cv2.INTER_NEAREST,
        )
        color_img = cv2.cvtColor(scaled_img, cv2.COLOR_GRAY2RGBA)
        
        # Color obstacles (value 50) as Bright Yellow in BGR (0, 255, 255)
        color_img[scaled_img == 50] = [0, 255, 255, 255]

        # Draw grid with labels at each full meter
        pixels_per_meter = (1.0 / resolution) * scale
        center_x = cropped_width * scale // 2
        blackboard["center_x"] = center_x
        center_y = cropped_height * scale // 2
        blackboard["center_y"] = center_y
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Draw exploration area circle (excluding margin)
        exploration_radius_m = min(image_width_m, image_height_m) / 2
        exploration_radius_px = int(exploration_radius_m * pixels_per_meter)
        overlay = color_img.copy()
        cv2.circle(
            overlay,
            (center_x, center_y),
            exploration_radius_px,
            (0, 165, 255, 255),  # Orange with alpha
            thickness=2 * scale,
        )
        cv2.addWeighted(overlay, alpha, color_img, 1 - alpha, 0, color_img)

        # Draw robot at actual (x, y) relative to the initial position (image center)
        robot_rel_x = int(((robot_x - init_x) / resolution) * scale) + (cropped_width * scale // 2)
        robot_rel_y = int((-(robot_y - init_y) / resolution) * scale) + (cropped_height * scale // 2)

        # Draw heading
        arrow_len = 10 * scale
        dx = int(np.cos(robot_yaw) * arrow_len)
        dy = int(np.sin(robot_yaw) * arrow_len)
        cv2.arrowedLine(
            color_img,
            (robot_rel_x, robot_rel_y),
            (robot_rel_x + dx, robot_rel_y - dy),
            (0, 0, 255, 255),
            scale // 2,
            tipLength=0.4,
        )

        # Draw Grid Overlay
        # Define fixed cell size in meters
        cell_size_meters = 2.0  # Safe cell size to balance density and spacing
        blackboard["cell_size_meters"] = cell_size_meters

        pixels_per_cell = int(cell_size_meters / resolution) * scale

        img_h, img_w = color_img.shape[:2]
        
        # Calculate number of rows and columns based on image size and fixed cell size
        grid_cols = max(1, img_w // pixels_per_cell)
        grid_rows = max(1, img_h // pixels_per_cell)
        
        cell_w = img_w // grid_cols
        cell_h = img_h // grid_rows

        grid_mapping = {}
        label_counter = 1
        
        # Draw Quadrants and labels
        color_img_h, color_img_w = color_img.shape[:2]
        
        # Draw central crosshair with true alpha blending (Cyan: B=255, G=255, R=0)
        cross_overlay = color_img.copy()
        cv2.line(cross_overlay, (0, center_y), (color_img_w, center_y), (255, 255, 0, 255), max(1, scale))
        cv2.line(cross_overlay, (center_x, 0), (center_x, color_img_h), (255, 255, 0, 255), max(1, scale))
        cv2.addWeighted(cross_overlay, 0.4, color_img, 0.6, 0, color_img)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.2 * scale
        text_thickness = max(1, int(0.3 * scale))
        
        # dynamic margin
        margin_x = max(10, int(color_img_w * 0.05))
        margin_y = max(10, int(color_img_h * 0.05))
        
        # Offset from circle
        text_offset = 10 * scale // 4
        
        # TOP
        top_text = "TOP"
        tw, th = cv2.getTextSize(top_text, font, text_scale, text_thickness)[0]
        tx, ty = center_x - tw//2, center_y - exploration_radius_px - text_offset
        cv2.putText(color_img, top_text, (tx, ty), font, text_scale, (0, 0, 0, 255), text_thickness + 1)
        cv2.putText(color_img, top_text, (tx, ty), font, text_scale, (0, 255, 0, 255), text_thickness)
        
        # BOTTOM
        bot_text = "BOTTOM"
        tw, th = cv2.getTextSize(bot_text, font, text_scale, text_thickness)[0]
        tx, ty = center_x - tw//2, center_y + exploration_radius_px + text_offset + th
        cv2.putText(color_img, bot_text, (tx, ty), font, text_scale, (0, 0, 0, 255), text_thickness + 1)
        cv2.putText(color_img, bot_text, (tx, ty), font, text_scale, (0, 255, 0, 255), text_thickness)

        # LEFT
        left_text = "LEFT"
        tw, th = cv2.getTextSize(left_text, font, text_scale, text_thickness)[0]
        tx, ty = center_x - exploration_radius_px - text_offset - tw, center_y + th//2
        cv2.putText(color_img, left_text, (tx, ty), font, text_scale, (0, 0, 0, 255), text_thickness + 1)
        cv2.putText(color_img, left_text, (tx, ty), font, text_scale, (0, 255, 0, 255), text_thickness)

        # RIGHT
        right_text = "RIGHT"
        tw, th = cv2.getTextSize(right_text, font, text_scale, text_thickness)[0]
        tx, ty = center_x + exploration_radius_px + text_offset, center_y + th//2
        cv2.putText(color_img, right_text, (tx, ty), font, text_scale, (0, 0, 0, 255), text_thickness + 1)
        cv2.putText(color_img, right_text, (tx, ty), font, text_scale, (0, 255, 0, 255), text_thickness)

        # Draw the robot as a red dot proportional to its real size (approx. 0.5m diameter)
        robot_radius_m = 0.25 
        robot_radius_px = int(robot_radius_m * pixels_per_meter)
        cv2.circle(color_img, (robot_rel_x, robot_rel_y), robot_radius_px, (0, 0, 255, 255), -1)

        # Draw explicit ROBOT text label
        robot_label = "ROBOT"
        r_scale = 0.1 * scale
        r_thick = max(1, int(0.2 * scale))
        tw, th = cv2.getTextSize(robot_label, font, r_scale, r_thick)[0]
        rx = robot_rel_x - (tw // 2)
        ry = robot_rel_y - (6 * scale) # Positioned above the 4*scale radius red dot
        # Text Outline (Black)
        cv2.putText(color_img, robot_label, (rx, ry), font, r_scale, (0, 0, 0, 255), r_thick + 1)
        # Text Fill (Red)
        cv2.putText(color_img, robot_label, (rx, ry), font, r_scale, (0, 0, 255, 255), r_thick)

        # 2. Draw labels ONLY on frontier cells (where unknown meets free space)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Mask out everything outside the exploration radius so edges don't trigger false frontiers
        circle_mask = np.zeros_like(scaled_img)
        cv2.circle(circle_mask, (center_x, center_y), exploration_radius_px, 255, -1)
        
        masked_scaled_img = scaled_img.copy()
        masked_scaled_img[circle_mask == 0] = 150  # Dummy value

        candidate_frontiers = []

        for r in range(grid_rows):
            for c in range(grid_cols):
                
                # Pixel bounds
                x1 = c * cell_w
                y1 = r * cell_h
                x2 = x1 + cell_w
                y2 = y1 + cell_h
                
                if c == grid_cols - 1: x2 = img_w
                if r == grid_rows - 1: y2 = img_h

                # Extract the cell region
                cell_region = masked_scaled_img[y1:y2, x1:x2]
                
                # --- Frontier check ---
                num_unknown = np.count_nonzero(cell_region == unknown_color)
                num_occupied = np.count_nonzero(cell_region == occupied_color)
                num_free = np.count_nonzero(cell_region == free_color)

                # Require both unknown AND free pixels, with at least 8% unknown
                # Also, MUST NOT have occupied pixels (obstacles)
                total_pixels = cell_region.size
                min_free_ratio = 0.08
                if not (num_unknown > 0 and num_free > 0) or num_occupied > 0:
                    continue
                if (num_unknown / total_pixels) < min_free_ratio:
                    continue

                # Find unknown (gray) pixels and free (white) pixels
                gray_y, gray_x = np.where(cell_region == unknown_color)
                free_y, free_x = np.where(cell_region == free_color)
                
                if len(gray_x) > 0 and len(free_x) > 0:
                    # 1. Find the "frontier boundary" (gray pixels adjacent to white)
                    kernel = np.ones((3,3), np.uint8)
                    free_mask = (cell_region == free_color).astype(np.uint8)
                    dilated_free = cv2.dilate(free_mask, kernel)
                    frontier_mask = (cell_region == unknown_color) & (dilated_free > 0)
                    front_y, front_x = np.where(frontier_mask)
                    
                    if len(front_x) > 0:
                        # Centroid of the frontier boundary
                        c_front = np.array([np.mean(front_x), np.mean(front_y)])
                        # Centroid of the white (explored) area within the cell
                        c_free = np.array([np.mean(free_x), np.mean(free_y)])
                        
                        # Direction vector from explored space towards the frontier
                        direction = c_front - c_free
                        norm = np.linalg.norm(direction)
                        if norm > 1e-3:
                            direction /= norm
                        else:
                            direction = np.array([0, 0])
                        
                        # Move the label further away from the boundary into the gray zone (deep unexplored area)
                        offset_px = 12.0 * scale
                        cx = np.clip(x1 + int(c_front[0] + direction[0] * offset_px), 0, img_w - 1)
                        cy = np.clip(y1 + int(c_front[1] + direction[1] * offset_px), 0, img_h - 1)
                        
                        # Safety check: ensure the pushed point is still in the gray (unexplored) zone
                        # If not (e.g. for small islands), reduce offset until it is
                        if scaled_img[cy, cx] != unknown_color:
                            for step in range(int(offset_px), 0, -1):
                                test_cx = np.clip(x1 + int(c_front[0] + direction[0] * step), 0, img_w - 1)
                                test_cy = np.clip(y1 + int(c_front[1] + direction[1] * step), 0, img_h - 1)
                                if scaled_img[test_cy, test_cx] == unknown_color:
                                    cx, cy = test_cx, test_cy
                                    break
                    else:
                        # Fallback to simple centroid of gray pixels
                        cx = x1 + int(np.mean(gray_x))
                        cy = y1 + int(np.mean(gray_y))
                else:
                    # Ultimate fallback: geometric center of the cell
                    cx = x1 + cell_w // 2
                    cy = y1 + cell_h // 2

                # Check if inside exploration circle
                dist_sq = (cx - center_x)**2 + (cy - center_y)**2
                if dist_sq > exploration_radius_px**2:
                    continue

                # Calculate world coordinates
                off_px_x = cx - robot_rel_x
                off_px_y = cy - robot_rel_y 
                world_dx = off_px_x / pixels_per_meter
                world_dy = -(off_px_y / pixels_per_meter)
                w_x = robot_x + world_dx
                w_y = robot_y + world_dy

                # Proximity filter (robot)
                dist_to_robot = np.hypot(world_dx, world_dy)
                #if dist_to_robot < 2.0:
                #    continue

                candidate_frontiers.append({
                    "x": w_x, "y": w_y, 
                    "cx": cx, "cy": cy,
                    "dist_to_robot": dist_to_robot
                })

        # Sort candidates by distance to the robot in descending order
        # so that when filtering redundant close frontiers, we always prioritize keeping
        # the ones that are further away from the robot.
        #candidate_frontiers.sort(key=lambda f: f["dist_to_robot"], reverse=True)

        # Filter redundant frontiers (those too close to each other)
        final_frontiers = []
        min_dist_between_ids = 5.0 # meters
        for cand in candidate_frontiers:
            is_redundant = False
            for final in final_frontiers:
                d = np.hypot(cand["x"] - final["x"], cand["y"] - final["y"])
                if d < min_dist_between_ids:
                    is_redundant = True
                    break
            if not is_redundant:
                final_frontiers.append(cand)

        # Draw final labels and populate grid_mapping
        grid_mapping = {}
        label_counter = 1
        for f in final_frontiers:
            label = str(label_counter)
            label_counter += 1
            
            # Draw label (Increased size and contrast)
            text_scale = 0.3 * scale
            thickness = max(2, int(0.6 * scale))
            text_size = cv2.getTextSize(label, font, text_scale, thickness)[0]
            tx = f["cx"] - text_size[0] // 2
            ty = f["cy"] + text_size[1] // 2
            
            # Draw thick black border for high contrast
            cv2.putText(color_img, label, (tx, ty), font, text_scale, (0, 0, 0, 255), thickness + 4)
            # Draw inner color (Blue)
            cv2.putText(color_img, label, (tx, ty), font, text_scale, (255, 0, 0, 255), thickness)

            grid_mapping[label] = {"x": f["x"], "y": f["y"]}


        blackboard["grid_mapping"] = grid_mapping
        blackboard["max_label"] = label_counter - 1

        # Save the image in the blackboard
        blackboard["map_image"] = color_img
        yasmin.YASMIN_LOG_INFO("Saved image to blackboard as 'map_image'")

        cv2.imwrite("map_image.png", color_img)

        return SUCCEED
