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
            timeout=10,
        )

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
        transform = self.get_robot_transform()
        while transform is None:
            yasmin.YASMIN_LOG_WARN("Waiting for robot transform...")
            time.sleep(0.5)
            transform = self.get_robot_transform()

        image_width_m = blackboard["image_width_m"]
        image_height_m = blackboard["image_height_m"]
        scale = blackboard["scale_factor"]

        # --- Add margin in meters ---
        margin_m = 1  # You can set to 2 or 3 as needed
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
                # Keep only last 100 points
                if len(self.position_history) > 100:
                    self.position_history.pop(0)

        init_x, init_y = self.initial_position

        resolution = msg.info.resolution
        blackboard["map_resolution"] = resolution
        map_width = msg.info.width
        map_height = msg.info.height
        map_origin = msg.info.origin

        data = np.array(msg.data, dtype=np.int8).reshape((map_height, map_width))

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
        occupied_color = 0

        img[cropped == -1] = unknown_color # Unknown -> White
        img[cropped == 0] = free_color # Free -> Gray
        img[cropped == 100] = occupied_color

        img = cv2.flip(img, 0)
        scaled_img = cv2.resize(
            img,
            (cropped_width * scale, cropped_height * scale),
            interpolation=cv2.INTER_NEAREST,
        )
        color_img = cv2.cvtColor(scaled_img, cv2.COLOR_GRAY2RGBA)

        # Draw grid with labels at each full meter
        pixels_per_meter = (1.0 / resolution) * scale
        center_x = cropped_width * scale // 2
        blackboard["center_x"] = center_x
        center_y = cropped_height * scale // 2
        blackboard["center_y"] = center_y
        font = cv2.FONT_HERSHEY_SIMPLEX

        # --- Draw exploration area circle (excluding margin) ---
        exploration_radius_m = min(image_width_m, image_height_m) / 2
        exploration_radius_px = int(exploration_radius_m * pixels_per_meter)
        overlay = color_img.copy()
        cv2.circle(
            overlay,
            (center_x, center_y),
            exploration_radius_px,
            (0, 165, 255, 255),  # Orange with alpha
            thickness=3 * scale,
        )
        cv2.addWeighted(overlay, alpha, color_img, 1 - alpha, 0, color_img)

        # Draw robot at actual (x, y) relative to the initial position (image center)
        robot_rel_x = int(((robot_x - init_x) / resolution) * scale) + (cropped_width * scale // 2)
        robot_rel_y = int((-(robot_y - init_y) / resolution) * scale) + (cropped_height * scale // 2)

        # Draw position history
        history_to_draw = self.position_history[-3:]
        if len(history_to_draw) > 1:
            pts = []
            for px, py in history_to_draw:
                rel_x = int(((px - init_x) / resolution) * scale) + (cropped_width * scale // 2)
                rel_y = int((-(py - init_y) / resolution) * scale) + (cropped_height * scale // 2)
                pts.append([rel_x, rel_y])
            pts = np.array(pts, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(color_img, [pts], isClosed=False, color=(255, 0, 0, 255), thickness=max(1, 2 * scale))

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

        # --- Draw Grid Overlay ---
        # Define fixed cell size in meters
        cell_size_meters = 2.0  # Reduced for higher precision

        pixels_per_cell = int(cell_size_meters / resolution) * scale

        img_h, img_w = color_img.shape[:2]
        
        # Calculate number of rows and columns based on image size and fixed cell size
        grid_cols = max(1, img_w // pixels_per_cell)
        grid_rows = max(1, img_h // pixels_per_cell)
        
        cell_w = img_w // grid_cols
        cell_h = img_h // grid_rows

        grid_mapping = {}
        label_counter = 1
        
        # 1. Draw grid lines on overlay for translucency
        overlay = color_img.copy()

        for r in range(grid_rows):
            for c in range(grid_cols):
                # Pixel bounds
                x1 = c * cell_w
                y1 = r * cell_h
                x2 = x1 + cell_w
                y2 = y1 + cell_h
                
                if c == grid_cols - 1: x2 = img_w
                if r == grid_rows - 1: y2 = img_h

                # Draw rectangle (grid lines)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0, 255), scale)
        
        # Blend grid lines - INCREASED OPACITY
        cv2.addWeighted(overlay, 0.6, color_img, 0.4, 0, color_img)

        cv2.circle(color_img, (robot_rel_x, robot_rel_y), 4 * scale, (0, 0, 255, 255), -1)

        #cv2.putText(
        #    color_img,
        #    "robot",
        #    (
        #        robot_rel_x - (7 * scale),
        #        robot_rel_y - (5 * scale),
        #    ),  # Position of the text on the image
        #    cv2.FONT_HERSHEY_SIMPLEX,
        #    0.2 * scale,  # Font scale
        #    (0, 0, 0, 255),  # White color for the text
        #    3,  # Thickness
        #)

        # 2. Draw labels ONLY on frontier cells (where unknown meets free space)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for r in range(grid_rows):
            for c in range(grid_cols):
                
                # Pixel bounds (re-calculated or cached, simple enough to recalc)
                x1 = c * cell_w
                y1 = r * cell_h
                x2 = x1 + cell_w
                y2 = y1 + cell_h
                
                if c == grid_cols - 1: x2 = img_w
                if r == grid_rows - 1: y2 = img_h

                # Center in pixels
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Check if inside circle
                dist_sq = (cx - center_x)**2 + (cy - center_y)**2
                if dist_sq > exploration_radius_px**2:
                    continue

                # --- Frontier check ---
                # Extract the cell region from the grayscale scaled image
                cell_region = scaled_img[y1:y2, x1:x2]
                total_pixels = cell_region.size
                num_unknown = np.count_nonzero(cell_region == unknown_color)
                num_free = np.count_nonzero(cell_region == free_color)

                has_unknown = num_unknown > 0
                has_free = num_free > 0

                # Require both unknown AND free pixels, with at least 10% free (gray)
                min_free_ratio = 0.08
                if not (has_unknown and has_free):
                    continue
                if (num_unknown / total_pixels) < min_free_ratio:
                    continue
                if (num_free / total_pixels) < min_free_ratio:
                    continue

                label = str(label_counter)
                label_counter += 1

                # Draw label directly on color_img for maximum visibility
                text_scale = 2.0 # Increased size
                thickness = 3 # Increased thickness
                text_size = cv2.getTextSize(label, font, text_scale, thickness)[0]
                text_x = cx - text_size[0] // 2
                text_y = cy + text_size[1] // 2
                
                # Text with White outline for maximum contrast against White/Gray/Red
                # Outline (White)
                cv2.putText(
                    color_img, label, (text_x, text_y), font, text_scale, (255, 255, 255, 255), thickness + 5
                )
                # Text (Blue)
                cv2.putText(
                    color_img, label, (text_x, text_y), font, text_scale, (255, 0, 0, 255), thickness
                )

                # Calculate world coordinates from image coordinates
                # robot_rel_x, robot_rel_y corresponds to robot_x, robot_y (world)
                
                # offset in pixels from robot
                off_px_x = cx - robot_rel_x
                # y increases down in image. 
                off_px_y = cy - robot_rel_y 
                
                # convert to meters
                world_dx = off_px_x / pixels_per_meter
                world_dy = -(off_px_y / pixels_per_meter)
                
                w_x = robot_x + world_dx
                w_y = robot_y + world_dy

                grid_mapping[label] = {"x": w_x, "y": w_y}


        blackboard["grid_mapping"] = grid_mapping
        blackboard["max_label"] = label_counter - 1

        # Save the image in the blackboard
        blackboard["map_image"] = color_img
        yasmin.YASMIN_LOG_INFO("Saved image to blackboard as 'map_image'")

        cv2.imwrite("map_image.png", color_img)

        return SUCCEED
