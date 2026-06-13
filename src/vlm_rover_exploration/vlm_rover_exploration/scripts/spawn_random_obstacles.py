#!/usr/bin/env python3
import subprocess
import random
import math
import sys

def spawn_wall(name, x, y, yaw, length, width, height, world="default"):
    """Spawns a box (wall) in Gazebo Sim using gz service"""
    sdf = f"""
    <?xml version='1.0' ?>
    <sdf version='1.6'>
      <model name='{name}'>
        <static>true</static>
        <link name='link'>
          <collision name='collision'>
            <geometry><box><size>{length} {width} {height}</size></box></geometry>
          </collision>
          <visual name='visual'>
            <geometry><box><size>{length} {width} {height}</size></box></geometry>
            <material>
              <ambient>0.1 0.1 0.1 1</ambient>
              <diffuse>0.1 0.1 0.1 1</diffuse>
            </material>
          </visual>
        </link>
      </model>
    </sdf>
    """
    # Command for Gazebo Sim using ROS 2 utility (more reliable in Jazzy)
    cmd = [
        "ros2", "run", "ros_gz_sim", "create",
        "-world", world,
        "-string", sdf,
        "-name", name,
        "-x", str(x),
        "-y", str(y),
        "-z", str(height),
        "-Y", str(yaw)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    combined_output = result.stdout + result.stderr
    if result.returncode != 0:
        print(f"Error spawning {name}: {result.stderr}")
    else:
        if "successful" not in combined_output.lower():
            print(f"Spawn failed for {name}: {combined_output}")

def spawn_enclosure(prefix, center_x, center_y, size, world="default"):
    """Spawns 4 walls forming a square box"""
    thickness = 0.2
    height = 2
    
    # North
    spawn_wall(f"{prefix}_N", center_x, center_y + size/2, 0, size, thickness, height, world)
    # South
    spawn_wall(f"{prefix}_S", center_x, center_y - size/2, 0, size, thickness, height, world)
    # East
    spawn_wall(f"{prefix}_E", center_x + size/2, center_y, 1.57, size, thickness, height, world)
    # West
    spawn_wall(f"{prefix}_W", center_x - size/2, center_y, 1.57, size, thickness, height, world)

if __name__ == "__main__":
    # Exploration radius is 10m (from 20m width/height in exploration_sm.py)
    # We spawn within 8m to avoid boundaries
    radius = 8.0
    
    # Random angle and distance
    angle = random.uniform(0, 2 * math.pi)
    dist = random.uniform(3, radius) # At least 3m from origin
    
    rx = dist * math.cos(angle)
    ry = dist * math.sin(angle)
    
    prefix = f"enclosure_{random.randint(100, 999)}"
    spawn_enclosure(prefix, rx, ry, size=4.0)
    
    print(f"Random obstacle '{prefix}' spawned at x={rx:.2f}, y={ry:.2f} (within exploration radius).")
