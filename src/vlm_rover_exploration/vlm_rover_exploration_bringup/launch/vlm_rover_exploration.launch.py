# Copyright (C) 2025 Miguel Ángel González Santamarta

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    package_directory = get_package_share_directory("vlm_rover_exploration_bringup")

    model_config_file = os.path.join(package_directory, "models", "Qwen3-VL.yaml")

    base_model = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("llama_bringup"), "launch", "base.launch.py"
            )
        ),
        launch_arguments={
            "params_file": model_config_file,
            "executable": "llava_node",
        }.items(),
    )

    rover_moon = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("rover_gazebo"), "launch", "moon.launch.py"
            )
        ),
        launch_arguments={"namespace": ""}.items(), # IMPORTANTE! llama cambia el namespace a "llama" 
    )

    exploration_sm_cmd = Node(
        package="vlm_rover_exploration",
        executable="exploration_sm",
        name="exploration_sm",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    yasmin_viewer_cmd = Node(
        package="yasmin_viewer",
        executable="yasmin_viewer_node",
        name="yasmin_viewer_node",
        output="screen",
        parameters=[{"port": 5000, "use_sim_time": True}],
    )

    ld = LaunchDescription()
    ld.add_action(SetEnvironmentVariable("VLM_MODEL_CONFIG_PATH", model_config_file))
    ld.add_action(base_model)
    ld.add_action(rover_moon)
    ld.add_action(exploration_sm_cmd)
    ld.add_action(yasmin_viewer_cmd)

    return ld
