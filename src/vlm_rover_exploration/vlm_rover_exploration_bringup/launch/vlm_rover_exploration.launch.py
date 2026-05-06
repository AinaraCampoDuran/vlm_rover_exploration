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
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, TimerAction, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    package_directory = get_package_share_directory("vlm_rover_exploration_bringup")

    vlm_model = LaunchConfiguration("vlm_model")
    vlm_model_arg = DeclareLaunchArgument(
        "vlm_model",
        default_value="Qwen3-VL.yaml",
        description="Name of the VLM model config file in the models directory"
    )

    model_config_file = PathJoinSubstitution([
        package_directory, "models", vlm_model
    ])

    base_model = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("llama_bringup"), "launch", "base.launch.py"
            )
        ),
        launch_arguments={
            "params_file": model_config_file,
            "executable": "llava_node",
            "use_sim_time": "True",
        }.items(),
    )

    rover_moon = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("rover_gazebo"), "launch", "moon.launch.py"
            )
        ),
        launch_arguments={
            "namespace": "", # IMPORTANTE! llama cambia el namespace a "llama" 
            "nav2_controller": "RPP",
            "nav2_planner": "SmacHybrid",
            "use_sim_time": "True",
        }.items(), 
    )

    repetition_index = LaunchConfiguration("repetition_index")
    repetition_index_arg = DeclareLaunchArgument(
        "repetition_index",
        default_value="0",
        description="Current repetition index"
    )

    total_repetitions = LaunchConfiguration("total_repetitions")
    total_repetitions_arg = DeclareLaunchArgument(
        "total_repetitions",
        default_value="1",
        description="Total number of repetitions"
    )

    exploration_sm_cmd = Node(
        package="vlm_rover_exploration",
        executable="exploration_sm",
        output="screen",
        parameters=[{
            "repetition_index": repetition_index,
            "total_repetitions": total_repetitions,
            "use_sim_time": True
        }]
    )

    imu_static_tf_cmd = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        arguments=[
            "0", "0", "0", "0", "0", "0", "base_link", "rover/base_link/imu_sensor"
        ],
        output="screen",
        parameters=[{"use_sim_time": True}]
    )

    delayed_exploration_sm_cmd = TimerAction(
        period=15.0,
        actions=[exploration_sm_cmd]
    )

    yasmin_viewer_cmd = Node(
        package="yasmin_viewer",
        executable="yasmin_viewer_node",
        name="yasmin_viewer_node",
        output="screen",
        parameters=[{"port": 5000, "use_sim_time": True}],
    )

    ld = LaunchDescription()
    ld.add_action(vlm_model_arg)
    ld.add_action(repetition_index_arg)
    ld.add_action(total_repetitions_arg)
    ld.add_action(SetEnvironmentVariable("VLM_MODEL_CONFIG_PATH", model_config_file))
    ld.add_action(base_model)
    ld.add_action(imu_static_tf_cmd)
    ld.add_action(rover_moon)
    ld.add_action(delayed_exploration_sm_cmd)
    ld.add_action(yasmin_viewer_cmd)

    return ld
