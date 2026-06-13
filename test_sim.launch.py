import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    
    world_script = LaunchConfiguration("world_script")
    world_script_arg = DeclareLaunchArgument(
        "world_script",
        default_value="moon.launch.py",
        description="Name of the world launch script (e.g., moon.launch.py or low_moon.launch.py)"
    )

    rover_moon = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory("rover_gazebo"), "launch"),
            "/", world_script
        ]),
        launch_arguments={
            "namespace": "",
            "nav2_controller": "RPP",
            "nav2_planner": "SmacHybrid",
            "use_sim_time": "True",
        }.items(), 
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

    ld = LaunchDescription()
    ld.add_action(world_script_arg)
    ld.add_action(rover_moon)
    ld.add_action(imu_static_tf_cmd)

    return ld
