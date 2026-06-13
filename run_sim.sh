#!/bin/bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash

echo "Iniciando simulador con los parámetros de VLM Exploration..."
echo "Puedes usar low_moon.launch.py pasando el argumento world_script:=low_moon.launch.py"

ros2 launch test_sim.launch.py
