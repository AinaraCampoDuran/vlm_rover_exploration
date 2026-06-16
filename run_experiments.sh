#!/bin/bash

# Configuration
MODELS=("Qwen3-VL.yaml" "InternVL3.yaml" "MiniCPM.yaml")
REPETITIONS=1
TIMEOUT="20m"

# Ensure ROS 2 environment is sourced
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    source /opt/ros/jazzy/setup.bash
fi

if [ -f "install/setup.bash" ]; then
    source install/setup.bash
else
    echo "Warning: install/setup.bash not found. Make sure you run this script from the ros2_ws root."
fi

for MODEL in "${MODELS[@]}"; do
    echo "================================================="
    echo "Starting evaluations for model: $MODEL"
    echo "================================================="
    for (( i=0; i<$REPETITIONS; i++ )); do
        echo ">>> Running repetition $i / $REPETITIONS for $MODEL"
        
        # Run launch file with a timeout
        timeout -k 10s $TIMEOUT ros2 launch vlm_rover_exploration_bringup vlm_rover_exploration.launch.py vlm_model:=$MODEL repetition_index:=$i total_repetitions:=$REPETITIONS
        EXIT_CODE=$?
        
        if [ $EXIT_CODE -eq 124 ]; then
            echo ">>> [TIMEOUT] Execution $i for $MODEL was killed after $TIMEOUT."
        elif [ $EXIT_CODE -ne 0 ]; then
            echo ">>> [ERROR] Execution $i for $MODEL failed with exit code $EXIT_CODE."
        else
            echo ">>> [SUCCESS] Execution $i for $MODEL completed."
        fi
        
        # Clean up Gazebo, ROS 2, and background processes
        echo ">>> Cleaning up processes..."
        bash ./cleanup_ros.sh
        
        # Small delay between runs to make sure ports and DDS resources are freed
        echo ">>> Waiting 5 seconds before the next run..."
        sleep 5
    done
done

echo "================================================="
echo "All experiments finished!"
echo "================================================="
