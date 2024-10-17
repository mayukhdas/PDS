#!/bin/bash

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is not installed. Please install jq to parse JSON."
    exit 1
fi

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "Error: tmux is not installed. Please install tmux to run experiments in parallel."
    exit 1
fi

# Path to the JSON file containing experiment configurations
CONFIG_FILE="experiments_config.json"

# Check if the config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file $CONFIG_FILE not found."
    exit 1
fi

# Create a new tmux session
tmux new-session -d -s experiments_set3

# Read the JSON file and iterate through the experiments
jq -c '.[]' "$CONFIG_FILE" | while read -r experiment; do
    env_name=$(echo "$experiment" | jq -r '.env_name')
    dataset_path=$(echo "$experiment" | jq -r '.dataset_path')
    
    # Create a new tmux window for each experiment
    tmux new-window -t experiments_set3: -n "$env_name"
    
    # Run the Python script with the appropriate arguments
    tmux send-keys -t "$env_name" "python lstmae_train.py --env_name '$env_name' --dataset_path '$dataset_path'" C-m
done

# Attach to the tmux session
tmux attach-session -t experiments_set3