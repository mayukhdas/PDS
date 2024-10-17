#!/bin/bash

# Read the JSON file and convert it to a list of command-line arguments
jq -c '.[]' mujoco_exps.json | while read -r item; do
    env=$(echo "$item" | jq -r '.env')
    iql_model_path=$(echo "$item" | jq -r '.iql_model_path')
    lstm_ae_model_path=$(echo "$item" | jq -r '.lstm_ae_model_path')
    std_dev=$(echo "$item" | jq -c '.std_dev')
    mean=$(echo "$item" | jq -c '.mean')
    beta=$(echo "$item" | jq -r '.beta')
    gamma=$(echo "$item" | jq -r '.gamma')
    
    # Create the command
    cmd="python eval_lstmae.py --env_name '$env' --iql_model_path '$iql_model_path' --ae_model_path '$lstm_ae_model_path' --std_dev '$std_dev' --mean '$mean' --beta $beta --gamma $gamma"
    
    # Print the command to pass it to parallel
    echo "$cmd"
done | parallel
