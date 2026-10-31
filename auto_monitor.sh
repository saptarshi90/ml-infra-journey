#!/bin/bash

# Master Automation Telemetry Daemon Loop
echo "Starting system telemetry daemon... Press [Ctrl+C] to stop."

while true
do
    # Execute your native Python tracking architecture
    python3 metric_logger.py
    
    # Pause execution for exactly 5 seconds before looping
    sleep 5
done
