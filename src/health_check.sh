#!/bin/bash

echo "Health Check Script Started"
echo "Press Ctrl+C to stop the script"

while true; do
    echo ""
    timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] Checking health endpoint..."

    response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8083/health/)

    if [ $? -ne 0 ] || [ "$response" != "200" ]; then
        echo "[$timestamp] ERROR: Health check failed with status code $response"
        echo "[$timestamp] Stopping health check script due to error"
        exit 1
    else
        echo "[$timestamp] SUCCESS: Health check passed"
    fi

    sleep 10
done