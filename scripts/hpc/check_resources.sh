#!/bin/bash
# Check available resources and job status

echo "=== SLURM Job Status ==="
squeue -u $(whoami)

echo -e "\n=== GPU Status ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "nvidia-smi not available (not on GPU node)"
fi

echo -e "\n=== Disk Usage ==="
echo "Project directory:"
du -sh $PROJECT_ROOT 2>/dev/null || echo "Project root not accessible"

echo "Temp storage:"
du -sh $TEMP_ROOT 2>/dev/null || echo "Temp root not accessible"

echo -e "\n=== Available Space ==="
df -h /data/gpfs/projects/${PROJECT_ID}

echo -e "\n=== Memory Usage ==="
free -h

echo -e "\n=== Load Average ==="
uptime
