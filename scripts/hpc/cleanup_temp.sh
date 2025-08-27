#!/bin/bash
# Clean up temporary storage to free space

echo "🧹 Cleaning temporary storage..."

if [ -d "${TEMP_ROOT}" ]; then
    echo "Current temp storage usage:"
    du -sh ${TEMP_ROOT}
    
    echo "Cleaning data cache..."
    rm -rf ${TEMP_ROOT}/active_training/data_cache/*
    
    echo "Cleaning temp outputs..."
    rm -rf ${TEMP_ROOT}/active_training/temp_outputs/*
    
    echo "Cleaning temp logs..."
    rm -rf ${TEMP_ROOT}/temp_logs/*
    
    echo "After cleanup:"
    du -sh ${TEMP_ROOT}
    
    echo "✅ Temp storage cleaned"
else
    echo "❌ Temp storage directory not found: ${TEMP_ROOT}"
fi
