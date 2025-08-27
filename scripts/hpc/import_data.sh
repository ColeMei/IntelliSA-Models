#!/bin/bash
# Import data from main llm-iac-seceval project

MAIN_PROJECT_PATH="/data/gpfs/projects/punim2518/llm-iac-seceval"

echo "📥 Importing data from main project..."

if [ -d "${MAIN_PROJECT_PATH}" ]; then
    # Import processed datasets
    if [ -d "${MAIN_PROJECT_PATH}/experiments/iac_filter_training/data/formatted_dataset" ]; then
        cp -r ${MAIN_PROJECT_PATH}/experiments/iac_filter_training/data/formatted_dataset/* ${PROJECT_ROOT}/data/processed/
        echo "✅ Imported processed datasets"
    else
        echo "⚠️  Processed datasets not found in main project"
    fi
    
    # Import oracle test data if available
    if [ -d "${MAIN_PROJECT_PATH}/data/oracle" ]; then
        cp -r ${MAIN_PROJECT_PATH}/data/oracle/* ${PROJECT_ROOT}/data/processed/
        echo "✅ Imported oracle test data"
    fi
else
    echo "❌ Main project not found at ${MAIN_PROJECT_PATH}"
    echo "Please ensure the main llm-iac-seceval project is cloned first"
fi
