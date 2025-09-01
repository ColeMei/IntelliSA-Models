#!/usr/bin/env python3
"""
Complete batch experiment workflow.
Orchestrates training and evaluation of multiple models.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchExperimentWorkflow:
    def __init__(self, config_path: str):
        """Initialize workflow with batch training config."""
        self.config_path = Path(config_path)
        self.project_root = Path(__file__).parent.parent
        
    def run_training_phase(self, dry_run: bool = False, max_experiments: int = None, 
                          filter_pattern: str = None) -> List[str]:
        """Run the training phase."""
        logger.info("🚀 Starting training phase...")
        
        cmd = [
            "python", "scripts/batch_train_models.py",
            "--config", str(self.config_path)
        ]
        
        if dry_run:
            cmd.append("--dry-run")
        if max_experiments:
            cmd.extend(["--max-experiments", str(max_experiments)])
        if filter_pattern:
            cmd.extend(["--filter", filter_pattern])
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        
        if result.returncode != 0:
            logger.error(f"Training phase failed: {result.stderr}")
            return []
        
        # Extract job IDs from output
        job_ids = []
        for line in result.stdout.split('\n'):
            if 'Job ID' in line:
                job_id = line.split()[-1]
                job_ids.append(job_id)
        
        logger.info(f"Training phase completed. Submitted {len(job_ids)} jobs: {job_ids}")
        return job_ids
    
    def wait_for_training_completion(self, job_ids: List[str], check_interval: int = 300) -> bool:
        """Wait for all training jobs to complete."""
        if not job_ids:
            logger.warning("No job IDs provided to monitor")
            return True
        
        logger.info(f"⏳ Monitoring {len(job_ids)} training jobs...")
        
        while True:
            # Check job status
            cmd = ["squeue", "--jobs", ",".join(job_ids), "--format", "%j %t"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Failed to check job status: {result.stderr}")
                return False
            
            # Parse job status
            running_jobs = []
            for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 2:
                        job_name, status = parts[0], parts[1]
                        if status in ['R', 'PD', 'CF']:  # Running, Pending, Configuring
                            running_jobs.append(job_name)
            
            if not running_jobs:
                logger.info("✅ All training jobs completed!")
                return True
            
            logger.info(f"⏳ {len(running_jobs)} jobs still running. Checking again in {check_interval}s...")
            time.sleep(check_interval)
    
    def run_evaluation_phase(self, dry_run: bool = False, max_models: int = None, 
                           filter_pattern: str = None) -> List[str]:
        """Run the evaluation phase."""
        logger.info("🔍 Starting evaluation phase...")
        
        cmd = [
            "python", "scripts/batch_evaluate_models.py"
        ]
        
        if dry_run:
            cmd.append("--dry-run")
        if max_models:
            cmd.extend(["--max-models", str(max_models)])
        if filter_pattern:
            cmd.extend(["--filter", filter_pattern])
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        
        if result.returncode != 0:
            logger.error(f"Evaluation phase failed: {result.stderr}")
            return []
        
        # Extract job IDs from output
        job_ids = []
        for line in result.stdout.split('\n'):
            if 'Job ID' in line:
                job_id = line.split()[-1]
                job_ids.append(job_id)
        
        logger.info(f"Evaluation phase completed. Submitted {len(job_ids)} jobs: {job_ids}")
        return job_ids
    
    def run_complete_workflow(self, phases: List[str] = None, dry_run: bool = False, 
                            max_experiments: int = None, filter_pattern: str = None,
                            wait_for_training: bool = True) -> bool:
        """Run the complete batch experiment workflow."""
        if phases is None:
            phases = ["training", "evaluation"]
        
        logger.info("🎯 Starting batch experiment workflow")
        logger.info(f"Phases: {phases}")
        logger.info(f"Config: {self.config_path}")
        
        training_job_ids = []
        evaluation_job_ids = []
        
        # Training phase
        if "training" in phases:
            training_job_ids = self.run_training_phase(dry_run, max_experiments, filter_pattern)
            if not training_job_ids and not dry_run:
                logger.error("Training phase failed")
                return False
            
            # Wait for training completion if requested
            if wait_for_training and training_job_ids and not dry_run:
                if not self.wait_for_training_completion(training_job_ids):
                    logger.error("Training jobs failed or timed out")
                    return False
        
        # Evaluation phase
        if "evaluation" in phases:
            evaluation_job_ids = self.run_evaluation_phase(dry_run, max_models=max_experiments, filter_pattern=filter_pattern)
            if not evaluation_job_ids and not dry_run:
                logger.error("Evaluation phase failed")
                return False
        
        logger.info("🎉 Batch experiment workflow completed successfully!")
        logger.info(f"Training jobs: {training_job_ids}")
        logger.info(f"Evaluation jobs: {evaluation_job_ids}")
        return True

def main():
    parser = argparse.ArgumentParser(description="Complete batch experiment workflow")
    parser.add_argument("--config", default="configs/encoder/batch_training_config.yaml", 
                       help="Batch training config file")
    parser.add_argument("--phases", nargs="+", 
                       choices=["training", "evaluation"],
                       default=["training", "evaluation"],
                       help="Phases to run")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Dry run mode (show what would be done)")
    parser.add_argument("--max-experiments", type=int, 
                       help="Maximum number of experiments to run")
    parser.add_argument("--filter", type=str, 
                       help="Filter experiments by pattern (e.g., 'codebert')")
    parser.add_argument("--no-wait", action="store_true", 
                       help="Don't wait for training completion")
    
    args = parser.parse_args()
    
    # Initialize workflow
    workflow = BatchExperimentWorkflow(args.config)
    
    # Run workflow
    success = workflow.run_complete_workflow(
        phases=args.phases,
        dry_run=args.dry_run,
        max_experiments=args.max_experiments,
        filter_pattern=args.filter,
        wait_for_training=not args.no_wait
    )
    
    if success:
        logger.info("🎉 Workflow completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Workflow failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
