#!/usr/bin/env python3
"""
Fix duplicate model storage for existing experiments.
Converts duplicate model copies in results/ to symlinks pointing to models/.
"""

import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DuplicateFixer:
    def __init__(self, models_dir: str = "models/experiments/encoder",
                 results_dir: str = "results/experiments/encoder"):
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.project_root = Path(__file__).parent.parent

        # Ensure we're working with absolute paths
        if not self.models_dir.is_absolute():
            self.models_dir = self.project_root / self.models_dir
        if not self.results_dir.is_absolute():
            self.results_dir = self.project_root / self.results_dir

    def find_duplicate_models(self) -> List[Tuple[Path, Path, float]]:
        """Find all duplicate model directories that need to be fixed."""
        duplicates = []

        if not self.results_dir.exists():
            logger.warning(f"Results directory {self.results_dir} does not exist")
            return duplicates

        for result_dir in self.results_dir.iterdir():
            if not result_dir.is_dir():
                continue

            # Check if this result has a model subdirectory
            model_subdir = result_dir / "model"
            if not model_subdir.exists():
                continue

            # Check if it's actually a directory (not already a symlink)
            if not model_subdir.is_dir() or model_subdir.is_symlink():
                continue

            # Get the size of the duplicate
            size = self._get_dir_size(model_subdir)

            # Find corresponding model in models directory
            # Extract job ID from result directory name
            result_name = result_dir.name
            if "_job" in result_name:
                job_id = result_name.split("_job")[-1]
                # Look for matching model directory
                matching_model = None
                if self.models_dir.exists():
                    for model_dir in self.models_dir.iterdir():
                        if model_dir.is_dir() and job_id in model_dir.name:
                            matching_model = model_dir
                            break

                if matching_model:
                    duplicates.append((result_dir, matching_model, size))
                else:
                    logger.warning(f"No matching model found for result: {result_name}")
            else:
                logger.warning(f"Could not extract job ID from: {result_name}")

        return duplicates

    def _get_dir_size(self, path: Path) -> float:
        """Get directory size in GB."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    pass
        return total_size / (1024**3)  # Convert to GB

    def fix_duplicates(self, dry_run: bool = True) -> Tuple[int, float]:
        """Fix all duplicate model directories."""
        duplicates = self.find_duplicate_models()

        if not duplicates:
            logger.info("No duplicate models found to fix")
            return 0, 0.0

        logger.info(f"Found {len(duplicates)} duplicate model directories")

        total_saved = 0.0
        fixed_count = 0

        for result_dir, model_dir, size in duplicates:
            model_subdir = result_dir / "model"

            if dry_run:
                logger.info("Would fix:")
                logger.info(f"  Remove duplicate: {model_subdir}")
                logger.info(f"  Size: {size:.2f} GB")
                logger.info(f"  Create symlink: {model_subdir} -> {model_dir}")
            else:
                try:
                    # Remove the duplicate directory
                    shutil.rmtree(model_subdir)
                    logger.info(f"✅ Removed duplicate: {model_subdir}")

                    # Create symlink
                    model_subdir.symlink_to(model_dir)
                    logger.info(f"✅ Created symlink: {model_subdir} -> {model_dir}")

                    fixed_count += 1
                    total_saved += size

                except Exception as e:
                    logger.error(f"❌ Failed to fix {result_dir.name}: {e}")

        action = "Would fix" if dry_run else "Fixed"
        logger.info(f"📈 {action} {len(duplicates)} duplicates")
        if not dry_run:
            logger.info(f"💾 Disk space saved: {total_saved:.2f} GB")

        return fixed_count, total_saved

    def show_summary(self):
        """Show summary of duplicates found."""
        duplicates = self.find_duplicate_models()

        if not duplicates:
            logger.info("✅ No duplicate models found!")
            return

        logger.info("🔍 Duplicate Model Analysis:")
        logger.info(f"📊 Found {len(duplicates)} duplicate model directories")

        total_size = sum(size for _, _, size in duplicates)
        logger.info(f"💾 Total duplicate size: {total_size:.2f} GB")

        logger.info("📁 Duplicate locations:")
        for result_dir, model_dir, size in duplicates[:5]:  # Show first 5
            logger.info(f"  • {result_dir.name}: {size:.2f} GB")

        if len(duplicates) > 5:
            logger.info(f"  ... and {len(duplicates) - 5} more")

        logger.info("💡 After fixing, these will become symlinks saving disk space!")

def main():
    parser = argparse.ArgumentParser(description="Fix duplicate model storage in results directories")
    parser.add_argument("--fix", action="store_true",
                       help="Actually fix the duplicates (default is dry-run)")
    parser.add_argument("--models-dir", default="models/experiments/encoder",
                       help="Path to models directory")
    parser.add_argument("--results-dir", default="results/experiments/encoder",
                       help="Path to results directory")

    args = parser.parse_args()

    fixer = DuplicateFixer(models_dir=args.models_dir, results_dir=args.results_dir)

    # Show current status
    fixer.show_summary()

    if not args.fix:
        logger.info("\n🔍 This was a dry run. To actually fix duplicates, use --fix")
        logger.info("Example: python scripts/fix_duplicate_models.py --fix")
    else:
        logger.info("\n🔧 Fixing duplicates...")
        fixed_count, saved_size = fixer.fix_duplicates(dry_run=False)

        if fixed_count > 0:
            logger.info("🎉 Successfully fixed duplicate models!")
            logger.info(f"💾 Disk space saved: {saved_size:.2f} GB")
        else:
            logger.info("ℹ️ No duplicates were found or fixed")

if __name__ == "__main__":
    main()
