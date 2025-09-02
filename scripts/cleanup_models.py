#!/usr/bin/env python3
"""
Disk cleanup script for trained models and checkpoints.
Helps manage disk quota by removing completed experiments from both models/ and results/ directories.
"""

import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelCleanup:
    def __init__(self, models_dir: str = "models/experiments/encoder",
                 results_dir: str = "results/experiments/encoder"):
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.project_root = Path(__file__).parent.parent

    def get_model_sizes(self) -> Dict[str, Dict]:
        """Get comprehensive model information from both models/ and results/ directories."""
        model_info = {}

        # Scan models directory
        if self.models_dir.exists():
            for item in self.models_dir.iterdir():
                if item.is_dir() and not item.name.endswith('_latest'):
                    base_name = item.name
                    size = self._get_dir_size(item)
                    model_info[base_name] = {
                        'model_size': size,
                        'model_path': item,
                        'results_path': None,
                        'symlink_path': None,
                        'total_size': size
                    }

                    # Find corresponding symlink
                    symlink_path = self.models_dir / f"{base_name}_latest"
                    if symlink_path.exists():
                        model_info[base_name]['symlink_path'] = symlink_path
        else:
            logger.warning(f"Models directory {self.models_dir} does not exist")

        # Scan results directory
        if self.results_dir.exists():
            for item in self.results_dir.iterdir():
                if item.is_dir():
                    base_name = item.name
                    size = self._get_dir_size(item)

                    if base_name in model_info:
                        model_info[base_name]['results_size'] = size
                        model_info[base_name]['results_path'] = item
                        model_info[base_name]['total_size'] = model_info[base_name]['model_size'] + size
                    else:
                        model_info[base_name] = {
                            'model_size': 0,
                            'results_size': size,
                            'model_path': None,
                            'results_path': item,
                            'symlink_path': None,
                            'total_size': size
                        }
        else:
            logger.warning(f"Results directory {self.results_dir} does not exist")

        return model_info

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

    def show_disk_usage(self):
        """Show current disk usage."""
        logger.info("🔍 Analyzing disk usage...")

        model_info = self.get_model_sizes()

        # Sort by total size
        sorted_models = sorted(model_info.items(), key=lambda x: x[1]['total_size'], reverse=True)

        total_model_size = sum(info['model_size'] for info in model_info.values())
        total_results_size = sum(info.get('results_size', 0) for info in model_info.values())
        total_size = total_model_size + total_results_size

        logger.info(f"📊 Total models: {len(model_info)}")
        logger.info(".2f")
        logger.info(".2f")
        logger.info(".2f")

        logger.info("📁 Largest models (by total size):")
        for name, info in sorted_models[:10]:
            logger.info(".2f")

        # Show directory breakdown
        if self.models_dir.exists():
            logger.info(f"📂 Models directory: {self.models_dir}")
        if self.results_dir.exists():
            logger.info(f"📂 Results directory: {self.results_dir}")

    def cleanup_by_pattern(self, pattern: str, dry_run: bool = True):
        """Clean up models matching a pattern from both models/ and results/ directories."""
        logger.info(f"🧹 Cleaning up models matching '{pattern}'")

        model_info = self.get_model_sizes()
        cleaned_size = 0
        cleaned_count = 0

        for name, info in model_info.items():
            if pattern in name:
                if dry_run:
                    logger.info(f"Would remove: {name}")
                    if info['model_path']:
                        logger.info(".2f")
                    if info['results_path']:
                        logger.info(".2f")
                    if info['symlink_path']:
                        logger.info(f"  - Symlink: {info['symlink_path'].name}")
                    logger.info(".2f")
                else:
                    # Remove model directory
                    if info['model_path']:
                        try:
                            shutil.rmtree(info['model_path'])
                            logger.info(f"✅ Removed model: {name}")
                        except Exception as e:
                            logger.error(f"❌ Failed to remove model {name}: {e}")

                    # Remove results directory
                    if info['results_path']:
                        try:
                            shutil.rmtree(info['results_path'])
                            logger.info(f"✅ Removed results: {name}")
                        except Exception as e:
                            logger.error(f"❌ Failed to remove results {name}: {e}")

                    # Remove symlink
                    if info['symlink_path']:
                        try:
                            info['symlink_path'].unlink()
                            logger.info(f"✅ Removed symlink: {info['symlink_path'].name}")
                        except Exception as e:
                            logger.error(f"❌ Failed to remove symlink {info['symlink_path']}: {e}")

                cleaned_size += info['total_size']
                cleaned_count += 1

        action = "Would clean up" if dry_run else "Cleaned up"
        logger.info(f"📈 {action} {cleaned_count} models, freeing {cleaned_size:.2f} GB")
        return cleaned_size, cleaned_count

    def cleanup_by_size(self, min_size_gb: float, dry_run: bool = True):
        """Clean up models larger than specified size from both directories."""
        logger.info(f"🧹 Cleaning up models larger than {min_size_gb} GB")

        model_info = self.get_model_sizes()
        candidates = [(name, info) for name, info in model_info.items() if info['total_size'] > min_size_gb]

        cleaned_size = 0
        cleaned_count = 0

        for name, info in candidates:
            if dry_run:
                logger.info(f"Would remove: {name}")
                if info['model_path']:
                    logger.info(".2f")
                if info['results_path']:
                    logger.info(".2f")
                if info['symlink_path']:
                    logger.info(f"  - Symlink: {info['symlink_path'].name}")
                logger.info(".2f")
            else:
                # Remove model directory
                if info['model_path']:
                    try:
                        shutil.rmtree(info['model_path'])
                        logger.info(f"✅ Removed model: {name}")
                    except Exception as e:
                        logger.error(f"❌ Failed to remove model {name}: {e}")

                # Remove results directory
                if info['results_path']:
                    try:
                        shutil.rmtree(info['results_path'])
                        logger.info(f"✅ Removed results: {name}")
                    except Exception as e:
                        logger.error(f"❌ Failed to remove results {name}: {e}")

                # Remove symlink
                if info['symlink_path']:
                    try:
                        info['symlink_path'].unlink()
                        logger.info(f"✅ Removed symlink: {info['symlink_path'].name}")
                    except Exception as e:
                        logger.error(f"❌ Failed to remove symlink {info['symlink_path']}: {e}")

            cleaned_size += info['total_size']
            cleaned_count += 1

        action = "Would clean up" if dry_run else "Cleaned up"
        logger.info(f"📈 {action} {cleaned_count} models, freeing {cleaned_size:.2f} GB")
        return cleaned_size, cleaned_count

    def cleanup_checkpoints_only(self, pattern: str = None, dry_run: bool = True):
        """Clean up only model checkpoints, keep evaluation results."""
        logger.info("🧹 Cleaning up model checkpoints only (keeping evaluation results)")

        model_info = self.get_model_sizes()
        cleaned_size = 0
        cleaned_count = 0

        for name, info in model_info.items():
            # Filter by pattern if specified
            if pattern and pattern not in name:
                continue

            if info['model_path'] and info['model_size'] > 0:
                if dry_run:
                    logger.info(f"Would remove checkpoints: {name} ({info['model_size']:.2f} GB)")
                    if info['symlink_path']:
                        logger.info(f"  - Symlink: {info['symlink_path'].name}")
                else:
                    # Remove model directory (checkpoints)
                    try:
                        shutil.rmtree(info['model_path'])
                        logger.info(f"✅ Removed checkpoints: {name}")
                    except Exception as e:
                        logger.error(f"❌ Failed to remove checkpoints {name}: {e}")
                        continue

                    # Remove symlink
                    if info['symlink_path']:
                        try:
                            info['symlink_path'].unlink()
                            logger.info(f"✅ Removed symlink: {info['symlink_path'].name}")
                        except Exception as e:
                            logger.error(f"❌ Failed to remove symlink {info['symlink_path']}: {e}")

                cleaned_size += info['model_size']
                cleaned_count += 1

        action = "Would clean up" if dry_run else "Cleaned up"
        logger.info(f"📈 {action} {cleaned_count} model checkpoints, freeing {cleaned_size:.2f} GB")
        logger.info("💡 Evaluation results preserved in results/ directory")

        return cleaned_size, cleaned_count

    def show_cleanup_plan(self, pattern: str = None, min_size: float = None):
        """Show detailed cleanup plan without actually removing anything."""
        logger.info("📋 Cleanup Plan Analysis")

        model_info = self.get_model_sizes()

        if pattern:
            candidates = {name: info for name, info in model_info.items() if pattern in name}
            non_candidates = {name: info for name, info in model_info.items() if pattern not in name}
            plan_type = f"models matching '{pattern}'"
        elif min_size:
            candidates = {name: info for name, info in model_info.items() if info['total_size'] > min_size}
            non_candidates = {name: info for name, info in model_info.items() if info['total_size'] <= min_size}
            plan_type = f"models larger than {min_size} GB"
        else:
            logger.info("❌ Please specify --pattern or --min-size for cleanup plan")
            return

        # Calculate statistics
        total_candidates = len(candidates)
        total_kept = len(non_candidates)
        size_to_remove = sum(info['total_size'] for info in candidates.values())
        size_to_keep = sum(info['total_size'] for info in non_candidates.values())

        logger.info(f"🎯 Plan: Remove {plan_type}")
        logger.info(f"📊 Models to remove: {total_candidates}")
        logger.info(f"📊 Models to keep: {total_kept}")
        logger.info(".2f")
        logger.info(".2f")

        if candidates:
            logger.info("\n🗑️  Models to be REMOVED:")
            for name, info in sorted(candidates.items(), key=lambda x: x[1]['total_size'], reverse=True):
                logger.info(f"  • {name}: {info['total_size']:.2f} GB")

        if non_candidates:
            logger.info("\n💾 Models to be KEPT:")
            kept_sample = dict(list(non_candidates.items())[:5])  # Show first 5
            for name, info in sorted(kept_sample.items(), key=lambda x: x[1]['total_size'], reverse=True):
                logger.info(f"  • {name}: {info['total_size']:.2f} GB")
            if len(non_candidates) > 5:
                logger.info(f"  ... and {len(non_candidates) - 5} more models")

def main():
    parser = argparse.ArgumentParser(description="Clean up trained models and results to manage disk space")
    parser.add_argument("--pattern", help="Remove models matching pattern (e.g., 'codet5_large', 'codebert')")
    parser.add_argument("--min-size", type=float, help="Remove models larger than X GB")
    parser.add_argument("--checkpoints-only", action="store_true",
                       help="Remove only model checkpoints, keep evaluation results")
    parser.add_argument("--plan", action="store_true",
                       help="Show detailed cleanup plan without removing anything")
    parser.add_argument("--dry-run", action="store_true", default=True,
                       help="Show what would be removed without actually removing")
    parser.add_argument("--force", action="store_true",
                       help="Actually perform cleanup (overrides dry-run)")
    parser.add_argument("--models-dir", default="models/experiments/encoder",
                       help="Path to models directory")
    parser.add_argument("--results-dir", default="results/experiments/encoder",
                       help="Path to results directory")

    args = parser.parse_args()

    if args.force:
        args.dry_run = False

    cleanup = ModelCleanup(models_dir=args.models_dir, results_dir=args.results_dir)

    # Show current usage
    cleanup.show_disk_usage()

    # Handle plan mode
    if args.plan:
        if args.pattern:
            cleanup.show_cleanup_plan(pattern=args.pattern)
        elif args.min_size:
            cleanup.show_cleanup_plan(min_size=args.min_size)
        else:
            logger.info("❌ Please specify --pattern or --min-size with --plan")
        return

    # Perform cleanup
    if args.checkpoints_only:
        cleanup.cleanup_checkpoints_only(args.pattern, args.dry_run)
    elif args.pattern:
        cleanup.cleanup_by_pattern(args.pattern, args.dry_run)
    elif args.min_size:
        cleanup.cleanup_by_size(args.min_size, args.dry_run)
    else:
        logger.info("\n📝 Usage examples:")
        logger.info("  # Check current disk usage")
        logger.info("  python scripts/cleanup_models.py")
        logger.info("")
        logger.info("  # Show detailed cleanup plan")
        logger.info("  python scripts/cleanup_models.py --plan --pattern 'codet5_large'")
        logger.info("  python scripts/cleanup_models.py --plan --min-size 5.0")
        logger.info("")
        logger.info("  # Show what would be cleaned up (dry run)")
        logger.info("  python scripts/cleanup_models.py --pattern 'codet5_large'")
        logger.info("  python scripts/cleanup_models.py --min-size 5.0")
        logger.info("  python scripts/cleanup_models.py --checkpoints-only")
        logger.info("")
        logger.info("  # Actually perform cleanup")
        logger.info("  python scripts/cleanup_models.py --pattern 'codet5_large' --force")
        logger.info("  python scripts/cleanup_models.py --min-size 5.0 --force")
        logger.info("  python scripts/cleanup_models.py --checkpoints-only --force")
        logger.info("")
        logger.info("  # HPC usage examples")
        logger.info("  python scripts/cleanup_models.py --pattern 'codet5' --force")
        logger.info("  python scripts/cleanup_models.py --min-size 2.0 --force")
        logger.info("  python scripts/cleanup_models.py --checkpoints-only --pattern 'large' --force")

if __name__ == "__main__":
    main()
