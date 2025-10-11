#!/usr/bin/env python3
"""
Compare performance across different label variants.
Used for label quality ablation analysis.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd

# Add repo src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))


def load_evaluation_results(results_dir: Path, prefix: str) -> Dict[str, Dict]:
    """Load all evaluation results matching prefix."""
    results = {}
    
    eval_dir = results_dir / "experiments" / "evaluation" / "label_comparison"
    if not eval_dir.exists():
        print(f"❌ Evaluation directory not found: {eval_dir}")
        return results
    
    for model_dir in sorted(eval_dir.glob(f"{prefix}*")):
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        model_results = {}
        
        # Load results for each test set
        for test_set in ['combined', 'chef', 'ansible', 'puppet']:
            result_file = model_dir / test_set / "encoder_eval" / "evaluation_results.json"
            if result_file.exists():
                with open(result_file) as f:
                    model_results[test_set] = json.load(f)
        
        if model_results:
            results[model_name] = model_results
    
    return results


def extract_label_variant(model_name: str) -> str:
    """Extract label variant from model name."""
    if "processed_label1" in model_name:
        return "label1"
    elif "processed_label2" in model_name:
        return "label2"
    elif "processed_label3" in model_name:
        return "label3"
    else:
        return "unknown"


def extract_seed(model_name: str) -> int:
    """Extract seed from model name."""
    import re
    match = re.search(r'seed(\d+)', model_name)
    return int(match.group(1)) if match else 0


def aggregate_by_variant(results: Dict[str, Dict]) -> pd.DataFrame:
    """Aggregate results by label variant."""
    rows = []
    
    for model_name, model_results in results.items():
        variant = extract_label_variant(model_name)
        seed = extract_seed(model_name)
        
        for test_set, metrics in model_results.items():
            row = {
                'label_variant': variant,
                'seed': seed,
                'test_set': test_set,
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1': metrics.get('f1', 0)
            }
            rows.append(row)
    
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame):
    """Print summary statistics by label variant."""
    print("\n" + "="*80)
    print("LABEL VARIANT COMPARISON SUMMARY")
    print("="*80 + "\n")
    
    # Overall comparison (combined test set)
    print("📊 Overall Performance (Combined Test Set)")
    print("-" * 80)
    combined_df = df[df['test_set'] == 'combined']
    summary = combined_df.groupby('label_variant')[['accuracy', 'precision', 'recall', 'f1']].agg(['mean', 'std'])
    print(summary)
    print()
    
    # Per-technology comparison
    for tech in ['chef', 'ansible', 'puppet']:
        print(f"📊 {tech.upper()} Performance")
        print("-" * 80)
        tech_df = df[df['test_set'] == tech]
        if not tech_df.empty:
            summary = tech_df.groupby('label_variant')[['accuracy', 'precision', 'recall', 'f1']].agg(['mean', 'std'])
            print(summary)
        print()
    
    # Best performing variant
    print("🏆 Best Performing Label Variant (by F1)")
    print("-" * 80)
    best_by_testset = combined_df.groupby('test_set')['f1'].mean().idxmax()
    best_variant = combined_df.groupby('label_variant')['f1'].mean().idxmax()
    best_f1 = combined_df.groupby('label_variant')['f1'].mean().max()
    print(f"   Winner: {best_variant}")
    print(f"   F1 Score: {best_f1:.4f}")
    print()
    
    # Ranking
    print("📈 Ranking by Average F1 Score")
    print("-" * 80)
    ranking = combined_df.groupby('label_variant')['f1'].mean().sort_values(ascending=False)
    for rank, (variant, f1) in enumerate(ranking.items(), 1):
        print(f"   {rank}. {variant}: {f1:.4f}")
    print()


def export_detailed_comparison(df: pd.DataFrame, output_file: Path):
    """Export detailed comparison to CSV."""
    # Pivot for easier comparison
    pivot = df.pivot_table(
        index=['test_set', 'seed'],
        columns='label_variant',
        values=['accuracy', 'precision', 'recall', 'f1']
    )
    
    pivot.to_csv(output_file)
    print(f"✅ Detailed comparison exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Compare label variant performance")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent.parent / "results",
        help="Results directory"
    )
    parser.add_argument(
        "--prefix",
        default="codet5p_220m_label_comparison_",
        help="Model name prefix to filter"
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export detailed comparison to CSV"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV file path"
    )
    
    args = parser.parse_args()
    
    # Load results
    print(f"🔍 Loading evaluation results from: {args.results_dir}")
    print(f"   Prefix filter: {args.prefix}")
    results = load_evaluation_results(args.results_dir, args.prefix)
    
    if not results:
        print("❌ No evaluation results found!")
        sys.exit(1)
    
    print(f"✅ Found {len(results)} models")
    
    # Aggregate by variant
    df = aggregate_by_variant(results)
    
    if df.empty:
        print("❌ No data to analyze!")
        sys.exit(1)
    
    # Print summary
    print_summary(df)
    
    # Export if requested
    if args.export_csv:
        output_file = args.output or Path("label_variant_comparison.csv")
        export_detailed_comparison(df, output_file)
    
    print("="*80)
    print("✅ Label variant comparison complete!")
    print("="*80)


if __name__ == "__main__":
    main()
