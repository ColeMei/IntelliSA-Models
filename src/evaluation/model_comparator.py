import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelComparator:
    """Compare performance of multiple models."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def compare_models(self, model_results: Dict[str, Dict], test_path: str) -> Dict:
        """Compare multiple models and generate comparison report."""
        logger.info(f"Comparing {len(model_results)} models...")
        
        if len(model_results) < 2:
            logger.warning("Need at least 2 models for comparison")
            return {}
        
        # Create comparison dataframe
        comparison_data = []
        
        for model_name, results in model_results.items():
            metrics = results.get('metrics', {})
            comparison_data.append({
                'model': model_name,
                'accuracy': metrics.get('accuracy', 0.0),
                'precision': metrics.get('precision', 0.0),
                'recall': metrics.get('recall', 0.0),
                'f1': metrics.get('f1', 0.0),
                'avg_confidence': results.get('average_confidence', 0.0),
                'eval_time': results.get('evaluation_time', 0.0),
                'num_samples': results.get('num_samples', 0)
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Generate comparison plots
        self._create_performance_plots(df)
        
        # Generate per-smell comparison
        smell_comparison = self._compare_per_smell_performance(model_results)
        
        # Create ranking
        ranking = self._rank_models(df)
        
        # Statistical significance tests (if we have detailed predictions)
        stat_tests = self._statistical_tests(model_results)
        
        # Generate comparison report
        comparison_results = {
            'test_path': test_path,
            'models_compared': list(model_results.keys()),
            'comparison_summary': df.to_dict('records'),
            'ranking': ranking,
            'per_smell_comparison': smell_comparison,
            'statistical_tests': stat_tests,
            'best_model': {
                'by_accuracy': df.loc[df['accuracy'].idxmax(), 'model'],
                'by_f1': df.loc[df['f1'].idxmax(), 'model'],
                'by_precision': df.loc[df['precision'].idxmax(), 'model'],
                'by_recall': df.loc[df['recall'].idxmax(), 'model'],
            }
        }
        
        # Save comparison results
        results_file = self.output_dir / "model_comparison.json"
        with open(results_file, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        # Generate HTML report
        self._generate_html_report(comparison_results, df)
        
        logger.info(f"Model comparison completed. Results saved to {self.output_dir}")
        
        return comparison_results
    
    def _create_performance_plots(self, df: pd.DataFrame):
        """Create performance comparison plots."""
        plt.style.use('default')
        
        # Metrics comparison bar plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i // 2, i % 2]
            bars = ax.bar(df['model'], df[metric], alpha=0.7)
            ax.set_title(title)
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Radar chart for overall performance
        self._create_radar_chart(df)
        
        # Confusion matrix comparison
        self._create_confusion_matrix_plots(model_results)

        logger.info("Performance plots created")
    
    def _create_radar_chart(self, df: pd.DataFrame):
        """Create radar chart comparing models."""
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        # Number of models and metrics
        N = len(metrics)
        
        # Compute angles for radar chart
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(df)))
        
        for i, (_, row) in enumerate(df.iterrows()):
            values = [row[metric] for metric in metrics]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['model'], color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.capitalize() for m in metrics])
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax.set_title('Model Performance Radar Chart', pad=20)
        
        plt.savefig(self.output_dir / "radar_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_confusion_matrix_plots(self, model_results: Dict[str, Dict]):
        """Create confusion matrix plots for each model."""
        num_models = len(model_results)
        if num_models == 0:
            return

        # Create subplot grid
        cols = min(2, num_models)
        rows = (num_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        if num_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        fig.suptitle('Confusion Matrix Comparison', fontsize=16)

        for i, (model_name, results) in enumerate(model_results.items()):
            if i >= len(axes):
                break

            ax = axes[i]
            cm_data = results.get('confusion_matrix', {})

            # Extract confusion matrix values
            tn = cm_data.get('tn', 0)
            fp = cm_data.get('fp', 0)
            fn = cm_data.get('fn', 0)
            tp = cm_data.get('tp', 0)

            # Create confusion matrix array
            cm_array = np.array([[tn, fp], [fn, tp]])

            # Plot confusion matrix
            im = ax.imshow(cm_array, interpolation='nearest', cmap=plt.cm.Blues, alpha=0.8)

            # Add text annotations
            thresh = cm_array.max() / 2.
            for i_cm in range(cm_array.shape[0]):
                for j_cm in range(cm_array.shape[1]):
                    ax.text(j_cm, i_cm, format(cm_array[i_cm, j_cm], 'd'),
                           ha="center", va="center",
                           color="white" if cm_array[i_cm, j_cm] > thresh else "black",
                           fontsize=14, fontweight='bold')

            ax.set_title(f'{model_name}', fontsize=12, pad=20)
            ax.set_xlabel('Predicted label')
            ax.set_ylabel('True label')

            # Set tick labels
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['FP', 'TP'])
            ax.set_yticklabels(['FP', 'TP'])

            # Add metrics annotation
            metrics = results.get('metrics', {})
            accuracy = metrics.get('accuracy', 0)
            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            f1 = metrics.get('f1', 0)

            metrics_text = '.3f'
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Hide unused subplots
        for i in range(len(model_results), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.output_dir / "confusion_matrices.png", dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("Confusion matrix plots created")

    def _compare_per_smell_performance(self, model_results: Dict[str, Dict]) -> Dict:
        """Compare per-smell performance across models."""
        smell_comparison = {}
        
        # Get all unique smells
        all_smells = set()
        for results in model_results.values():
            smell_metrics = results.get('smell_metrics', {})
            all_smells.update(smell_metrics.keys())
        
        for smell in all_smells:
            smell_data = {}
            for model_name, results in model_results.items():
                smell_metrics = results.get('smell_metrics', {})
                if smell in smell_metrics:
                    smell_data[model_name] = smell_metrics[smell]
                else:
                    smell_data[model_name] = {
                        'count': 0, 'accuracy': 0.0, 'precision': 0.0, 
                        'recall': 0.0, 'f1': 0.0, 'tp_actual': 0, 'fp_actual': 0
                    }
            
            smell_comparison[smell] = smell_data
        
        return smell_comparison
    
    def _rank_models(self, df: pd.DataFrame) -> List[Dict]:
        """Rank models by different metrics."""
        ranking = []
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        for metric in metrics:
            ranked_df = df.sort_values(metric, ascending=False)
            metric_ranking = []
            
            for i, (_, row) in enumerate(ranked_df.iterrows()):
                metric_ranking.append({
                    'rank': i + 1,
                    'model': row['model'],
                    'score': row[metric]
                })
            
            ranking.append({
                'metric': metric,
                'ranking': metric_ranking
            })
        
        # Overall ranking (weighted average)
        weights = {'accuracy': 0.25, 'precision': 0.25, 'recall': 0.25, 'f1': 0.25}
        df['weighted_score'] = sum(df[metric] * weight for metric, weight in weights.items())
        
        overall_ranked = df.sort_values('weighted_score', ascending=False)
        overall_ranking = []
        
        for i, (_, row) in enumerate(overall_ranked.iterrows()):
            overall_ranking.append({
                'rank': i + 1,
                'model': row['model'],
                'weighted_score': row['weighted_score'],
                'individual_scores': {
                    'accuracy': row['accuracy'],
                    'precision': row['precision'],
                    'recall': row['recall'],
                    'f1': row['f1']
                }
            })
        
        ranking.append({
            'metric': 'overall_weighted',
            'ranking': overall_ranking
        })
        
        return ranking
    
    def _statistical_tests(self, model_results: Dict[str, Dict]) -> Dict:
        """Perform statistical significance tests (placeholder)."""
        # This would require detailed predictions from each model
        # For now, return basic comparison
        
        stat_tests = {
            'note': 'Statistical tests require detailed prediction data',
            'tests_performed': [],
            'p_values': {}
        }
        
        # TODO: Implement McNemar's test, paired t-test, etc.
        # when detailed predictions are available
        
        return stat_tests
    
    def _generate_html_report(self, comparison_results: Dict, df: pd.DataFrame):
        """Generate HTML comparison report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ font-weight: bold; }}
                .best {{ background-color: #90EE90; }}
                .image {{ text-align: center; margin: 20px 0; }}
                h1, h2 {{ color: #333; }}
            </style>
        </head>
        <body>
            <h1>Chef Detection Model Comparison Report</h1>
            
            <h2>Summary</h2>
            <p><strong>Test Dataset:</strong> {comparison_results['test_path']}</p>
            <p><strong>Models Compared:</strong> {', '.join(comparison_results['models_compared'])}</p>
            
            <h2>Performance Comparison</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>Avg Confidence</th>
                    <th>Eval Time (s)</th>
                </tr>
        """
        
        # Find best scores for highlighting
        best_acc = df['accuracy'].max()
        best_prec = df['precision'].max()
        best_rec = df['recall'].max()
        best_f1 = df['f1'].max()
        
        for _, row in df.iterrows():
            html_content += f"""
                <tr>
                    <td class="metric">{row['model']}</td>
                    <td{'class="best"' if row['accuracy'] == best_acc else ''}>{row['accuracy']:.4f}</td>
                    <td{'class="best"' if row['precision'] == best_prec else ''}>{row['precision']:.4f}</td>
                    <td{'class="best"' if row['recall'] == best_rec else ''}>{row['recall']:.4f}</td>
                    <td{'class="best"' if row['f1'] == best_f1 else ''}>{row['f1']:.4f}</td>
                    <td>{row['avg_confidence']:.4f}</td>
                    <td>{row['eval_time']:.2f}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Best Models</h2>
            <ul>
        """
        
        best_models = comparison_results['best_model']
        for metric, model in best_models.items():
            html_content += f"<li><strong>Best {metric.replace('_', ' ').title()}:</strong> {model}</li>"
        
        html_content += """
            </ul>
            
            <div class="image">
                <h2>Performance Visualization</h2>
                <img src="performance_comparison.png" alt="Performance Comparison" style="max-width: 100%;"/>
                <br><br>
                <img src="radar_comparison.png" alt="Radar Chart" style="max-width: 100%;"/>
                <br><br>
                <img src="confusion_matrices.png" alt="Confusion Matrix Comparison" style="max-width: 100%;"/>
            </div>
            
            <h2>Ranking</h2>
        """
        
        # Add ranking tables
        for ranking_data in comparison_results['ranking']:
            if ranking_data['metric'] == 'overall_weighted':
                html_content += f"<h3>Overall Ranking (Weighted)</h3>"
            else:
                html_content += f"<h3>{ranking_data['metric'].title()} Ranking</h3>"
            
            html_content += "<table><tr><th>Rank</th><th>Model</th><th>Score</th></tr>"
            
            for item in ranking_data['ranking']:
                score_key = 'weighted_score' if 'weighted_score' in item else 'score'
                html_content += f"<tr><td>{item['rank']}</td><td>{item['model']}</td><td>{item[score_key]:.4f}</td></tr>"
            
            html_content += "</table>"
        
        html_content += """
            </body>
        </html>
        """
        
        # Save HTML report
        html_file = self.output_dir / "comparison_report.html"
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {html_file}")