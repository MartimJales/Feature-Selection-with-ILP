"""
Results aggregator: consolidates outputs and produces reporting tables.
"""

import pandas as pd
import json
import logging
from pathlib import Path
from typing import List, Dict
from .ilp_runner import ILPRunResult

logger = logging.getLogger(__name__)


class ResultsAggregator:
    """Aggregates and reports Idea2 results."""

    def __init__(self, output_dir: Path = Path("./reports/idea2")):
        """
        Initialize aggregator.

        Args:
            output_dir: Base output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.all_results = []

    def add_result(self, result: ILPRunResult) -> None:
        """Add a run result."""
        self.all_results.append(result)

    def add_results(self, results: List[ILPRunResult]) -> None:
        """Add multiple run results."""
        self.all_results.extend(results)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        rows = []
        for result in self.all_results:
            row = {
                'window_id': result.window_id,
                'sample_size': result.sample_size,
                'seed': result.seed,
                'n_features': result.n_features,
                'n_rules': result.n_rules,
                'status': result.status,
                'elapsed_time': result.elapsed_time,
                'solver_time': result.solver_time,
                'train_accuracy': result.train_accuracy,
                'train_precision': result.train_precision,
                'train_recall': result.train_recall,
                'train_f1': result.train_f1,
                'val_accuracy': result.val_accuracy,
                'val_precision': result.val_precision,
                'val_recall': result.val_recall,
                'val_f1': result.val_f1,
                'error_message': result.error_message,
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def save_summary(self, filename: str = "summary.csv") -> Path:
        """Save summary table."""
        df = self.to_dataframe()
        path = self.output_dir / filename
        df.to_csv(path, index=False)
        logger.info(f"✓ Summary saved to {path}")
        return path

    def save_by_window(self, filename: str = "by_window.csv") -> Path:
        """Aggregate by window."""
        df = self.to_dataframe()

        # Group by window
        by_window = df.groupby('window_id').agg({
            'n_features': 'first',
            'n_rules': ['mean', 'std', 'min', 'max'],
            'status': lambda x: (x == 'success').sum() / len(x),  # success rate
            'elapsed_time': ['mean', 'max'],
            'train_accuracy': 'mean',
            'train_f1': 'mean',
        }).round(4)

        by_window.columns = ['_'.join(col).strip('_') for col in by_window.columns.values]

        path = self.output_dir / filename
        by_window.to_csv(path)
        logger.info(f"✓ By-window aggregation saved to {path}")
        return path

    def save_by_sample_size(self, filename: str = "by_sample_size.csv") -> Path:
        """Aggregate by sample size."""
        df = self.to_dataframe()

        by_size = df.groupby('sample_size').agg({
            'n_rules': ['mean', 'std', 'min', 'max'],
            'status': lambda x: (x == 'success').sum() / len(x),
            'elapsed_time': ['mean', 'max'],
            'train_accuracy': 'mean',
            'train_f1': 'mean',
        }).round(4)

        by_size.columns = ['_'.join(col).strip('_') for col in by_size.columns.values]

        path = self.output_dir / filename
        by_size.to_csv(path)
        logger.info(f"✓ By-sample-size aggregation saved to {path}")
        return path

    def save_top_runs(self, top_k: int = 10, filename: str = "top_runs.csv") -> Path:
        """Save top runs by success + rules quality."""
        df = self.to_dataframe()

        # Score: success * n_rules * mean_f1
        df['score'] = (
            (df['status'] == 'success').astype(int) *
            df['n_rules'] *
            df['train_f1'].fillna(0)
        )

        top = df.nlargest(top_k, 'score')

        path = self.output_dir / filename
        top.to_csv(path, index=False)
        logger.info(f"✓ Top {top_k} runs saved to {path}")
        return path

    def save_rules(self, filename_template: str = "rules_w{window_id}_s{seed}.txt") -> List[Path]:
        """Save rules from each run."""
        paths = []

        for result in self.all_results:
            if result.rules:
                filename = filename_template.format(window_id=result.window_id, seed=result.seed)
                path = self.output_dir / "rules" / filename
                path.parent.mkdir(parents=True, exist_ok=True)

                with open(path, 'w') as f:
                    f.write(f"# Window {result.window_id}, Sample {result.sample_size}, Seed {result.seed}\n")
                    f.write(f"# Features: {', '.join(result.feature_names)}\n")
                    f.write(f"# Status: {result.status}, N Rules: {len(result.rules)}\n")
                    f.write("#\n")
                    for rule in result.rules:
                        f.write(f"{rule}\n")

                paths.append(path)
                logger.debug(f"Saved {len(result.rules)} rules to {path}")

        logger.info(f"✓ Saved rules from {len(paths)} runs")
        return paths

    def print_summary(self) -> None:
        """Print summary statistics."""
        df = self.to_dataframe()

        print("\n" + "="*80)
        print("IDEA 2 RESULTS SUMMARY")
        print("="*80)
        print(f"\nTotal runs: {len(df)}")
        print(f"Windows: {df['window_id'].max() + 1}")
        print(f"Sample sizes: {sorted(df['sample_size'].unique())}")
        print(f"Seeds: {sorted(df['seed'].unique())}")

        print(f"\nStatus distribution:")
        print(df['status'].value_counts().to_string())

        print(f"\nSuccess rate: {(df['status'] == 'success').sum() / len(df):.2%}")

        print(f"\nMetrics (on successful runs):")
        df_success = df[df['status'] == 'success']
        if len(df_success) > 0:
            print(f"  Avg rules: {df_success['n_rules'].mean():.1f}")
            print(f"  Avg elapsed time: {df_success['elapsed_time'].mean():.1f}s")
            if df_success['train_accuracy'].notna().any():
                print(f"  Avg train accuracy: {df_success['train_accuracy'].mean():.3f}")

        print("\nBy window:")
        print(df.groupby('window_id')[['n_rules', 'elapsed_time', 'status']].apply(
            lambda x: f"Rules: {x['n_rules'].mean():.1f}, Time: {x['elapsed_time'].mean():.1f}s, "
                     f"Success: {(x['status'] == 'success').sum()}/{len(x)}"
        ).to_string())

        print("\n" + "="*80)
