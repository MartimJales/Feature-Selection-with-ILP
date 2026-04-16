"""
Idea2 Pipeline: orchestrates the full Feature Windows + ILP workflow.

Phases:
- Phase A: Proof of viability (fast)
- Phase B: Benchmark main
- Phase C: Expansion (optional)
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
import json

from .data_loader import Idea2DataLoader
from .window_generator import FeatureWindowGenerator
from .sampling import SamplingStrategy
from .ilp_runner import ILPRunner, ILPRunResult
from .aggregator import ResultsAggregator

logger = logging.getLogger(__name__)


class Idea2Pipeline:
    """Main orchestrator for Idea2 experiments."""

    def __init__(
        self,
        features_path: str = "./reports/extracted_features.parquet",
        labels_path: str = "./data/training_set.csv",
        rankings_path: str = "./reports/feature_analysis/feature_rankings_all.parquet",
        output_dir: str = "./reports/idea2",
        padtai_dir: str = "./PADTAI",
        max_timeout: int = 1800,
        window_size: int = 30,
        debug: bool = False,
    ):
        """
        Initialize pipeline.

        Args:
            features_path: Path to extracted features file
            labels_path: Path to labels CSV
            rankings_path: Path to rankings Parquet
            output_dir: Output directory
            padtai_dir: PADTAI installation directory
            max_timeout: Max ILP timeout per run (seconds)
            window_size: Feature window size (default: 30)
            debug: Enable ILP raw-output debug logging
        """
        self.features_path = features_path
        self.labels_path = labels_path
        self.rankings_path = rankings_path
        self.output_dir = Path(output_dir)
        self.padtai_dir = padtai_dir
        self.max_timeout = max_timeout
        self.window_size = window_size
        self.debug = debug

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.X = None
        self.y = None
        self.df_rankings = None
        self.label_column = None

        self.windows = []
        self.all_results = []

    def initialize(self) -> None:
        """Load data and prepare pipeline."""
        logger.info("="*60)
        logger.info("INITIALIZING IDEA2 PIPELINE")
        logger.info("="*60)

        # Load data
        loader = Idea2DataLoader(self.features_path, self.labels_path, self.rankings_path)
        self.X, self.y, self.df_rankings = loader.load()
        self.label_column = loader.get_label_column()

        logger.info(f"✓ Data loaded: {self.X.shape}")
        logger.info(f"✓ Label column: {self.label_column}")

    def generate_windows(self, n_windows: int = 5) -> None:
        """Generate feature windows."""
        logger.info("\n" + "="*60)
        logger.info("GENERATING FEATURE WINDOWS")
        logger.info("="*60)

        available_features = set(self.X.columns)
        ranked_features = [
            feature for feature in self.df_rankings['feature'].astype(str).tolist()
            if feature in available_features
        ]
        missing_ranked = len(self.df_rankings) - len(ranked_features)
        if missing_ranked:
            logger.warning(f"⚠ Skipping {missing_ranked} ranked features not present in the extracted matrix")

        generator = FeatureWindowGenerator(ranked_features, window_size=self.window_size)

        self.windows = generator.generate_windows(n_windows=n_windows)

        logger.info(f"\n{generator.windows_summary()}")

    def run_phase_a(self, n_seeds: int = 2) -> None:
        """
        Phase A: Proof of viability (fast)

        - Windows: [1-30, 31-60, 61-90]
        - Sample size: medium (50%)
        - Seeds: 2 per window
        """
        logger.info("\n" + "="*80)
        logger.info("PHASE A: PROOF OF VIABILITY")
        logger.info("="*80)

        if len(self.windows) < 3:
            logger.warning(f"Only {len(self.windows)} windows available, expected 3+")
            windows_to_test = self.windows
        else:
            windows_to_test = self.windows[:3]

        sampler = SamplingStrategy()
        sample_sizes = sampler.get_sample_sizes(len(self.X), num_levels=3)
        medium_sample_size = sample_sizes[1] if len(sample_sizes) > 1 else sample_sizes[0]

        logger.info(f"Windows: {len(windows_to_test)}")
        logger.info(f"Sample size: {medium_sample_size} ({medium_sample_size/len(self.X):.1%})")
        logger.info(f"Seeds: {n_seeds}")
        logger.info(f"Total runs: {len(windows_to_test) * n_seeds}")

        runner = ILPRunner(
            padtai_dir=self.padtai_dir,
            max_timeout=self.max_timeout,
            debug=self.debug,
            debug_output_dir=str(self.output_dir / "ilp_logs"),
        )

        for window_id, features in enumerate(windows_to_test):
            for seed in range(n_seeds):
                # Sample with seed
                sampler_seed = SamplingStrategy(random_seed=42 + seed)
                X_sample, y_sample, indices = sampler_seed.stratified_sample(
                    self.X, self.y, medium_sample_size
                )

                # Run ILP
                result = runner.run(
                    X=X_sample,
                    y=y_sample,
                    features=features,
                    window_id=window_id,
                    sample_size=medium_sample_size,
                    seed=seed,
                    output_dir=self.output_dir,
                    label_column=self.label_column,
                )

                self.all_results.append(result)

        logger.info(f"\n✓ Phase A complete: {len([r for r in self.all_results if r.status == 'success'])} successful runs")

    def run_phase_b(self, n_seeds: int = 3) -> None:
        """
        Phase B: Benchmark main

        - Windows: first 5-10
        - Sample sizes: 3 levels (10%, 50%, 100%)
        - Seeds: 3 per combination
        """
        logger.info("\n" + "="*80)
        logger.info("PHASE B: BENCHMARK MAIN")
        logger.info("="*80)

        if len(self.windows) < 5:
            logger.warning(f"Only {len(self.windows)} windows, expected 5+")
            windows_to_test = self.windows
        else:
            windows_to_test = self.windows[:10]

        sampler = SamplingStrategy()
        sample_sizes = sampler.get_sample_sizes(len(self.X), num_levels=3)

        logger.info(f"Windows: {len(windows_to_test)}")
        logger.info(f"Sample sizes: {sample_sizes}")
        logger.info(f"Seeds: {n_seeds}")
        total_runs = len(windows_to_test) * len(sample_sizes) * n_seeds
        logger.info(f"Total runs: {total_runs}")

        runner = ILPRunner(
            padtai_dir=self.padtai_dir,
            max_timeout=self.max_timeout,
            debug=self.debug,
            debug_output_dir=str(self.output_dir / "ilp_logs"),
        )

        run_count = 0
        for window_id, features in enumerate(windows_to_test):
            for sample_size in sample_sizes:
                for seed in range(n_seeds):
                    run_count += 1
                    logger.info(f"\n[{run_count}/{total_runs}] Window {window_id}, Size {sample_size}, Seed {seed}")

                    sampler_seed = SamplingStrategy(random_seed=42 + seed)
                    X_sample, y_sample, indices = sampler_seed.stratified_sample(
                        self.X, self.y, sample_size
                    )

                    result = runner.run(
                        X=X_sample,
                        y=y_sample,
                        features=features,
                        window_id=window_id,
                        sample_size=sample_size,
                        seed=seed,
                        output_dir=self.output_dir,
                        label_column=self.label_column,
                    )

                    self.all_results.append(result)

        logger.info(f"\n✓ Phase B complete: {len([r for r in self.all_results if r.status == 'success'])} successful runs")

    def consolidate_results(self) -> None:
        """Consolidate and save results."""
        logger.info("\n" + "="*80)
        logger.info("CONSOLIDATING RESULTS")
        logger.info("="*80)

        aggregator = ResultsAggregator(output_dir=self.output_dir)
        aggregator.add_results(self.all_results)

        # Save outputs
        aggregator.save_summary("summary.csv")
        aggregator.save_by_window("by_window.csv")
        aggregator.save_by_sample_size("by_sample_size.csv")
        aggregator.save_top_runs(top_k=15, filename="top_runs.csv")
        aggregator.save_rules()

        # Print summary
        aggregator.print_summary()

        # Save configuration
        config = {
            'features_path': self.features_path,
            'labels_path': self.labels_path,
            'rankings_path': self.rankings_path,
            'output_dir': str(self.output_dir),
            'padtai_dir': self.padtai_dir,
            'max_timeout': self.max_timeout,
            'window_size': self.window_size,
            'debug': self.debug,
            'n_windows': len(self.windows),
            'n_results': len(self.all_results),
            'n_successful': len([r for r in self.all_results if r.status == 'success']),
        }

        config_path = self.output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"✓ Configuration saved to {config_path}")

    def run_phase_a_only(self, n_seeds: int = 2) -> None:
        """Convenience method to run only Phase A."""
        self.initialize()
        self.generate_windows(n_windows=3)
        self.run_phase_a(n_seeds=n_seeds)
        self.consolidate_results()

    def run_phase_b_only(self, n_seeds: int = 3) -> None:
        """Convenience method to run only Phase B."""
        self.initialize()
        self.generate_windows(n_windows=10)
        self.run_phase_b(n_seeds=n_seeds)
        self.consolidate_results()

    def run_full(self, n_seeds_a: int = 2, n_seeds_b: int = 3) -> None:
        """Run full pipeline: Phase A + Phase B."""
        self.initialize()
        self.generate_windows(n_windows=10)
        self.run_phase_a(n_seeds=n_seeds_a)
        self.run_phase_b(n_seeds=n_seeds_b)
        self.consolidate_results()
