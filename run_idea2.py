#!/usr/bin/env python3
"""
Idea2 main execution script: Feature Windows + ILP with Global Sampling

Usage:
    python run_idea2.py --phase a      # Phase A only (fast)
    python run_idea2.py --phase b      # Phase B only (main benchmark)
    python run_idea2.py --phase full   # Full pipeline
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.idea2.pipeline import Idea2Pipeline

LOG_DIR = Path(__file__).parent / "logs" / "idea2"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'idea2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Idea2: Feature Windows + ILP Baseline"
    )

    parser.add_argument(
        '--phase',
        choices=['a', 'b', 'full'],
        default='a',
        help='Phase to run: a (proof of viability), b (benchmark), full (both)'
    )

    parser.add_argument(
        '--seeds-a',
        type=int,
        default=2,
        help='Number of seeds for Phase A (default: 2)'
    )

    parser.add_argument(
        '--seeds-b',
        type=int,
        default=3,
        help='Number of seeds for Phase B (default: 3)'
    )

    parser.add_argument(
        '--window-size',
        type=int,
        default=30,
        help='Feature window size (default: 30)'
    )

    parser.add_argument(
        '--timeout',
        type=int,
        default=1800,
        help='Max ILP timeout per run in seconds (default: 1800)'
    )

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("IDEA 2: FEATURE WINDOWS + ILP WITH GLOBAL SAMPLING")
    logger.info("="*80)
    logger.info(f"Phase: {args.phase}")
    logger.info(f"Seeds (A): {args.seeds_a}, Seeds (B): {args.seeds_b}")
    logger.info(f"Window size: {args.window_size}")
    logger.info(f"Timeout: {args.timeout}s")
    logger.info("="*80)

    # Create pipeline
    pipeline = Idea2Pipeline(
        features_path="./reports/extracted_features.parquet",
        labels_path="./data/training_set.csv",
        rankings_path="./reports/feature_analysis/feature_rankings_all.parquet",
        output_dir="./reports/idea2",
        padtai_dir="./PADTAI",
        max_timeout=args.timeout,
        window_size=args.window_size,
    )

    try:
        if args.phase == 'a':
            pipeline.run_phase_a_only(n_seeds=args.seeds_a)
        elif args.phase == 'b':
            pipeline.run_phase_b_only(n_seeds=args.seeds_b)
        elif args.phase == 'full':
            pipeline.run_full(n_seeds_a=args.seeds_a, n_seeds_b=args.seeds_b)

        logger.info("\n" + "="*80)
        logger.info("✓ IDEA2 PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"✗ Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
