"""
ILP Runner: executes PADTAI on feature windows with logging and metric capture.
"""

import pandas as pd
import subprocess
import tempfile
import json
import logging
import time
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ILPRunResult:
    """Result of a single ILP run."""
    window_id: int
    sample_size: int
    seed: int
    n_features: int
    feature_names: List[str]

    # Execution metrics
    elapsed_time: float
    solver_time: float
    status: str  # "success", "timeout", "error", "no_solution"

    # Rules and coverage
    n_rules: int
    rules: List[str]

    # Predictive metrics
    train_accuracy: float = None
    train_precision: float = None
    train_recall: float = None
    train_f1: float = None

    val_accuracy: float = None
    val_precision: float = None
    val_recall: float = None
    val_f1: float = None

    # Metadata
    dataset_path: str = None
    output_path: str = None
    error_message: str = None


class ILPRunner:
    """Runner for PADTAI ILP inference on feature windows."""

    def __init__(
        self,
        padtai_dir: str = "./PADTAI",
        max_timeout: int = 600,  # 10 minutes
        solver: str = "nuwls",
    ):
        """
        Initialize ILP runner.

        Args:
            padtai_dir: Path to PADTAI installation
            max_timeout: Max timeout per run (seconds)
            solver: Solver to use (nuwls or rc2)
        """
        self.padtai_dir = Path(padtai_dir)
        self.max_timeout = max_timeout
        self.solver = solver

        if not self.padtai_dir.exists():
            raise FileNotFoundError(f"PADTAI directory not found: {self.padtai_dir}")

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features: List[str],
        window_id: int,
        sample_size: int,
        seed: int,
        output_dir: Path,
        label_column: str = "label",
    ) -> ILPRunResult:
        """
        Execute ILP on a feature window with a given sample.

        Args:
            X: Feature matrix (full dataset)
            y: Labels
            features: Selected features for this window
            window_id: Window index
            sample_size: Number of samples used
            seed: Random seed
            output_dir: Directory to save outputs
            label_column: Name of label column in output CSV

        Returns:
            ILPRunResult with execution metrics
        """
        logger.info(
            f"\n{'='*60}"
        )
        logger.info(
            f"Running ILP | Window {window_id} | Size {sample_size} | Seed {seed}"
        )
        logger.info(f"Features: {features}")
        logger.info(f"{'='*60}")

        start_time = time.time()
        result = ILPRunResult(
            window_id=window_id,
            sample_size=sample_size,
            seed=seed,
            n_features=len(features),
            feature_names=features,
            elapsed_time=0,
            solver_time=0,
            status="pending",
            n_rules=0,
            rules=[],
        )

        try:
            # Create temporary CSV for PADTAI
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.csv', delete=False
            ) as f:
                temp_path = f.name
                dataset_path = Path(temp_path)

            logger.info(f"Creating dataset at {dataset_path}...")
            df_subset = X[features].copy()
            df_subset[label_column] = y.values
            df_subset.to_csv(dataset_path, index=False)

            result.dataset_path = str(dataset_path)
            result.output_path = str(output_dir)

            logger.info(f"✓ Dataset created: {len(df_subset)} rows × {len(features)} cols")

            # Execute PADTAI
            logger.info(f"Executing PADTAI (timeout={self.max_timeout}s)...")
            padtai_output = self._run_padtai(
                dataset_path=str(dataset_path),
                output_dir=str(output_dir),
                timeout=self.max_timeout,
            )

            # Parse output
            rules = self._extract_rules(padtai_output)
            solver_time = self._extract_time(padtai_output)

            result.n_rules = len(rules)
            result.rules = rules
            result.solver_time = solver_time
            result.status = "success" if len(rules) > 0 else "no_solution"

            logger.info(f"✓ Found {len(rules)} rules")
            if len(rules) > 0:
                logger.info(f"  Top rule: {rules[0][:80]}")

        except subprocess.TimeoutExpired:
            result.status = "timeout"
            result.error_message = f"PADTAI timeout ({self.max_timeout}s)"
            logger.warning(f"✗ Timeout: {result.error_message}")
        except Exception as e:
            result.status = "error"
            result.error_message = str(e)
            logger.error(f"✗ Error: {e}")

        finally:
            # Cleanup temp file
            try:
                if 'dataset_path' in locals():
                    Path(dataset_path).unlink()
            except:
                pass

        result.elapsed_time = time.time() - start_time
        logger.info(f"Total time: {result.elapsed_time:.1f}s | Status: {result.status}")

        return result

    def _run_padtai(
        self,
        dataset_path: str,
        output_dir: str,
        timeout: int = 1800,
    ) -> str:
        """
        Execute PADTAI subprocess.

        Args:
            dataset_path: Path to CSV dataset
            output_dir: Output directory for PADTAI
            timeout: Timeout in seconds

        Returns:
            STDOUT output from PADTAI
        """
        padtai_script = self.padtai_dir / "padtai.py"
        if not padtai_script.exists():
            raise FileNotFoundError(f"PADTAI script not found: {padtai_script}")

        cmd = [
            "python",
            str(padtai_script),
            "--table_path", dataset_path,
            "--out_path", output_dir,
            "--solver", self.solver,
            "--timeout", str(timeout),
        ]

        logger.debug(f"Command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=str(self.padtai_dir),
            capture_output=True,
            text=True,
            timeout=timeout + 10,  # Add 10s buffer
        )

        if result.returncode != 0:
            logger.warning(f"PADTAI returned code {result.returncode}")
            if result.stderr:
                logger.warning(f"STDERR: {result.stderr[:500]}")

        return result.stdout + result.stderr

    def _extract_rules(self, output: str) -> List[str]:
        """
        Extract rules from PADTAI output.

        Args:
            output: STDOUT/STDERR from PADTAI

        Returns:
            List of rule strings
        """
        rules = []

        # Pattern: "Rule: head :- body" or similar
        patterns = [
            r"Rule:\s*(.+?)\s*:-\s*(.+?)(?:\n|$)",
            r"^([a-z_][a-z0-9_]*\([^)]*\))\s*:-\s*(.+?)$",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, output, re.MULTILINE | re.IGNORECASE):
                rule = match.group(0).strip()
                if rule and rule not in rules:
                    rules.append(rule)

        return rules

    def _extract_time(self, output: str) -> float:
        """Extract solver time from output."""
        patterns = [
            r"time[d]?:\s*([\d.]+)",
            r"elapsed[d]?:\s*([\d.]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        return 0.0
