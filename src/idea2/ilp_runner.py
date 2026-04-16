"""
ILP Runner: executes PADTAI on feature windows with logging and metric capture.
"""

import pandas as pd
import subprocess
import tempfile
import logging
import time
import re
import os
import json
import signal
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

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
    debug_output_path: str = None
    feature_mapping_path: str = None
    error_message: str = None


class ILPRunner:
    """Runner for PADTAI ILP inference on feature windows."""

    def __init__(
        self,
        padtai_dir: str = "./PADTAI",
        max_timeout: int = 600,  # 10 minutes
        solver: str = "nuwls",
        debug: bool = False,
        debug_output_dir: str | None = None,
    ):
        """
        Initialize ILP runner.

        Args:
            padtai_dir: Path to PADTAI installation
            max_timeout: Max timeout per run (seconds)
            solver: Solver to use (nuwls or rc2)
            debug: If True, save raw PADTAI output per run
            debug_output_dir: Directory for debug logs (used only when debug=True)
        """
        self.padtai_dir = Path(padtai_dir).resolve()
        self.max_timeout = max_timeout
        self.solver = solver
        self.debug = debug
        self.debug_output_dir = Path(debug_output_dir).resolve() if debug_output_dir else None

        if self.debug and self.debug_output_dir is not None:
            try:
                self.debug_output_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"✓ Debug output dir created: {self.debug_output_dir}")
            except Exception as e:
                logger.error(f"✗ Failed to create debug dir {self.debug_output_dir}: {e}")
                raise

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
            original_to_sanitized, sanitized_to_original = self._build_feature_name_mapping(features)
            df_subset = X[features].copy().rename(columns=original_to_sanitized)
            safe_label_column = "label"
            df_subset[safe_label_column] = y.values
            df_subset.to_csv(dataset_path, index=False)

            result.dataset_path = str(dataset_path)
            result.output_path = str(output_dir)

            logger.info(f"✓ Dataset created: {len(df_subset)} rows × {len(features)} cols")
            logger.info("✓ Applied feature sanitization with feature-index mapping")

            # Execute PADTAI
            logger.info(f"Executing PADTAI (timeout={self.max_timeout}s)...")
            padtai_output = self._run_padtai(
                dataset_path=str(dataset_path),
                sample_size=len(df_subset),
                timeout=self.max_timeout,
            )

            if self.debug and self.debug_output_dir is not None:
                debug_file = self._save_debug_output(
                    padtai_output=padtai_output,
                    window_id=window_id,
                    sample_size=sample_size,
                    seed=seed,
                    n_rows=len(df_subset),
                    n_features=len(features),
                    sanitized_to_original=sanitized_to_original,
                )
                result.debug_output_path = str(debug_file)
                result.feature_mapping_path = str(debug_file.with_suffix(".map.json"))

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

    def _save_debug_output(
        self,
        padtai_output: str,
        window_id: int,
        sample_size: int,
        seed: int,
        n_rows: int,
        n_features: int,
        sanitized_to_original: Dict[str, str],
    ) -> Path:
        """Persist raw PADTAI output for debugging."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = (
            f"ilp_w{window_id}_n{sample_size}_seed{seed}_{timestamp}.txt"
        )
        file_path = self.debug_output_dir / filename
        mapping_path = file_path.with_suffix(".map.json")

        logger.info(f"💾 Saving debug output to: {file_path}")
        logger.info(f"   Directory exists: {self.debug_output_dir.exists()}")
        logger.info(f"   Directory is writable: {os.access(self.debug_output_dir, os.W_OK)}")

        try:
            with open(mapping_path, "w", encoding="utf-8") as fmap:
                json.dump(
                    {
                        "window_id": window_id,
                        "sample_size": sample_size,
                        "seed": seed,
                        "sanitized_to_original": sanitized_to_original,
                    },
                    fmap,
                    ensure_ascii=False,
                    indent=2,
                )

            with open(file_path, "w", encoding="utf-8") as f:
                f.write("=== IDEA2 ILP DEBUG OUTPUT ===\n")
                f.write(f"window_id={window_id}\n")
                f.write(f"sample_size={sample_size}\n")
                f.write(f"seed={seed}\n")
                f.write(f"rows={n_rows}\n")
                f.write(f"n_features={n_features}\n")
                f.write(f"solver={self.solver}\n")
                f.write(f"timeout={self.max_timeout}\n")
                f.write(f"feature_mapping_path={mapping_path}\n")
                f.write("\n=== FEATURE MAPPING (sanitized -> original) ===\n")
                for sanitized, original in sanitized_to_original.items():
                    f.write(f"{sanitized} -> {original}\n")
                f.write("\n=== RAW PADTAI STDOUT/STDERR ===\n")
                f.write(padtai_output or "")

            logger.info(f"✓ Debug output saved successfully to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"✗ Failed to save debug output to {file_path}: {e}")
            raise

    def _run_padtai(
        self,
        dataset_path: str,
        sample_size: int,
        timeout: int = 1800,
    ) -> str:
        """
        Execute PADTAI subprocess.

        Args:
            dataset_path: Path to CSV dataset
            sample_size: Number of rows to use inside PADTAI parser
            timeout: Timeout in seconds

        Returns:
            STDOUT output from PADTAI
        """
        padtai_script = self.padtai_dir / "padtai.py"
        if not padtai_script.exists():
            raise FileNotFoundError(f"PADTAI script not found: {padtai_script}")

        cmd = [
            "python",
            str(padtai_script.resolve()),
            dataset_path,
            "--solver", self.solver,
            "--sample-size", str(sample_size),
            "--max-timeout", str(timeout),
            "--debug", "all" if self.debug else "none",
        ]

        logger.debug(f"Command: {' '.join(cmd)}")

        # Run PADTAI in its own process group so timeout can kill child processes too.
        proc = subprocess.Popen(
            cmd,
            cwd=str(self.padtai_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )

        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            logger.warning(f"PADTAI hard timeout reached ({timeout}s). Killing process tree...")
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception as kill_err:
                logger.warning(f"Failed to kill PADTAI process group cleanly: {kill_err}")

            # Ensure process is reaped and collect any partial output
            stdout, stderr = proc.communicate()
            partial_output = (stdout or "") + (stderr or "")
            raise subprocess.TimeoutExpired(
                cmd=cmd,
                timeout=timeout,
                output=partial_output,
                stderr=stderr,
            ) from exc

        class _Completed:
            def __init__(self, returncode: int, stdout: str, stderr: str):
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr

        result = _Completed(proc.returncode, stdout, stderr)

        if result.returncode != 0:
            logger.warning(f"PADTAI returned code {result.returncode}")
            if result.stderr:
                logger.warning(f"STDERR: {result.stderr[:500]}")

        return result.stdout + result.stderr

    def _build_feature_name_mapping(self, features: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Build stable feature mapping to Prolog-safe names.

        Uses feature index (f0, f1, ...) to avoid syntax issues in PADTAI/Prolog.
        Returns both directions for easier debugging and post-analysis.
        """
        original_to_sanitized = {name: f"f{i}" for i, name in enumerate(features)}
        sanitized_to_original = {sanitized: original for original, sanitized in original_to_sanitized.items()}
        return original_to_sanitized, sanitized_to_original

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
