"""
Feature window generator: creates consecutive non-overlapping blocks of features
from ranked IG list.
"""

from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class FeatureWindowGenerator:
    """Generate consecutive feature windows from ranked list."""

    def __init__(self, ranked_features: List[str], window_size: int = 30):
        """
        Initialize window generator.

        Args:
            ranked_features: Ordered list of feature names (by IG, descending)
            window_size: Size of each window (default: 30)
        """
        self.ranked_features = ranked_features
        self.window_size = window_size
        self.windows = []

    def generate_windows(self, n_windows: int = None) -> List[List[str]]:
        """
        Generate non-overlapping consecutive windows.

        Args:
            n_windows: Number of windows to generate (None = as many as possible)

        Returns:
            List of feature lists, each of size window_size
        """
        if n_windows is None:
            n_windows = len(self.ranked_features) // self.window_size

        windows = []
        for i in range(n_windows):
            start = i * self.window_size
            end = start + self.window_size
            if end <= len(self.ranked_features):
                window = self.ranked_features[start:end]
                windows.append(window)
                logger.info(
                    f"Window {i+1}: [{start+1}-{end}] "
                    f"(first: {window[0]}, last: {window[-1]})"
                )

        self.windows = windows
        logger.info(f"✓ Generated {len(windows)} windows of size {self.window_size}")
        return windows

    def get_window_by_index(self, idx: int) -> List[str]:
        """Get window by index."""
        if idx >= len(self.windows):
            raise IndexError(f"Window index {idx} out of range (max: {len(self.windows)-1})")
        return self.windows[idx]

    def get_window_features_range(self, idx: int) -> Tuple[int, int]:
        """
        Get the range (1-indexed) of this window in the full ranking.

        Returns: (start_pos, end_pos) 1-indexed
        """
        start = idx * self.window_size + 1
        end = start + self.window_size - 1
        return start, end

    def windows_summary(self) -> str:
        """Return summary of generated windows."""
        summary = f"Total windows: {len(self.windows)}\n"
        summary += f"Window size: {self.window_size}\n"
        summary += f"Total features covered: {len(self.windows) * self.window_size}\n"
        summary += f"Available features: {len(self.ranked_features)}\n"
        return summary

    def get_all_windows(self) -> List[List[str]]:
        """Return all generated windows."""
        return self.windows
