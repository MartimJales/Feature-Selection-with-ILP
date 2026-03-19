from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
	accuracy_score,
	balanced_accuracy_score,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


def parse_top_k(values: Iterable[str]) -> List[int]:
	return sorted({int(v) for v in values})


def parse_depths(values: Iterable[str]) -> List[int | None]:
	parsed: List[int | None] = []
	for v in values:
		if v.lower() == "none":
			parsed.append(None)
		else:
			parsed.append(int(v))
	# keep original order but remove duplicates
	unique: List[int | None] = []
	for v in parsed:
		if v not in unique:
			unique.append(v)
	return unique


def load_top_features(rankings_path: Path, top_k: int) -> List[str]:
	if rankings_path.suffix == ".parquet":
		rankings = pd.read_parquet(rankings_path)
	else:
		rankings = pd.read_csv(rankings_path)

	if "feature" not in rankings.columns:
		raise ValueError("Ranking file must contain a 'feature' column")

	return rankings.head(top_k)["feature"].astype(str).tolist()


def load_selected_dataset(
	features_path: Path,
	labels_path: Path,
	selected_features: List[str],
) -> tuple[pd.DataFrame, pd.Series]:
	# Read only needed columns from parquet when possible (faster/lower memory)
	if features_path.suffix == ".parquet":
		cols = ["file_hash"] + selected_features
		# Some ranking features may not exist in the extracted dataset
		available_cols = pd.read_parquet(features_path, columns=None).columns.tolist()
		kept = [c for c in cols if c in available_cols]
		if "file_hash" not in kept:
			raise ValueError("Features file must contain 'file_hash' column")
		X_df = pd.read_parquet(features_path, columns=kept)
	else:
		# CSV fallback
		X_df = pd.read_csv(features_path)
		if "file_hash" not in X_df.columns:
			raise ValueError("Features file must contain 'file_hash' column")
		kept = ["file_hash"] + [c for c in selected_features if c in X_df.columns]
		X_df = X_df[kept]

	labels_df = pd.read_csv(labels_path)[["sha256", "label"]]
	X_df["file_hash"] = X_df["file_hash"].str.lower().str.strip()
	labels_df["sha256"] = labels_df["sha256"].str.lower().str.strip()

	merged = X_df.merge(labels_df, left_on="file_hash", right_on="sha256", how="inner")
	merged = merged.drop(columns=["file_hash", "sha256"], errors="ignore")
	merged = merged.dropna(subset=["label"])

	y = merged["label"].astype(int)
	X = merged.drop(columns=["label"])
	return X, y


def evaluate_tree(X: pd.DataFrame, y: pd.Series, max_depth: int | None, seed: int) -> dict:
	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.2,
		random_state=seed,
		stratify=y,
	)

	clf = DecisionTreeClassifier(
		criterion="gini",
		max_depth=max_depth,
		random_state=seed,
		class_weight="balanced",
		min_samples_leaf=5,
	)
	clf.fit(X_train, y_train)

	y_pred = clf.predict(X_test)
	y_score = clf.predict_proba(X_test)[:, 1]

	return {
		"max_depth": str(max_depth),
		"n_train": len(X_train),
		"n_test": len(X_test),
		"accuracy": accuracy_score(y_test, y_pred),
		"balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
		"precision": precision_score(y_test, y_pred, zero_division=0),
		"recall": recall_score(y_test, y_pred, zero_division=0),
		"f1": f1_score(y_test, y_pred, zero_division=0),
		"roc_auc": roc_auc_score(y_test, y_score),
		"tree_depth": clf.get_depth(),
		"n_leaves": clf.get_n_leaves(),
		"model": clf,
		"feature_names": X.columns.tolist(),
	}


def export_feature_importance(
	clf: DecisionTreeClassifier,
	feature_names: List[str],
	output_file: Path,
	top_n: int = 30,
) -> None:
	imp = pd.DataFrame(
		{
			"feature": feature_names,
			"importance": clf.feature_importances_,
		}
	)
	imp = imp.sort_values("importance", ascending=False)
	imp = imp[imp["importance"] > 0].head(top_n)
	imp.to_csv(output_file, index=False)


def export_tree_png(
	clf: DecisionTreeClassifier,
	feature_names: List[str],
	output_file: Path,
	plot_max_depth: int | None = None,
) -> None:
	# Dynamic sizing to avoid leaf overlap as tree gets wider/deeper.
	full_depth = clf.get_depth()
	shown_depth = full_depth if plot_max_depth is None else min(plot_max_depth, full_depth)
	effective_leaves = min(clf.get_n_leaves(), 2 ** max(shown_depth, 1))

	fig_w = max(24, min(220, effective_leaves * 1.4))
	fig_h = max(12, min(80, (shown_depth + 1) * 2.8))
	font_size = 8 if effective_leaves <= 64 else 6

	fig, ax = plt.subplots(figsize=(fig_w, fig_h))
	plot_tree(
		clf,
		feature_names=feature_names,
		class_names=["goodware", "malware"],
		filled=True,
		rounded=True,
		fontsize=font_size,
		max_depth=plot_max_depth,
		ax=ax,
	)
	ax.set_title("Decision Tree (best model)", fontsize=14, fontweight="bold")
	fig.tight_layout()
	fig.savefig(output_file, dpi=250, bbox_inches="tight")
	plt.close(fig)


def export_tree_svg(
	clf: DecisionTreeClassifier,
	feature_names: List[str],
	output_file: Path,
	plot_max_depth: int | None = None,
) -> None:
	# Same dynamic layout as PNG, but vector format allows deep zoom/pan.
	full_depth = clf.get_depth()
	shown_depth = full_depth if plot_max_depth is None else min(plot_max_depth, full_depth)
	effective_leaves = min(clf.get_n_leaves(), 2 ** max(shown_depth, 1))

	fig_w = max(24, min(260, effective_leaves * 1.6))
	fig_h = max(12, min(100, (shown_depth + 1) * 3.0))
	font_size = 8 if effective_leaves <= 64 else 6

	fig, ax = plt.subplots(figsize=(fig_w, fig_h))
	plot_tree(
		clf,
		feature_names=feature_names,
		class_names=["goodware", "malware"],
		filled=True,
		rounded=True,
		fontsize=font_size,
		max_depth=plot_max_depth,
		ax=ax,
	)
	ax.set_title("Decision Tree (best model)", fontsize=14, fontweight="bold")
	fig.tight_layout()
	fig.savefig(output_file, format="svg", bbox_inches="tight")
	plt.close(fig)


def main() -> None:
	parser = argparse.ArgumentParser(description="Decision Tree comparison on top-K features")
	parser.add_argument("--features", default="./reports/extracted_features.parquet")
	parser.add_argument("--labels", default="./data/training_set.csv")
	parser.add_argument("--rankings", default="./reports/feature_analysis/feature_rankings_all.parquet")
	parser.add_argument("--top-k", nargs="+", default=["200", "500", "1000"])
	parser.add_argument("--depths", nargs="+", default=["5", "10", "20", "None"])
	parser.add_argument("--plot-max-depth", type=int, default=6, help="Max depth to display in tree PNG (use -1 for full tree)")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--output-dir", default="./reports/decision_tree")
	args = parser.parse_args()

	features_path = Path(args.features)
	labels_path = Path(args.labels)
	rankings_path = Path(args.rankings)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	top_k_values = parse_top_k(args.top_k)
	depths = parse_depths(args.depths)

	summary_rows = []

	for k in top_k_values:
		selected_features = load_top_features(rankings_path, k)
		X, y = load_selected_dataset(features_path, labels_path, selected_features)

		run_results = []
		for depth in depths:
			result = evaluate_tree(X, y, max_depth=depth, seed=args.seed)
			run_results.append(result)
			summary_rows.append(
				{
					"top_k": k,
					"max_depth": result["max_depth"],
					"n_train": result["n_train"],
					"n_test": result["n_test"],
					"accuracy": result["accuracy"],
					"balanced_accuracy": result["balanced_accuracy"],
					"precision": result["precision"],
					"recall": result["recall"],
					"f1": result["f1"],
					"roc_auc": result["roc_auc"],
					"tree_depth": result["tree_depth"],
					"n_leaves": result["n_leaves"],
					"n_features_used": X.shape[1],
				}
			)

		# Export top importances for best depth at this top-k (by F1)
		best = max(run_results, key=lambda r: r["f1"])
		best_depth_str = best["max_depth"].lower() if isinstance(best["max_depth"], str) else str(best["max_depth"])
		imp_file = output_dir / f"decision_tree_top{k}_best_depth_{best_depth_str}_feature_importance.csv"
		export_feature_importance(best["model"], best["feature_names"], imp_file, top_n=30)

		plot_depth = None if args.plot_max_depth < 0 else args.plot_max_depth
		tree_png = output_dir / f"decision_tree_top{k}_best_depth_{best_depth_str}.png"
		export_tree_png(best["model"], best["feature_names"], tree_png, plot_max_depth=plot_depth)
		tree_svg = output_dir / f"decision_tree_top{k}_best_depth_{best_depth_str}.svg"
		export_tree_svg(best["model"], best["feature_names"], tree_svg, plot_max_depth=plot_depth)

	summary_df = pd.DataFrame(summary_rows).sort_values(["top_k", "f1"], ascending=[True, False])
	summary_file = output_dir / "decision_tree_comparison_topk.csv"
	summary_df.to_csv(summary_file, index=False)

	print("\nDecision Tree comparison completed.")
	print(f"Summary: {summary_file}")
	print(f"Output directory: {output_dir}")
	print("\nTop results by top_k (best F1):")
	best_per_k = summary_df.groupby("top_k", as_index=False).first()
	print(best_per_k[["top_k", "max_depth", "f1", "balanced_accuracy", "roc_auc", "tree_depth", "n_leaves"]].to_string(index=False))


if __name__ == "__main__":
	main()
