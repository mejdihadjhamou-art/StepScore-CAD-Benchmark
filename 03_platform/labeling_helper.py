#!/usr/bin/env python3
"""
StepScore Labeling Helper - Interactive tool to review pairs and assign labels

Usage:
    python labeling_helper.py <path_to_excel_file>

    Example:
    python labeling_helper.py ".stepscore_harness_runs/final73_anthropic_run_02/labeled_pairs_for_review.xlsx"
"""

import pandas as pd
import json
import sys
from pathlib import Path
from typing import Tuple, Optional
import os

class LabelingHelper:
    """Interactive tool for reviewing and labeling CAD generation pairs"""

    # Decision thresholds
    QUALITY_SCORE_POSITIVE = 70        # quality_score > 70 = likely positive
    QUALITY_SCORE_REVIEW = 40          # 40 <= quality_score <= 70 = review
    QUALITY_SCORE_NEGATIVE = 40        # quality_score < 40 = likely negative

    # Key metrics to focus on (in priority order)
    KEY_METRICS = [
        'quality_score_0_100',
        'valid_cad_rate',
        'watertight_manifold_pass',
        'component_count_match',
        'chamfer_distance_mm',
        'hausdorff_95p_mm',
        'volume_diff_percent',
        'alignment_quality_icp_fitness',
    ]

    # Metric quality thresholds (used to compute recommendation)
    METRIC_THRESHOLDS = {
        'quality_score_0_100': {'good': 70, 'ok': 40},
        'valid_cad_rate': {'good': 0.95, 'ok': 0.7},
        'watertight_manifold_pass': {'good': 1.0, 'ok': 0.8},
        'component_count_match': {'good': 1.0, 'ok': 0.8},
        'chamfer_distance_mm': {'good': 2.0, 'ok': 5.0},
        'hausdorff_95p_mm': {'good': 3.0, 'ok': 6.0},
        'hausdorff_99p_mm': {'good': 5.0, 'ok': 10.0},
        'volume_diff_percent': {'good': 1.0, 'ok': 5.0},
        'alignment_quality_icp_fitness': {'good': 0.95, 'ok': 0.80},
        'bbox_error_max_mm': {'good': 1.0, 'ok': 3.0},
        'surface_area_diff_percent': {'good': 2.0, 'ok': 5.0},
    }

    def __init__(self, excel_path: str):
        self.excel_path = Path(excel_path)
        if not self.excel_path.exists():
            print(f"❌ Error: File not found: {self.excel_path}")
            sys.exit(1)

        print(f"📂 Loading: {self.excel_path}")
        self.df = pd.read_excel(self.excel_path)
        self.total_pairs = len(self.df)

        # Count existing labels
        if 'label' in self.df.columns:
            self.labeled_count = self.df['label'].notna().sum()
        else:
            print("❌ Error: 'label' column not found in Excel file")
            sys.exit(1)

        print(f"✅ Loaded {self.total_pairs} pairs ({self.labeled_count} already labeled)")
        print()

    def get_next_unlabeled(self) -> Optional[int]:
        """Get index of next unlabeled pair"""
        unlabeled = self.df[self.df['label'].isna()]
        if len(unlabeled) > 0:
            return unlabeled.index[0]
        return None

    def compute_recommendation(self, row) -> Tuple[str, int]:
        """
        Compute a recommendation label based on metrics.
        Returns: (recommendation, confidence_0_to_10)
        """
        scores = {'positive': 0, 'review': 0, 'negative': 0}
        total_checks = 0

        # Check each key metric
        for metric in self.KEY_METRICS:
            if metric not in row or pd.isna(row[metric]):
                continue

            value = row[metric]
            total_checks += 1

            if metric not in self.METRIC_THRESHOLDS:
                continue

            thresholds = self.METRIC_THRESHOLDS[metric]

            # Metrics where lower is better
            if metric in ['chamfer_distance_mm', 'hausdorff_95p_mm', 'hausdorff_99p_mm',
                         'volume_diff_percent', 'bbox_error_max_mm', 'surface_area_diff_percent']:
                if value <= thresholds['good']:
                    scores['positive'] += 2
                elif value <= thresholds['ok']:
                    scores['review'] += 1
                else:
                    scores['negative'] += 2

            # Metrics where higher is better
            else:
                if value >= thresholds['good']:
                    scores['positive'] += 2
                elif value >= thresholds['ok']:
                    scores['review'] += 1
                else:
                    scores['negative'] += 2

        # Compute recommendation
        if total_checks == 0:
            return 'review', 0

        if scores['positive'] > scores['negative'] * 2:
            confidence = min(10, int(scores['positive'] / total_checks))
            return 'positive', confidence
        elif scores['negative'] > scores['positive'] * 2:
            confidence = min(10, int(scores['negative'] / total_checks))
            return 'negative', confidence
        else:
            confidence = min(10, int(scores['review'] / total_checks))
            return 'review', confidence

    def display_pair(self, idx: int):
        """Display a pair with all metrics and recommendation"""
        if idx >= len(self.df):
            print(f"❌ Invalid index: {idx}")
            return

        row = self.df.iloc[idx]

        print("\n" + "="*70)
        print(f"PAIR #{idx + 1} of {self.total_pairs}")
        print("="*70)

        # Display metadata
        print("\n📋 METADATA:")
        print(f"  Task: {row.get('task_id', 'N/A')}")
        print(f"  Model: {row.get('model_name', 'N/A')}")
        print(f"  Prompt Level: {row.get('prompt_level', 'N/A')}")

        # Get recommendation
        recommendation, confidence = self.compute_recommendation(row)
        confidence_bars = "█" * confidence + "░" * (10 - confidence)
        print(f"\n💡 RECOMMENDATION: {recommendation.upper()} [{confidence_bars}] ({confidence}/10)")

        # Display key metrics
        print("\n🔍 KEY METRICS:")
        print("-" * 70)

        for metric in self.KEY_METRICS:
            if metric not in row or pd.isna(row[metric]):
                continue

            value = row[metric]

            # Format value based on type
            if isinstance(value, float):
                if metric.endswith('_percent') or metric == 'valid_cad_rate':
                    formatted = f"{value:.2f}"
                elif metric.endswith('_mm'):
                    formatted = f"{value:.3f}"
                else:
                    formatted = f"{value:.4f}"
            else:
                formatted = str(value)

            # Get quality assessment
            if metric in self.METRIC_THRESHOLDS:
                thresholds = self.METRIC_THRESHOLDS[metric]
                is_lower_better = metric in ['chamfer_distance_mm', 'hausdorff_95p_mm',
                                             'hausdorff_99p_mm', 'volume_diff_percent',
                                             'bbox_error_max_mm', 'surface_area_diff_percent']

                if is_lower_better:
                    if value <= thresholds['good']:
                        status = "✅ GOOD"
                    elif value <= thresholds['ok']:
                        status = "⚠️  OK"
                    else:
                        status = "❌ BAD"
                else:
                    if value >= thresholds['good']:
                        status = "✅ GOOD"
                    elif value >= thresholds['ok']:
                        status = "⚠️  OK"
                    else:
                        status = "❌ BAD"
            else:
                status = ""

            print(f"  {metric:35s} = {formatted:>10s}  {status}")

        # Display all other metrics (non-key)
        print("\n📊 ALL OTHER METRICS:")
        print("-" * 70)
        other_metrics = [col for col in self.df.columns
                        if col not in self.KEY_METRICS + ['label', 'task_id', 'model_name', 'prompt_level']]

        for i, metric in enumerate(sorted(other_metrics)):
            if metric not in row or pd.isna(row[metric]):
                continue

            value = row[metric]
            if isinstance(value, float):
                formatted = f"{value:.4f}"
            else:
                formatted = str(value)

            print(f"  {metric:35s} = {formatted}")
            if (i + 1) % 3 == 0:
                print()

        # Display decision guide
        print("\n📝 DECISION GUIDE:")
        print("-" * 70)
        print("  POSITIVE:  This CAD output is high-quality (matches reference well)")
        print("  REVIEW:    This output is borderline - human judgment needed")
        print("  NEGATIVE:  This CAD output is poor-quality (significant errors)")

        # Current label status
        current_label = row.get('label', '')
        if pd.isna(current_label) or current_label == '':
            print(f"\n  Current Label: [UNLABELED]")
        else:
            print(f"\n  Current Label: {current_label.upper()}")

    def interactive_mode(self):
        """Run interactive labeling session"""
        while True:
            next_idx = self.get_next_unlabeled()

            if next_idx is None:
                print("\n" + "="*70)
                print("🎉 ALL PAIRS LABELED!")
                print("="*70)
                print(f"✅ Total labeled: {self.labeled_count} / {self.total_pairs}")
                self.save_file()
                break

            self.display_pair(next_idx)

            # Get user input
            while True:
                user_input = input("\n👤 Enter label (positive/negative/review) or command (s=save, q=quit): ").strip().lower()

                if user_input == 'q':
                    self.save_file()
                    print("\n👋 Quitting. Progress saved.")
                    return

                elif user_input == 's':
                    self.save_file()
                    print("\n💾 Progress saved!")
                    continue

                elif user_input in ['positive', 'negative', 'review', 'p', 'n', 'r']:
                    # Map shorthand to full names
                    label_map = {'p': 'positive', 'n': 'negative', 'r': 'review'}
                    label = label_map.get(user_input, user_input)

                    self.df.at[next_idx, 'label'] = label
                    self.labeled_count += 1
                    print(f"\n✅ Labeled as: {label.upper()}")
                    break

                else:
                    print("❌ Invalid input. Use: positive/negative/review (or p/n/r shorthand)")

    def save_file(self):
        """Save the Excel file with labels"""
        self.df.to_excel(self.excel_path, index=False)
        print(f"💾 Saved to: {self.excel_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python labeling_helper.py <path_to_excel_file>")
        print("\nExample:")
        print('  python labeling_helper.py ".stepscore_harness_runs/final73_anthropic_run_02/labeled_pairs_for_review.xlsx"')
        sys.exit(1)

    excel_path = sys.argv[1]
    helper = LabelingHelper(excel_path)
    helper.interactive_mode()


if __name__ == '__main__':
    main()
