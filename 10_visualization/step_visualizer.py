#!/usr/bin/env python3
"""
StepScore STEP File Visualizer - Display metrics and guide labeling decisions

Usage:
    python step_visualizer.py <pair_index>

    Example:
    python step_visualizer.py 0        # View pair #1
    python step_visualizer.py 45       # View pair #46
"""

import sys
import os
from pathlib import Path
import json
import pandas as pd

class STEPVisualizer:
    """Help visualize and understand CAD pair quality through metrics"""

    def __init__(self, excel_path: str):
        self.excel_path = Path(excel_path)

        if not self.excel_path.exists():
            print(f"❌ Excel file not found: {self.excel_path}")
            sys.exit(1)

        # Load Excel
        self.df = pd.read_excel(self.excel_path)
        print(f"✅ Loaded {len(self.df)} pairs")

    def display_pair(self, pair_index: int):
        """Display detailed metrics for a pair"""

        if pair_index >= len(self.df) or pair_index < 0:
            print(f"❌ Invalid pair index: {pair_index} (valid: 0-{len(self.df)-1})")
            return False

        row = self.df.iloc[pair_index]
        pair_id = row.get('pair_id', f'pair_{pair_index}')

        print("\n" + "="*90)
        print(f"📊 PAIR #{pair_index + 1} of {len(self.df)}")
        print(f"   ID: {pair_id}")
        print("="*90)

        # SUMMARY METRICS (most important for labeling decision)
        print("\n🎯 SUMMARY METRICS (Use these to decide):")
        print("-" * 90)

        summary_metrics = [
            ('quality_score_0_100', 'Overall Quality Score', 0, 100, 'higher'),
            ('valid_cad_rate', 'Valid CAD Rate', 0, 1, 'higher'),
            ('watertight_manifold_pass', 'Watertight (1.0=yes)', 0, 1, 'higher'),
            ('component_count_match', 'Component Count Match', 0, 1, 'higher'),
            ('chamfer_distance_mm', 'Chamfer Distance (surface error)', 0, 10, 'lower'),
            ('hausdorff_95p_mm', 'Hausdorff 95% (max surface error)', 0, 10, 'lower'),
            ('hausdorff_99p_mm', 'Hausdorff 99% (extreme error)', 0, 15, 'lower'),
            ('volume_diff_percent', 'Volume Difference (%)', 0, 10, 'lower'),
            ('alignment_quality_icp_fitness', 'ICP Fitness (alignment)', 0, 1, 'higher'),
            ('bbox_error_max_mm', 'Bounding Box Error', 0, 5, 'lower'),
        ]

        for metric_name, display_name, _, _, direction in summary_metrics:
            if metric_name not in row:
                continue

            value = row[metric_name]
            if pd.isna(value):
                continue

            # Format the value
            if isinstance(value, float):
                if 'percent' in metric_name or 'rate' in metric_name:
                    if metric_name.endswith('_rate'):
                        value_str = f"{value:.2f}"
                    else:
                        value_str = f"{value:.2f}%"
                elif 'mm' in metric_name or 'distance' in metric_name:
                    value_str = f"{value:.3f}mm"
                else:
                    value_str = f"{value:.4f}"
            else:
                value_str = str(value)

            # Determine quality assessment
            if metric_name == 'quality_score_0_100':
                if value > 75:
                    status = "✅ GOOD"
                elif value > 50:
                    status = "⚠️  OK"
                else:
                    status = "❌ BAD"
            elif metric_name in ['valid_cad_rate', 'watertight_manifold_pass', 'component_count_match', 'alignment_quality_icp_fitness']:
                if value >= 0.95:
                    status = "✅ EXCELLENT"
                elif value >= 0.8:
                    status = "✅ GOOD"
                elif value >= 0.7:
                    status = "⚠️  OK"
                else:
                    status = "❌ BAD"
            elif 'distance' in metric_name or 'error' in metric_name or 'diff' in metric_name:
                if value <= 1:
                    status = "✅ EXCELLENT"
                elif value <= 3:
                    status = "✅ GOOD"
                elif value <= 5:
                    status = "⚠️  OK"
                else:
                    status = "❌ BAD"
            else:
                status = ""

            print(f"  {display_name:40s} {value_str:>10s}  {status}")

        # DETAILED METRICS
        print("\n📋 ALL 44 METRICS (detailed):")
        print("-" * 90)

        all_metrics = [col for col in self.df.columns
                      if col not in ['label', 'pair_id', 'task_id', 'model_name', 'prompt_level']]

        for i, metric in enumerate(sorted(all_metrics)):
            if metric not in row:
                continue

            value = row[metric]
            if pd.isna(value):
                continue

            if isinstance(value, float):
                if abs(value) < 0.0001:
                    value_str = f"{value:.2e}"
                elif value > 100:
                    value_str = f"{value:.1f}"
                else:
                    value_str = f"{value:.4f}"
            else:
                value_str = str(value)

            print(f"  {metric:45s} = {value_str}")

        # DECISION GUIDE
        print("\n💡 LABELING DECISION GUIDE:")
        print("-" * 90)

        quality = row.get('quality_score_0_100', 0)
        valid_cad = row.get('valid_cad_rate', 0)
        watertight = row.get('watertight_manifold_pass', 0)
        chamfer = row.get('chamfer_distance_mm', float('inf'))
        icp = row.get('alignment_quality_icp_fitness', 0)

        print("\n  Decision Logic:")
        print(f"    Quality Score: {quality:.1f}/100", end="")
        if quality > 70:
            print("  ✅ (Good)")
        elif quality > 50:
            print("  ⚠️  (Medium)")
        else:
            print("  ❌ (Poor)")

        print(f"    Valid CAD: {valid_cad:.1%}", end="")
        if valid_cad > 0.95:
            print("  ✅")
        elif valid_cad > 0.7:
            print("  ⚠️")
        else:
            print("  ❌")

        print(f"    Watertight: {watertight:.1f}", end="")
        if watertight >= 0.95:
            print("  ✅")
        elif watertight >= 0.8:
            print("  ⚠️")
        else:
            print("  ❌")

        print(f"    Surface Error (Chamfer): {chamfer:.3f}mm", end="")
        if chamfer < 2:
            print("  ✅")
        elif chamfer < 5:
            print("  ⚠️")
        else:
            print("  ❌")

        print(f"    Alignment (ICP): {icp:.4f}", end="")
        if icp > 0.95:
            print("  ✅")
        elif icp > 0.8:
            print("  ⚠️")
        else:
            print("  ❌")

        # Final recommendation
        print("\n  RECOMMENDATION:")
        if quality > 70 and valid_cad > 0.9 and icp > 0.94:
            rec = "➡️  POSITIVE (Likely high-quality)"
            print(f"    {rec}")
        elif quality < 45 or valid_cad < 0.7 or chamfer > 5:
            rec = "➡️  NEGATIVE (Likely poor-quality)"
            print(f"    {rec}")
        else:
            rec = "➡️  REVIEW (Borderline - human judgment needed)"
            print(f"    {rec}")

        # Current label
        print("\n  Current Label Status:")
        current_label = row.get('label', '')
        if pd.isna(current_label) or current_label == '':
            print(f"    [NOT YET LABELED]")
        else:
            print(f"    {current_label.upper()}")

        print("\n" + "="*90)
        return True

    def interactive_browse(self):
        """Interactive browsing through pairs"""
        current_idx = 0

        while True:
            if not self.display_pair(current_idx):
                break

            cmd = input("\n👤 Commands: (n)ext, (p)rev, (j)ump, (q)uit: ").strip().lower()

            if cmd == 'n':
                current_idx = min(current_idx + 1, len(self.df) - 1)
            elif cmd == 'p':
                current_idx = max(current_idx - 1, 0)
            elif cmd == 'j':
                try:
                    num = int(input(f"Enter pair number (1-{len(self.df)}): ")) - 1
                    current_idx = max(0, min(num, len(self.df) - 1))
                except ValueError:
                    print("❌ Invalid input")
            elif cmd == 'q':
                print("👋 Exiting visualizer")
                break
            else:
                print("❌ Invalid command")


def main():
    excel_path = ".stepscore_harness_runs/final73_anthropic_run_02/labeled_pairs_for_review.xlsx"

    visualizer = STEPVisualizer(excel_path)

    if len(sys.argv) > 1:
        try:
            pair_idx = int(sys.argv[1])
            visualizer.display_pair(pair_idx)
        except ValueError:
            print(f"❌ Invalid index: {sys.argv[1]}")
            sys.exit(1)
    else:
        print("\n📊 STEP Metrics Visualizer - Interactive Mode")
        print("Review pair metrics to decide: positive / negative / review\n")
        visualizer.interactive_browse()


if __name__ == '__main__':
    main()
