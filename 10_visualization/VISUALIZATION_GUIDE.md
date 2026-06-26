# STEP File Visualizer - Guide

## What This Tool Does

The **STEP Visualizer** helps you understand why a CAD generation is positive, negative, or review by:

1. **Displaying all 44 metrics** for each pair
2. **Highlighting the 10 key metrics** that matter most
3. **Providing a recommendation** based on metrics
4. **Explaining the decision logic** so you understand why

This tool focuses on **metrics visualization** (not 3D rendering) because:
- The metrics already tell you the quality of the geometry
- Chamfer distance, Hausdorff distance, etc. measure how far the surface is from the reference
- ICP fitness measures alignment quality
- Volume and surface area differences show shape preservation

## How to Use It

### Option 1: View a Specific Pair

```bash
cd "./cad42_platform"
python step_visualizer.py 0      # View pair #1
python step_visualizer.py 45     # View pair #46 
```

### Option 2: Interactive Browse

```bash
python step_visualizer.py
```

Then use commands:
- **`n`** → Next pair
- **`p`** → Previous pair
- **`j`** → Jump to pair number
- **`q`** → Quit

## What You'll See

### 🎯 SUMMARY METRICS (top 10)

These 10 metrics are the most important:

```
Overall Quality Score (0-100)
  └─ Higher is better, >70 is good
  
Valid CAD Rate (0-1)
  └─ Percentage of geometry that's valid CAD, should be >0.95
  
Watertight Manifold (0-1)
  └─ Is the mesh a closed surface? 1.0 = yes
  
Component Count Match (0-1)
  └─ Do generated features match reference? 1.0 = perfect match
  
Chamfer Distance (mm) ⭐ KEY
  └─ Average surface distance from reference
  └─ Lower is better: <1mm=excellent, <3mm=good, >5mm=bad
  
Hausdorff 95% (mm) ⭐ KEY
  └─ 95th percentile of surface distance
  └─ Lower is better: <2mm=excellent, <3mm=good, >6mm=bad
  
Hausdorff 99% (mm)
  └─ 99th percentile (worst outlier)
  └─ Lower is better
  
Volume Difference (%)
  └─ How much volume differs: <1%=excellent, <5%=good, >10%=bad
  
ICP Fitness (0-1) ⭐ KEY
  └─ Alignment quality: >0.95=excellent, >0.85=good
  
Bounding Box Error (mm)
  └─ How far is the overall size? Lower is better
```

### 📋 ALL 44 METRICS

All metrics listed (geometry, topology, alignment, etc.)

### 💡 DECISION GUIDE

Shows your decision logic:
- ✅ = metric is good
- ⚠️  = metric is borderline
- ❌ = metric is poor

Final recommendation: **POSITIVE** / **NEGATIVE** / **REVIEW**

## Decision Rules (Quick Reference)

### ✅ POSITIVE
- Quality Score > 70 AND
- Valid CAD Rate > 90% AND  
- ICP Fitness > 0.94

→ This CAD is high-quality, use it

### ❌ NEGATIVE
- Quality Score < 45 OR
- Valid CAD Rate < 70% OR
- Chamfer Distance > 5mm

→ This CAD has serious errors, skip it

### ⚠️ REVIEW
- Everything else (metric values in between)

→ This is borderline, needs human judgment

## Tips

### What the Key Metrics Mean

| Metric | What It Measures | Good Range | Why It Matters |
|--------|------------------|------------|----------------|
| Chamfer Distance | Avg surface error | <1.5mm | Tells if geometry is close to reference |
| Hausdorff 95% | Max common surface error | <2.5mm | 95% of surface within this error |
| Volume Diff | Shape preservation | <1% | Did the model keep its size/volume? |
| ICP Fitness | Alignment quality | >0.94 | How well do reference & generated align? |
| Valid CAD Rate | Geometry validity | >0.95 | Is all the CAD geometry valid/usable? |

### How to Read the Display

```
🎯 SUMMARY METRICS (Use these to decide):
─────────────────────────────────────────────────────────────────────────
  Overall Quality Score              76.50  ✅ GOOD
  Valid CAD Rate                      0.98  ✅ EXCELLENT
  Watertight (1.0=yes)                1.00  ✅ EXCELLENT
  Component Count Match               1.00  ✅ EXCELLENT
  Chamfer Distance (surface error)    0.95  ✅ EXCELLENT
  Hausdorff 95% (max surface error)   1.72  ✅ GOOD
  Hausdorff 99% (extreme error)       2.09  ✅ GOOD
  Volume Difference (%)               0.00  ✅ EXCELLENT
  ICP Fitness (alignment)             0.99  ✅ EXCELLENT
  Bounding Box Error                  0.11  ✅ EXCELLENT

→ RECOMMENDATION: POSITIVE (Likely high-quality)
```

This pair has most metrics in the GOOD/EXCELLENT range → **Label as POSITIVE** ✅

### Another Example: Mixed Metrics

```
🎯 SUMMARY METRICS:
─────────────────────────────────────────────────────────────────────────
  Overall Quality Score              55.20  ⚠️  OK
  Valid CAD Rate                      0.78  ⚠️  OK
  Watertight (1.0=yes)                0.85  ⚠️  OK
  Component Count Match               0.60  ⚠️  OK
  Chamfer Distance (surface error)    2.45  ⚠️  OK
  ...

→ RECOMMENDATION: REVIEW (Borderline - human judgment needed)
```

This pair has mixed metrics → **Label as REVIEW** ⚠️

### Another Example: Clear Errors

```
🎯 SUMMARY METRICS:
─────────────────────────────────────────────────────────────────────────
  Overall Quality Score              32.10  ❌ BAD
  Valid CAD Rate                      0.55  ❌ BAD
  Chamfer Distance (surface error)    8.90  ❌ BAD
  Hausdorff 95% (max surface error)  12.50  ❌ BAD
  Volume Difference (%)              15.30% ❌ BAD

→ RECOMMENDATION: NEGATIVE (Likely poor-quality)
```

This pair has most metrics BAD → **Label as NEGATIVE** ❌

## Workflow

### Combined with Labeling Helper

1. **Use visualizer first** to understand the metrics for a pair
   ```bash
   python step_visualizer.py 0
   ```

2. **Then use labeling helper** to label it
   ```bash
   python labeling_helper.py ".stepscore_harness_runs/final73_anthropic_run_02/labeled_pairs_for_review.xlsx"
   ```

3. The labeling helper will show the same pair and ask for your label

### Benefits of This Two-Tool Approach

- **Visualizer**: Deep dive into metrics for understanding
- **Labeling Helper**: Fast interactive labeling with auto-recommendations

## Troubleshooting

### "Excel file not found"
Make sure you're in the correct directory:
```bash
cd "./cad42_platform"
```

### "Invalid pair index"
Valid indices are 0-145 (for 146 pairs)
- `python step_visualizer.py 0` = Pair #1
- `python step_visualizer.py 145` = Pair #146

### Want to see specific metrics?
The tool shows all 44 metrics. Look for the ones you care about in the detailed section.

---

**Ready?** Start visualizing pairs with:
```bash
python step_visualizer.py
```
