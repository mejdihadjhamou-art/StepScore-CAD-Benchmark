# 🎨 3D CAD Viewer - Complete Guide

## What This Tool Does

The **3D CAD Viewer** is an interactive web application that lets you:

✅ **View 3D Models Side-by-Side**
- Reference model on the left
- Generated model (Claude's output) on the right
- Both fully interactive and rotatable

✅ **Understand Quality Visually**
- See if the generated shape matches the reference
- Spot missing features or errors immediately
- Compare overall geometry and dimensions

✅ **Make Labeling Decisions**
- Use visual + metric data together
- Click buttons to label: POSITIVE / NEGATIVE / REVIEW
- Auto-saves and advances to next pair

✅ **Navigate Efficiently**
- Jump to any pair #1-146
- Next/Previous buttons
- Keyboard shortcuts (← →)

---

## Prerequisites

```bash
pip install flask flask-cors pandas openpyxl
```

Check if installed:
```bash
python -c "import flask, flask_cors; print('✅ Ready')"
```

---

## How to Start

### Step 1: Open Terminal

```bash
cd "/Users/mejdi/Documents/New project/cad42_platform"
```

### Step 2: Run the Server

```bash
python 3d_viewer_server.py
```

You'll see:
```
======================================================================
🚀 Starting 3D CAD Viewer Server
======================================================================

📂 Harness directory: .stepscore_harness_runs/final73_anthropic_run_02
📊 Excel file: .stepscore_harness_runs/final73_anthropic_run_02/labeled_pairs_for_review.xlsx
✅ Loaded 146 pairs

🌐 Open your browser: http://localhost:5000
======================================================================
```

### Step 3: Open Your Browser

Go to: **http://localhost:5000**

You'll see the 3D viewer with:
- **Left side**: Reference model (what it should look like)
- **Right side**: Generated model (what Claude created)
- **Bottom**: Metrics and labeling buttons

---

## How to Use the 3D Viewer

### Navigation

```
Pair Input:     Type 1-146 and Enter to jump
← Prev Button:  Go to previous pair
Next → Button:  Go to next pair
← → Keys:       Arrow keys also work
```

### 3D Controls

**In either 3D viewer:**

```
Left Click + Drag:          Rotate the model
Scroll Wheel:               Zoom in/out
Ctrl + Left Click + Drag:   Pan around
```

Both models rotate together for easy comparison!

### Interpreting the Models

Look at the 3D shapes and ask yourself:

**✅ POSITIVE:**
- Generated shape matches reference very closely
- All features present (holes, pockets, fillets)
- No obvious errors or deformations
- Overall dimensions look correct

**❌ NEGATIVE:**
- Generated shape is very different from reference
- Missing features or completely wrong geometry
- Obvious deformations, overlaps, or breaks
- Doesn't look like the same part

**⚠️ REVIEW:**
- Mostly correct but with some minor issues
- Slightly off dimensions or smoothing
- Some small features missing or slightly wrong
- Not clearly good or clearly bad

---

## Example Workflow

### Pair #1 - POSITIVE Example

```
📋 Reference: A box with 4 corner holes
🤖 Generated: Looks identical to reference

🔍 Metrics:
  - Chamfer Distance: 0.946mm ✅
  - ICP Fitness: 0.9858 ✅  
  - Volume Diff: 0.0% ✅
  - Quality Score: 67.8 ⚠️

Visual Assessment: The shapes look identical
Metric Assessment: Most metrics excellent

Decision: This is borderline positive
→ Click ✅ POSITIVE (good enough despite medium quality_score)
```

### Pair #45 - NEGATIVE Example

```
📋 Reference: A bracket with specific geometry
🤖 Generated: Looks deformed and misshapen

🔍 Metrics:
  - Chamfer Distance: 8.5mm ❌
  - ICP Fitness: 0.62 ❌
  - Volume Diff: 25% ❌
  - Quality Score: 28 ❌

Visual Assessment: Shapes are quite different
Metric Assessment: Almost all metrics bad

Decision: This is clearly negative
→ Click ❌ NEGATIVE (major quality issues)
```

### Pair #73 - REVIEW Example

```
📋 Reference: A complex shaft with features
🤖 Generated: Similar shape but slightly different

🔍 Metrics:
  - Chamfer Distance: 2.8mm ⚠️
  - ICP Fitness: 0.85 ⚠️
  - Volume Diff: 3.2% ⚠️
  - Quality Score: 62 ⚠️

Visual Assessment: Similar shapes with minor differences
Metric Assessment: Mixed results - some good, some mediocre

Decision: This is borderline
→ Click ⚠️ REVIEW (not clearly good or bad)
```

---

## Tips for Efficient Labeling

### 1. **Use Your Eyes First**
The 3D visualization is the primary tool. If the shapes look obviously different, it's negative. If they look the same, it's positive.

### 2. **Then Check Metrics**
The metrics confirm what you see:
- **Chamfer Distance**: Surface error (< 1.5mm = good)
- **ICP Fitness**: Alignment (> 0.94 = good)
- **Volume Diff**: Shape preservation (< 1% = excellent)
- **Quality Score**: Overall score

### 3. **Rotate to Different Angles**
Don't just look from one angle. Rotate both models to see:
- Front view
- Side view
- Top view
- Isometric view

Drag and rotate to really compare!

### 4. **When in Doubt**
If you're unsure, click **REVIEW**. The threshold tuning process will figure out which metrics best separate good from bad.

### 5. **Be Consistent**
- If pair #5 is positive, and pair #50 looks similar, it should also be positive
- If pair #10 is negative with 20% volume error, pairs with 25% error should also be negative

---

## Keyboard Shortcuts

```
← Arrow Left    Previous pair
→ Arrow Right   Next pair
Enter           Confirm labeling decision
```

---

## Metrics Explained (For Reference)

| Metric | What It Measures | Good Range | Why It Matters |
|--------|------------------|------------|----------------|
| **Chamfer Distance** | Average surface distance | < 1.5mm | Tells if geometry matches |
| **ICP Fitness** | Alignment quality | > 0.94 | How well reference & generated align |
| **Hausdorff 95%** | 95th percentile error | < 2.5mm | Worst 5% of surface error |
| **Volume Diff** | Shape size preservation | < 1% | Did model keep correct size? |
| **Quality Score** | Overall rating (0-100) | > 70 | Composite quality metric |

---

## Troubleshooting

### "Cannot connect to localhost:5000"
1. Make sure the server is still running
2. Check the terminal where you ran `python 3d_viewer_server.py`
3. If crashed, restart the server

### "Error loading reference" or "Error loading generated"
1. The STL files might be missing or corrupted
2. Try navigating to a different pair
3. Check the server logs in terminal

### 3D Models Not Appearing
1. Refresh the browser (Ctrl+R or Cmd+R)
2. Restart the server and try again
3. Make sure Flask is installed: `pip install flask flask-cors`

### Want to Stop the Server
Press **Ctrl+C** in the terminal where you ran the server

### Want to Check Progress
Your labels are automatically saved to the Excel file as you label each pair. You can see progress in:
```
.stepscore_harness_runs/final73_anthropic_run_02/labeled_pairs_for_review.xlsx
```

---

## Complete Workflow

1. **Start server**: `python 3d_viewer_server.py`
2. **Open browser**: http://localhost:5000
3. **View pair #1**: 3D models load automatically
4. **Rotate and inspect**: Use mouse to rotate both models
5. **Check metrics**: Bottom panel shows key metrics
6. **Make decision**: Click ✅ POSITIVE, ❌ NEGATIVE, or ⚠️ REVIEW
7. **Auto-advance**: Next pair loads automatically
8. **Repeat**: Continue through all 146 pairs
9. **Done**: All labels saved to Excel

---

## Expected Time

- **First 10 pairs**: ~15-20 min (getting familiar)
- **Pairs 11-146**: ~45-60 min (1-3 pairs per minute)
- **Total**: ~60-90 minutes for all 146 pairs

---

## After Labeling

Once you finish labeling all 146 pairs:

1. Close the 3D viewer (press Ctrl+C in terminal)
2. Tell the AI assistant: "I've finished labeling all 146 pairs"
3. The AI will:
   - Read your labels from the Excel file
   - Run threshold tuning on all 44 metrics
   - Generate calibrated thresholds based on YOUR judgments
   - Update sales materials with real results

---

**Ready?** Run:
```bash
python 3d_viewer_server.py
```

Then open: http://localhost:5000

🚀 Happy labeling!
