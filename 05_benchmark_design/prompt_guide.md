# Prompt Writing Guide for STEP-Based CAD Benchmarking

## Goal
Write prompts that are:
- Fully specified (minimal ambiguity)
- Objectively measurable against a reference STEP
- Reproducible across models and replicates

## Core Rules
1. Always specify units (default: mm).
2. Define coordinate frame and origin explicitly.
3. Specify all critical dimensions and positions.
4. State geometric relationships (centered, symmetric, tangent, concentric, equal spacing).
5. Specify cut behavior (through vs blind + depth).
6. Require single connected component unless task says otherwise.
7. Avoid subjective wording ("nice", "smooth", "ergonomic").

## Required Prompt Structure
Use this exact section order.

1. Part identity
- One line naming the part and objective.

2. Units and frame
- Units: mm
- Coordinate system: right-handed XYZ
- Origin location
- Axis meaning (which axis is length/height/etc.)

3. Base geometry
- Primitive type(s), dimensions, exact placement.

4. Additive features
- Bosses/ribs/extrusions with exact dimensions and positions.

5. Subtractive features
- Holes/slots/pockets with exact dimensions, axis, center, and depth.

6. Constraints
- Symmetry, concentricity, equal spacing, tangency, coplanarity.

7. Topology/output constraints
- Single connected solid
- No floating bodies
- Match dimensions exactly

## Prompt Template
Create a single connected mechanical CAD part.

Units and frame:
- Units: millimeters.
- Coordinate system: right-handed XYZ.
- Origin: [0, 0, 0] at <define exact location>.
- Orientation: <define axis roles>.

Base geometry:
- Create <primitive> with dimensions <...>.
- Place it at <...>.

Additive features:
- Feature A: <type>, dimensions <...>, positioned at <...>.
- Feature B: <type>, dimensions <...>, positioned at <...>.

Subtractive features:
- Cut A: <type>, dimensions <...>, center <...>, axis <...>, depth <through/blind + value>.
- Cut B: <...>.

Constraints:
- <symmetry/concentric/equal spacing/tangent/...>.
- Keep exactly one connected solid component.

Return CAD geometry matching these constraints exactly.

## Good vs Bad
Bad:
Make a bracket with two holes and a slot.

Good:
Create a single connected L-bracket. Units mm. Origin at the outer bottom corner where legs meet.
Leg A: 80 x 30 x 8 (X x Y x Z), spanning X=0..80, Y=0..30, Z=0..8.
Leg B: 30 x 8 x 60, spanning X=0..30, Y=0..8, Z=0..60.
Through-hole 1: diameter 8, axis +Z, center (20, 15, 4).
Through-hole 2: diameter 8, axis +Z, center (60, 15, 4).
Slot on Leg B front face: rounded slot, length 24, width 8, centered at (15, 4, 35), long axis +Z, through Y thickness.
Exactly one connected solid component.

## Author Checklist
- [ ] Units specified
- [ ] Origin and axes specified
- [ ] All critical dimensions specified
- [ ] All critical positions specified
- [ ] Through/blind depth specified
- [ ] Pattern count/spacing/reference specified
- [ ] Single/multi-component requirement specified
- [ ] No ambiguous adjectives
- [ ] Prompt is measurable by Chamfer/Hausdorff/volume/bbox/topology

## Replicate Guidance
- Use identical prompt template across models.
- Run at least 3 replicates per model-task pair.
- Track mean and std for Chamfer and Hausdorff 95p.
- Flag unstable tasks with high metric variance.
