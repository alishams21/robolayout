# RoboLayout

## Overview

This repository extends [LayoutVLM](https://github.com/StanfordVL/layoutvlm) with agent-aware layout optimization for indoor scenes. It introduces explicit differentiable reachability constraints to ensure generated layouts are navigable and actionable by embodied agents with diverse physical capabilities (e.g., robots, humans, animals). The framework also includes a local refinement stage that selectively re-optimizes problematic object placements without rerunning full-scene optimization, improving stability and convergence while preserving semantic and physical plausibility.

## Architecture
RoboLayout comprises three main layers. Orchestration: The central orchestrator responsible for coordinating group-ings, rendering and other cognitive processes of the solver and sandbox. Sandbox: Translates constraints into feasible scene layouts. Solver: optimizer based on hard and soft constraints for spatial arrangements and refinement of the final optimized scene. Following shows the architecture diagram:

<p align="center">
  <img src="figures/architecture.png" alt="architecture diagram" >
</p>

### Implementation pipeline

1. **Initial state** — The pipeline takes the **language instruction** (`task_description`, `layout_criteria`) and the **room shape** (`boundary.floor_vertices`, `wall_height`) plus the list of assets as the initial program state. The sandbox is initialized with walls and asset variables; positions and rotations are set to random feasible placements inside the boundary.

2. **Furniture grouping** (non–one_shot modes) — An LLM groups furniture based on the prompt and task (e.g. “bed + nightstands”, “rug”, “seating”). Each group has a name and key spatial relations between assets. Grouping is written to `grouping.json`; placement then runs **group by group** with group-specific layout criteria.

3. **Pose estimation and spatial relations** — For each (group of) assets, the system produces **pose and spatial relations** by calling an LLM with the current scene (top-down and side renderings) and the layout criteria. The LLM returns a **constraint program**: high-level relations such as `against_wall`, `align_with`, `point_towards`, `distance_constraint`, `on_top_of`.

4. **Conversion to Python program** — The LLM output is parsed into executable Python: constraint calls (e.g. `solver.against_wall(...)`, `solver.distance_constraint(...)`) are executed in the sandbox. That updates the in-memory constraint list; the sandbox also exports the full program to `complete_sandbox_program.py` for inspection.

5. **Optimization with reachability** — The gradient-based solver (e.g. `GradSolver.optimize`) minimizes a loss over positions and rotations: **overlap loss** (no intersection), **existing- and new-constraint losses** (satisfy spatial relations), and **reachability loss**. Reachability encourages clearance between furniture so a virtual disc of radius `robot_radius` can pass; it is disabled if `robot_radius` is None or ≤ 0. Optimization runs per group (or once in one_shot), producing poses and optional per-step GIFs.

6. **Refinement** — After the main optimization, a **cleanup** step (e.g. `run_cleanup_step`) identifies problematic pairs (e.g. overlapping footprints), freezes non-problematic assets, and re-runs optimization only for the problematic subset. This refines the answer without full-scene re-optimization.

### Self-consistent decoding vs self-consistency filter

- **Self-consistent decoding** usually means: sample **multiple** outputs from the model (e.g. several constraint programs), then **select** one by a criterion (e.g. majority vote or best score). This codebase does **not** use self-consistent decoding: it obtains a single LLM constraint program per group (with retries on parse/execution failure), not multiple samples followed by selection.

- **Self-consistency filter** is what **is** implemented: before optimization, the sandbox runs **`self_consistency_filtering`** on the new constraints. It checks consistency with the current scene and existing constraints: rejects duplicate or conflicting constraints (e.g. duplicate `against_wall`, or a second orientation constraint on the same object), resolves `against_wall` to the nearest wall, and tightens distance constraints to feasible ranges. Rejected or updated constraints are logged (e.g. in `new_constraints.txt` with “(rejected)” / “(updated)”). Only the filtered constraint list is passed to the optimizer. So the pipeline uses a **constraint-level self-consistency filter**, not multi-sample self-consistent decoding.

## Installation

1. Clone this repository
2. Install dependencies (python 3.11):

```bash
pip install -r requirements.txt
```

3. Install Rotated IOU Loss ([https://github.com/lilanxiao/Rotated\_IoU](https://github.com/lilanxiao/Rotated\_IoU))

```
cd third_party/Rotated_IoU/cuda_op
python setup.py install
```

4. To run the code you need to run following command. There are two modes for running code: one-shot and finetuned. In one_shot: Place every asset in a single step using one layout prompt (no grouping). At finetuned: First use the LLM to group assets (e.g. bed + nightstands, then rug), then run layout and constraints group by group with group-specific prompts.:

```
python main.py \                                                                        
  --scene_json_file benchmark_tasks/...json \
  --openai_api_key Your-Key \
  --save_dir ./results/... \
  --model gpt-4o \
  --asset_dir ./objaverse_processed
  --mode finetuned
```

5. The final GIF showing progress of layout optimization (2D top-down view of furniture placement) is saved as `final.gif` in your results folder (e.g. `results/test_run/final.gif`). Example:

<p align="center">
  <img src="figures/final.gif" alt="Layout optimization progress" width="50%">
</p>



6. Loss curves to watch progress of optimization can be found in `results/loss`. Example:

<p align="center">
  <img src="figures/loss_curves.png" alt="Loss Curve" width="50%">
</p>


## Data preprocessing

1. Download the dataset [https://drive.google.com/file/d/1WGbj8gWn-f-BRwqPKfoY06budBzgM0pu/view?usp=sharing](https://drive.google.com/file/d/1WGbj8gWn-f-BRwqPKfoY06budBzgM0pu/view?usp=sharing)
2. Unzip it.

Refer to [https://github.com/allenai/Holodeck](https://github.com/allenai/Holodeck) and [https://github.com/allenai/objathor](https://github.com/allenai/objathor) for how we preprocess Objaverse assets.

## Usage

Prepare a scene configuration JSON file of Objaverse assets with the following structure:

```json
{
    "task_description": ...,
    "layout_criteria": ...,
    "boundary": {
        "floor_vertices": [[x1, y1, z1], [x2, y2, z2], ...],
        "wall_height": height
    },
    "assets": {
        "asset_id": {
            "path": "path/to/asset.glb",
            "assetMetadata": {
                "boundingBox": {
                    "x": width,
                    "y": depth,
                    "z": height
                }
            }
        }
    }
}
```

## Output

The script will generate a layout.json file in the specified save directory containing the optimized positions and orientations of all assets in the scene.

## BibTeX

```bibtex
@inproceedings{RoboLayout,
  title={RoboLayout: A Semantics-Driven 3D Layout Optimizer with Robotic Reachability Constraints},
  author={Ali Shamsaddinlou}
}
```
