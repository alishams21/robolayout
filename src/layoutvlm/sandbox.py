import random
import torch
import re
import numpy as np
import os
import sys
from typing import Any
from .grad_solver import GradSolver

# ANSI colors for constraint log (conflict=red, no conflict=green)
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"


def _is_conflict_line(line: str) -> bool:
    """True if this log line indicates a rejected/conflict constraint."""
    line = line.strip()
    if not line:
        return False
    return (
        "==> (rejected)" in line
        or "==> (reject " in line
        or " is already specified" in line
        or "conflicts with existing" in line
    )


def _use_color() -> bool:
    """Use ANSI colors only when stdout is a TTY."""
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _red(s: str) -> str:
    return f"{RED}{s}{RESET}" if _use_color() else s


def _print_constraint_log(existing_constraint_str: str, new_constraint_str: str) -> None:
    """Print constraint log to console with green for accepted and red for conflicts."""
    use_color = _use_color()
    green = GREEN if use_color else ""
    red = RED if use_color else ""
    reset = RESET if use_color else ""

    print("\n--- Existing constraints (no conflict) ---")
    for line in existing_constraint_str.strip().splitlines():
        print(f"{green}{line}{reset}")

    print("\n--- New constraints ---")
    for line in new_constraint_str.strip().splitlines():
        if _is_conflict_line(line):
            print(f"{red}{line}{reset}")
        else:
            print(f"{green}{line}{reset}")
from .constraints import Constraint
from .constraints import align_with, on_top_of
from .constraints import distance_constraint
from .constraints import point_towards
from .constraints import against_wall
from .constraints import symmetric_pair
from utils.placement_utils import get_random_placement
from .scene import AssetInstance, Wall
from .device_utils import get_device_with_index, to_device
from . import loss_utils




class SandBoxEnv:
    def __init__(self, task, mode="default", save_dir=None, robot_radius=None):
        self.task = task
        self.mode = mode
        self.boundary = task["boundary"]["floor_vertices"]
        self.save_dir = save_dir
        # Radius of the virtual robot used for reachability constraints
        self.robot_radius = robot_radius
        self.grad_solver = GradSolver(self.boundary, robot_radius=self.robot_radius)
        self.local_vars = {}
        self.all_code = ""
        self.all_constraints = []
        self.solver_step = 0

    def execute_code(self, code):
        self.all_code += code + "\n"
        self.export_code()
        exec(code, self.local_vars)

    def export_code(self, save_dir=None):
        if save_dir is None:
            save_dir = self.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        with open(f"{self.save_dir}/complete_sandbox_program.py", "w") as f:
            f.write(self.all_code)
    
    def initialize_variables(self):
        setup_code = ""
        for original_uid, asset in self.task["assets"].items():
            var_name = asset["asset_var_name"]
            asset_idx = int(original_uid.split('-')[-1])
            position = get_random_placement(self.task["boundary"]["floor_vertices"], add_z=True)
            position[-1] = asset["assetMetadata"]["boundingBox"]["z"]/2
            if asset['onCeiling']:
                position[-1] = 3
                setup_code += f"{var_name}.onCeiling = True\n"

            setup_code += f"{var_name}[{asset_idx}].position = {position}\n"
            setup_code += f"{var_name}[{asset_idx}].rotation = [0, 0, 0]\n"

        self.all_code += setup_code
        self.export_code()
        self.execute_code(setup_code)
        # Validate
        for var_name, asset in self.local_vars.items():
            if type(asset).__name__ == "Assets":
                for instance in asset.placements:
                    assert instance.position is not None
                    assert instance.rotation is not None
    
    def assign_instance_ids(self):
        setup_code = ""
        for var_name, asset in self.local_vars.items():
            if type(asset).__name__ == "Assets":
                for idx, _ in enumerate(asset.placements):
                    asset_instance_id = f"{var_name}_{idx}"
                    setup_code += f"{var_name}[{idx}].instance_id = '{asset_instance_id}'\n"
        for wall_idx in range(len(self.boundary)):
            setup_code += f"walls[{wall_idx}].instance_id = 'walls_{wall_idx}'\n"
        self.execute_code(setup_code)

    def setup_initial_assets(self):
        num_walls = len(self.boundary)
        wall_assets = {}
        for idx in range(num_walls):
            vertices = np.array([
                self.boundary[idx],
                self.boundary[(idx+1)%num_walls]
            ]).astype(np.float32)
            wall_id = f"walls_{idx}"
            wall_assets[wall_id] = Wall(
                wall_id,
                vertices=[vertices[0], vertices[1]],
            )
        return wall_assets

    def sanity_check(self, group_assets, entire_program, constraint_for_all=False):
        _local_vars = {}
        try:
            exec(self.all_code, _local_vars)
            exec(entire_program, _local_vars)
        except Exception as e:
            assert False, f"Error in the sandbox code: {e}"

        for var_name, asset in _local_vars.items():
            if type(asset).__name__ == "Assets":
                for instance in asset.placements:
                    assert instance.instance_id is not None
                    assert instance.position is not None
                    assert instance.rotation is not None
            if type(asset).__name__ == "Walls":
                for wall in asset.walls:
                    assert wall.instance_id is not None
                    assert wall.corner1 is not None
                    assert wall.corner2 is not None

        if constraint_for_all:
            # check if all the instances to be placed are specified with constraints
            all_instance_ids = []
            for constraint in _local_vars['solver'].constraints:
                all_instance_ids.extend(constraint[1])
            for instance_var_name in group_assets:
                if instance_var_name not in all_instance_ids:
                    print(f"Instance {instance_var_name} is not specified in the constraints")
                    assert False

    def setup_optimization_param(self, placed_instance_ids, new_instance_ids, new_constraints):
        all_instance_ids = placed_instance_ids + new_instance_ids
        for constraint, instance_ids in self.all_constraints + new_constraints:
            all_instance_ids.extend(instance_ids)
        solver_assets = self.setup_initial_assets()
        for instance_id in all_instance_ids:
            if instance_id.startswith("walls_") or instance_id == "room_0":
                continue
            ### NOTE: we want fixed point assetinstnaces to be created
            ### this is for the random instance that LLMs generate (for constraints with absolute coordinates)
            asset_var_name = "_".join(instance_id.split("_")[:-1])
            instance_idx = int(instance_id.split("_")[-1])
            if len(self.local_vars[asset_var_name].placements[instance_idx].rotation) == 1:
                self.local_vars[asset_var_name].placements[instance_idx].rotation = [0, 0, self.local_vars[asset_var_name].placements[instance_idx].rotation[0]]
            # change all to radians
            #self.local_vars[asset_var_name].placements[instance_idx].rotation = [np.deg2rad(r) for r in self.local_vars[asset_var_name].placements[instance_idx].rotation]

            if self.mode == "no_initialization":
                random_pos = get_random_placement(self.boundary, add_z=True)
                solver_assets[instance_id] = AssetInstance(
                    id=instance_id,
                    position=random_pos,
                    rotation=[0, 0, random.uniform(0, 360)],
                    size=self.local_vars[asset_var_name].size,
                    onCeiling=self.local_vars[asset_var_name].onCeiling,
                    optimize=1 if not instance_id.startswith("fixed_point") else 0
                )
            else:
                solver_assets[instance_id] = AssetInstance(
                    id=instance_id,
                    position=self.local_vars[asset_var_name].placements[instance_idx].position,
                    rotation=self.local_vars[asset_var_name].placements[instance_idx].rotation,
                    size=self.local_vars[asset_var_name].size,
                    onCeiling=self.local_vars[asset_var_name].onCeiling,
                    optimize=1 if not instance_id.startswith("fixed_point") else 0
                )
        return solver_assets

    def self_consistency_filtering(self, solver_assets, new_constraints):
        existing_constraint_str = ""
        for constraint, instance_ids in self.all_constraints:
            params_str = ', '.join([f"{k}={v}" for k, v in constraint.params.items()])
            existing_constraint_str += f"solver.{constraint.constraint_name}({instance_ids[0]}, {instance_ids[1]}, {params_str})\n"

        filtered_new_constraints = []
        new_constraint_str = ""
        for constraint, instance_ids in new_constraints:
            if constraint.constraint_name == "on_top_of":
                if instance_ids[0].startswith("fixed_point") or instance_ids[1].startswith("fixed_point"):
                    continue
                if not any(pair[0] == instance_ids[0] for pair in self.grad_solver.on_top_of_assets):
                    self.grad_solver.on_top_of_assets.append((instance_ids[0], instance_ids[1]))
                    filtered_new_constraints.append((constraint, instance_ids))
                    new_constraint_str += f"solver.on_top_of({instance_ids[0]}, {instance_ids[1]})\n"

        for constraint, instance_ids in new_constraints:
            if instance_ids[0].startswith("fixed_point") or instance_ids[1].startswith("fixed_point"):
                continue

            existing_constraints_count = sum(
                1 for c in self.all_constraints + filtered_new_constraints
                if c[0].constraint_name == constraint.constraint_name and c[1][0] == instance_ids[0]
            )
            
            if constraint.constraint_name == "against_wall":
                if existing_constraints_count > 0:
                    new_constraint_str += f"constraint {constraint.constraint_name} with instance_ids {instance_ids} is already specified\n"
                    new_constraint_str += f"==> (rejected) solver.{constraint.constraint_name}({instance_ids[0]}, {instance_ids[1]})\n"
                    continue
                asset_pos = solver_assets[instance_ids[0]].position.detach().clone().cpu().numpy()
                def get_distance_to_wall(asset_pos, wall_id):
                    wall_start = np.array(solver_assets[wall_id].corner1[:2])
                    wall_end = np.array(solver_assets[wall_id].corner2[:2])
                    # Calculate the vector from wall_start to wall_end
                    wall_vector = wall_end - wall_start
                    # Calculate the vector from wall_start to the asset
                    asset_vector = asset_pos[:2] - wall_start
                    # Project asset_vector onto wall_vector
                    projection = np.dot(asset_vector, wall_vector) / np.dot(wall_vector, wall_vector)
                    # Calculate the closest point on the wall
                    closest_point = wall_start + projection * wall_vector
                    # Calculate the distance from the asset to the closest point
                    distance = np.linalg.norm(asset_pos[:2] - closest_point)
                    return distance
                
                # get the wall_id with the minimum distance
                min_distance = float('inf')
                min_wall_id = None
                for wall_id in solver_assets.keys():
                    if wall_id.startswith("walls_"):
                        distance = get_distance_to_wall(asset_pos, wall_id)
                        if distance < min_distance:
                            min_distance = distance
                            min_wall_id = wall_id
                instance_ids = [instance_ids[0], min_wall_id]
                filtered_new_constraints.append((constraint, instance_ids))
                if min_wall_id != instance_ids[1]:
                    new_constraint_str += f"==> (updated) solver.{constraint.constraint_name}({instance_ids[0]}, {instance_ids[1]})\n"
                else:
                    new_constraint_str += f"solver.{constraint.constraint_name}({instance_ids[0]}, {instance_ids[1]})\n"
                continue

                filtered_new_constraints.append((constraint, instance_ids))
                new_constraint_str += f"solver.{constraint.constraint_name}({instance_ids[0]}, {instance_ids[1]})\n"

            elif constraint.constraint_name == "distance_constraint":
                if existing_constraints_count > 2:
                    print(_red(f"constraint {constraint.constraint_name} with instance_ids {instance_ids} is already specified"))
                    new_constraint_str += f"==> (rejected) solver.{constraint.constraint_name}({instance_ids[0]}, {instance_ids[1]}, {constraint.params['min_distance']}, {constraint.params['max_distance']})\n"
                    continue
                if solver_assets[instance_ids[0]].onCeiling or solver_assets[instance_ids[1]].onCeiling:
                    new_constraint_str += f"==> (reject distance constraints with Ceiling assets) solver.{constraint.constraint_name}({instance_ids[0]}, {instance_ids[1]})\n"
                    continue
                if solver_assets[instance_ids[0]].size is None or solver_assets[instance_ids[1]].size is None:
                    print(_red(f"constraint {constraint.constraint_name} with instance_ids {instance_ids} is not (rejected as distance to wall not supported)"))
                    new_constraint_str += f"==> (rejected as distance to wall not supported) solver.{constraint.constraint_name}({instance_ids[0]}, {instance_ids[1]}, {constraint.params['min_distance']}, {constraint.params['max_distance']})\n"
                    continue
                # Calculate the distance between two assets using self.local_vars
                with torch.no_grad():
                    pos1 = solver_assets[instance_ids[0]].position.detach().clone()
                    pos2 = solver_assets[instance_ids[1]].position.detach().clone()
                    distance = torch.norm(pos1 - pos2).item()
                    min_distance = min(solver_assets[instance_ids[0]].size[0], solver_assets[instance_ids[0]].size[1])/2 + min(solver_assets[instance_ids[1]].size[0], solver_assets[instance_ids[1]].size[1])/2
                    if constraint.params["min_distance"] is not None:
                        constraint.params["min_distance"] = min(distance, constraint.params["min_distance"])
                        if constraint.params["min_distance"] < min_distance:
                            constraint.params["min_distance"] = min_distance 
                    if constraint.params["max_distance"] is not None:
                        constraint.params["max_distance"] = max(distance, constraint.params["max_distance"])
                        ### NOTE: (added heurstics) max distance should not be too tight
                        if constraint.params["max_distance"] < min_distance * 1.5:
                            constraint.params["max_distance"] = min_distance * 1.5

                filtered_new_constraints.append((constraint, instance_ids))
                new_constraint_str += f"solver.distance_constraint({instance_ids[0]}, {instance_ids[1]}, {constraint.params['min_distance']}, {constraint.params['max_distance']})\n"

            elif constraint.constraint_name == "point_towards":
                # check if instance_ids[0] already has a point_towards or align_with constraint
                has_existing_orientation_constraint = any(
                    (c[0].constraint_name in ["point_towards", "against_wall"] and c[1][0] == instance_ids[0])
                    for c in self.all_constraints + filtered_new_constraints
                )
                if has_existing_orientation_constraint:
                    print(_red(f"constraint {constraint.constraint_name} with instance_ids {instance_ids} conflicts with existing constraint"))
                    new_constraint_str += f"==> (rejected) solver.{constraint.constraint_name}({instance_ids[0]}, {instance_ids[1]}, {constraint.params['angle']})\n"
                    continue

                filtered_new_constraints.append((constraint, instance_ids))
                new_constraint_str += f"solver.{constraint.constraint_name}({instance_ids[0]}, {instance_ids[1]}, {constraint.params['angle']})\n"

            elif constraint.constraint_name == "align_with":
                # check if instance_ids[0] already has a point_towards or align_with constraint
                has_existing_orientation_constraint = any(
                    (c[0].constraint_name in ["align_with", "against_wall"] and c[1][0] == instance_ids[0])
                    for c in self.all_constraints + filtered_new_constraints
                )
                if has_existing_orientation_constraint:
                    print(_red(f"constraint {constraint.constraint_name} with instance_ids {instance_ids} conflicts with existing constraint"))
                    new_constraint_str += f"==> (rejected) solver.{constraint.constraint_name}({instance_ids[0]}, {instance_ids[1]}, {constraint.params['angle']})\n"
                    continue

                filtered_new_constraints.append((constraint, instance_ids))
                new_constraint_str += f"solver.{constraint.constraint_name}({instance_ids[0]}, {instance_ids[1]}, {constraint.params['angle']})\n"

            elif constraint.constraint_name == "symmetric_pair":
                # instance_ids = [id_a, id_b, id_reference]; reject if fixed_point or duplicate pair
                if len(instance_ids) != 3:
                    new_constraint_str += f"==> (rejected) solver.symmetric_pair(...) requires exactly 3 arguments (asset_a, asset_b, reference)\n"
                    continue
                if any(uid.startswith("fixed_point") for uid in instance_ids):
                    continue
                pair = frozenset([instance_ids[0], instance_ids[1]])
                already_has = any(
                    c[0].constraint_name == "symmetric_pair" and frozenset(c[1][:2]) == pair
                    for c in self.all_constraints + filtered_new_constraints
                )
                if already_has:
                    new_constraint_str += f"==> (rejected) solver.symmetric_pair({instance_ids[0]}, {instance_ids[1]}, {instance_ids[2]}) — pair already has symmetry\n"
                    continue
                filtered_new_constraints.append((constraint, instance_ids))
                new_constraint_str += f"solver.symmetric_pair({instance_ids[0]}, {instance_ids[1]}, {instance_ids[2]})\n"

            elif constraint.constraint_name == "on_top_of":
                pass

            else:
                assert False, f"constraint {constraint.constraint_name} is not supported"

        return filtered_new_constraints, [existing_constraint_str, new_constraint_str]

    def build_constraint_functions(self):
        constraints_for_solver = []
        for constraint in self.local_vars['solver'].constraints:
            function_name = constraint[0].constraint_name
            # skip fixed_point constraints
            if constraint[1][0].startswith("fixed_point"):
                continue

            if function_name == "against_wall":
                constraints_for_solver.append(
                    (
                        Constraint(
                            constraint_name=function_name,
                            constraint_func=against_wall,
                        ),
                        constraint[1]
                    )
                )
            elif function_name == "distance_constraint":
                constraints_for_solver.append(
                    (
                        Constraint(
                            constraint_name=function_name,
                            constraint_func=distance_constraint,
                            min_distance=constraint[0].params["min_distance"],
                            max_distance=constraint[0].params["max_distance"],
                            weight=constraint[0].params["weight"]
                        ),
                        constraint[1]
                    )
                )
                constraint
            elif function_name == "on_top_of":
                constraints_for_solver.append(
                    (
                        Constraint(
                            constraint_name=function_name,
                            constraint_func=on_top_of,
                        ),
                        constraint[1]
                    )
                )
            elif function_name == "align_with":
                constraints_for_solver.append(
                    (
                        Constraint(
                            constraint_name=function_name,
                            constraint_func=align_with,
                            angle=constraint[0].params["angle"]
                        ),
                        constraint[1]
                    )
                )
            elif function_name == "point_towards":
                constraints_for_solver.append(
                    (
                        Constraint(
                            constraint_name=function_name,
                            constraint_func=point_towards,
                            angle=constraint[0].params["angle"]
                        ),
                        constraint[1]
                    )
                )
            elif function_name == "symmetric_pair":
                # constraint[1] = [id_a, id_b, id_reference] (reference = wall or central asset)
                constraints_for_solver.append(
                    (
                        Constraint(
                            constraint_name=function_name,
                            constraint_func=symmetric_pair,
                            orientation_weight=constraint[0].params.get("orientation_weight", 0.5),
                        ),
                        constraint[1]
                    )
                )
            elif function_name == "align_x":
                assert False, "align_x should not be used"
                #constraints_for_solver.append(
                #    (
                #        Constraint(
                #            constraint_name=function_name,
                #            constraint_func=align_x,
                #        ),
                #        constraint[1]
                #    )
                #)
            elif function_name == "align_y":
                assert False, "align_y should not be used"
                #constraints_for_solver.append(
                #    (
                #        Constraint(
                #            constraint_name=function_name,
                #            constraint_func=align_y,
                #        ),
                #        constraint[1]
                #    )
                #)
            else:
                assert False

        return constraints_for_solver

    def export_layout(self, incomplete_scene=False, use_degree=True):
        results = dict()
        if incomplete_scene:
            for original_uid, asset in self.task["assets"].items():
                var_name = asset["asset_var_name"]
                asset_idx = int(original_uid.split('-')[-1])
                if self.local_vars[var_name].placements[asset_idx].optimize == 2:
                    assert self.local_vars[var_name].placements[asset_idx].position
                    results[original_uid] = {
                        "position": self.local_vars[var_name].placements[asset_idx].position,
                        "rotation": [np.rad2deg(x) for x in self.local_vars[var_name].placements[asset_idx].rotation]
                    }
        else:
            for original_uid, asset in self.task["assets"].items():
                var_name = asset["asset_var_name"]
                asset_idx = int(original_uid.split('-')[-1])
                try:
                    results[original_uid] = {
                        "position": self.local_vars[var_name].placements[asset_idx].position,
                        "rotation": [np.rad2deg(x) for x in self.local_vars[var_name].placements[asset_idx].rotation]
                    }
                except:
                    print(f"extract placement for asset {original_uid} failed.")
        return results

    def solve(self, placed_assets, group_assets, program_segment, save_dir, only_initialize=False):
        ### replace '-' with '_' in the instance ids (for correspondence to variable names)
        placed_instance_ids = [_instance.replace('-', '_') for _instance in list(placed_assets.keys())]
        group_assets = [_instance_id.replace('-', '_') for _instance_id in group_assets]
        # initialize the asset positions/rotations
        self.execute_code(program_segment)
        ### Create a sandbox environment to safely execute the code
        #sandbox_globals = {}
        #sandbox_locals = self.local_vars.copy()
        #try:
        #    # Execute the code in the sandbox environment
        #    exec(program_segment, sandbox_globals, sandbox_locals)
        #    # If execution succeeds, update the actual environment
        #    self.local_vars.update(sandbox_locals)
        #except Exception as e:
        #    print(f"Error executing code in sandbox: {e}")
        #    # Handle the error appropriately, maybe log it or raise a custom exception
        #    return

        # After successful initialization
        new_constraints = self.build_constraint_functions()
        solver_assets = self.setup_optimization_param(placed_instance_ids, group_assets, new_constraints)
        print("assets given to grad_solver", solver_assets.keys())
        for k in solver_assets:
            print(solver_assets[k])

        if only_initialize:
            return self.export_layout(incomplete_scene=True)

        if "no_self_consistency" not in self.mode:
            new_constraints, [existing_constraint_str, new_constraint_str] = self.self_consistency_filtering(solver_assets, new_constraints)
            _print_constraint_log(existing_constraint_str, new_constraint_str)
            with open(f"{save_dir}/new_constraints.txt", "w") as f:
                f.write(existing_constraint_str)
                f.write("\n=================================\n")
                f.write(new_constraint_str)

        ### no constraint optimization mode (ablation)
        if len(new_constraints) == 0 or "no_constraint" in self.mode:
            solver_code = "solver.constraints = []\n"
            for instance_id in solver_assets.keys():
                asset_var_name = "_".join(instance_id.split("_")[:-1])
                instance_idx = int(instance_id.split("_")[-1])
                solver_code += f"{asset_var_name}[{instance_idx}].optimize = 2\n"
        else:
            ### use constraints to further optimize the pose of assets
            loss_dir = loss_utils.get_loss_dir(save_dir)
            results = self.grad_solver.optimize(
                assets=solver_assets,
                existing_constraints=self.all_constraints,
                new_constraints=new_constraints,
                temp_dir=f"{save_dir}/temp_{self.solver_step}",
                output_gif_path=f"{save_dir}/out.gif",
                loss_dir=loss_dir,
                loss_filename_suffix=str(self.solver_step),
            )
            ##########################################################################
            ### merge these results back with self.local_vars
            ##########################################################################
            # NOTE: or keep the old constriants?
            solver_code = "solver.constraints = []\n"
            for instance_id in results.keys():
                asset_var_name = "_".join(instance_id.split("_")[:-1])
                instance_idx = int(instance_id.split("_")[-1])
                solver_code += f"{asset_var_name}[{instance_idx}].position = {results[instance_id]['position']}\n"
                solver_code += f"{asset_var_name}[{instance_idx}].rotation = {results[instance_id]['rotation']}\n"
                ### continue to optimize the assets
                solver_code += f"{asset_var_name}[{instance_idx}].optimize = 2\n"

        # Execute solver_code in sandbox environment
        self.execute_code(solver_code)
        self.solver_step += 1
        self.all_constraints += new_constraints
        # export code to the specific save_dir for this solver call
        self.export_code(save_dir)
        return self.export_layout(incomplete_scene=True)
