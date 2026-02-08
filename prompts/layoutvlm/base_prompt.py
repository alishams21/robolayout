CODE_FOR_SANDBOX = """from math import cos, sin, radians
import uuid
from pydantic import BaseModel, Field, conint
from typing import List, Optional, Union


class Wall(BaseModel):
    optimize: int = Field(description="Whether to optimize the position and rotation of the asset", default=0)
    corner1: List[float] = Field(description="2d coordinates of the first corner of this wall")
    corner2: List[float] = Field(description="2d coordinates of the second corner of this wall")
    instance_id: Optional[str] = Field(description="Unique identifier for the wall", default=None)

class AssetInstance(BaseModel):
    optimize: int = Field(description="Whether to optimize the position and rotation of the asset", default=1)
    position: List[float] = Field(description="Position of the asset", default=[0,0,0])
    rotation: List[float] = Field(description="Rotation of the asset in degrees", default=[0,0,0])
    instance_id: Optional[str] = Field(description="Unique identifier for the asset instance", default=None)
    size: Optional[List[float]] = Field(description="Bounding box size of the asset instance", default=0.01)

    # if the position is 2d, add a zero z-coordinate
    def __init__(self, **data):
        if 'position' in data and len(data['position']) == 2:
            data['position'].append(0)
        super().__init__(**data)

    # Method to set size from the parent Assets class if not provided
    def set_size_from_parent(self, parent_size: Optional[List[float]]):
        if self.size is None and parent_size is not None:
            self.size = parent_size


class Assets(BaseModel):
    description: str = Field(description="Description of the asset")
    placements: List[AssetInstance] = Field(description="List of asset instances of this 3D asset", default=None)
    size: Optional[List[float]] = Field(description="Bounding box size of the asset (z-axis up)", default=None)
    onCeiling: bool = Field(description="Whether the asset is on the ceiling", default=False)

    def __getitem__(self, index: int) -> AssetInstance:
        "Allow indexing into placements."
        return self.placements[index]
    
    def __len__(self) -> int:
        "Allow using len() to get the number of placements."
        return len(self.placements)


class Constraint:
    def __init__(self, constraint_name, **params):
        self.constraint_name = constraint_name
        self.params = params
    
    def evaluate(self, assets: list):
        return self.constraint_func(assets, **self.params)


fixed_pointtt = Assets(description="global absolute marks", size=[0.01, 0.01, 0.01], placements=[])
def get_instance_id(asset):
    global fixed_pointtt
    if asset.instance_id is not None:
        return asset.instance_id

    if isinstance(asset, AssetInstance):
        # fine the name of the variable that is an instance of AssetInstance
        for var_name in globals(): 
            var = globals()[var_name]
            if isinstance(var, Assets):
                for instance_idx, placement in enumerate(var.placements):
                    if id(placement) == id(asset):
                        return f"{var_name}_{instance_idx}"

        # new instantiation of AssetInstance, return random instance_id
        instance_index = len(fixed_pointtt.placements)
        fixed_pointtt.placements.append(
            AssetInstance(
                position=asset.position,
                rotation=[0,0,0],
                optimize=False,
                instance_id=f"fixed_pointtt_{instance_index}"
            )
        )
        return fixed_pointtt.placements[-1].instance_id

    if isinstance(asset, Wall):
        var = globals()['walls']
        for instance_idx, wall_instance in enumerate(var):
            if id(wall_instance) == id(asset):
                return f"walls_{instance_idx}"


class ConstraintSolver:
    def __init__(self):
        self.constraints = []

    def handle_fixed_pointtt(self, asset):
        global fixed_pointtt
        if isinstance(asset, tuple):
            instance_index = len(fixed_pointtt.placements)
            fixed_pointtt.placements.append(
                AssetInstance(
                    position=list(asset),
                    rotation=[0,0,0],
                    optimize=False,
                    instance_id=f"fixed_pointtt_{instance_index}"
                )
            )
            return fixed_pointtt.placements[-1]
        return asset
    
    def point_towards(self, asset1: AssetInstance, asset2: Union[AssetInstance, tuple], angle=0):
        asset1.instance_id = get_instance_id(asset1)
        asset2 = self.handle_fixed_pointtt(asset2)
        asset2.instance_id = get_instance_id(asset2)
        self.constraints.append([
            Constraint("point_towards", angle=angle),
            [asset1.instance_id, asset2.instance_id]
        ])

    def distance_constraint(self, asset1: AssetInstance, asset2: Union[AssetInstance, tuple], min_distance=0, max_distance=10000, weight=1):
        asset1.instance_id = get_instance_id(asset1)
        asset2 = self.handle_fixed_pointtt(asset2)
        asset2.instance_id = get_instance_id(asset2)
        self.constraints.append([
            Constraint("distance_constraint", min_distance=min_distance, max_distance=max_distance, weight=weight),
            [asset1.instance_id, asset2.instance_id]
        ])

    def against_wall(self, asset1: AssetInstance, wall: Wall):
        asset1.instance_id = get_instance_id(asset1)
        wall.instance_id = get_instance_id(wall)
        self.constraints.append([
            Constraint("against_wall"),
            [asset1.instance_id, wall.instance_id]
        ])

    def on_top_of(self, asset1: AssetInstance, asset2: AssetInstance):
        asset1.instance_id = get_instance_id(asset1)
        asset2 = self.handle_fixed_pointtt(asset2)
        asset2.instance_id = get_instance_id(asset2)
        self.constraints.append([
            Constraint("on_top_of"),
            [asset1.instance_id, asset2.instance_id]
        ])
    
    def align_with(self, asset1: AssetInstance, asset2: AssetInstance, angle=0):
        asset1.instance_id = get_instance_id(asset1)
        asset2.instance_id = get_instance_id(asset2)
        self.constraints.append([
            Constraint("align_with", angle=angle),
            [asset1.instance_id, asset2.instance_id]
        ])

    def symmetric_pair(self, asset_a: AssetInstance, asset_b: AssetInstance, reference: Union[AssetInstance, Wall]):
        asset_a.instance_id = get_instance_id(asset_a)
        asset_b.instance_id = get_instance_id(asset_b)
        ref_instance_id = get_instance_id(reference)
        self.constraints.append([
            Constraint("symmetric_pair"),
            [asset_a.instance_id, asset_b.instance_id, ref_instance_id]
        ])

    def align_x(self, asset1: AssetInstance, asset2: AssetInstance):
        \"\"\"
        Add a constraint that asset1 should have the same x-coordinate as asset2.
        \"\"\"
        asset1.instance_id = get_instance_id(asset1)
        asset2.instance_id = get_instance_id(asset2)
        self.constraints.append([
            Constraint("align_x"),
            [asset1.instance_id, asset2.instance_id]
        ])

    def align_y(self, asset1: AssetInstance, asset2: AssetInstance):
        \"\"\"
        Add a constraint that asset1 should have the same y-coordinate as asset2.
        \"\"\"
        asset1.instance_id = get_instance_id(asset1)
        asset2.instance_id = get_instance_id(asset2)
        self.constraints.append([
            Constraint("align_y"),
            [asset1.instance_id, asset2.instance_id]
        ])

    def solve(self):
        pass

        
solver = ConstraintSolver()
"""


BASE_CLASS_DEFINITIONS = """from math import cos, sin
from pydantic import BaseModel, Field, conint
from typing import List, Optional


class Wall(BaseModel):
    corner1: List[float] = Field(description="2d coordinates of the first corner of this wall")
    corner2: List[float] = Field(description="2d coordinates of the second corner of this wall")

class AssetInstance(BaseModel):
    position: Optional[List[float]] = Field(description="Position of the asset", default=None)
    rotation: List[float] = Field(description="counterclockwise rotation of the asset in degrees (not radians). Only consider the last axis as objects are always upright. ", default=[0, 0, 0])

class Assets(BaseModel):
    description: str = Field(description="Description of the asset")
    placements: List[AssetInstance] = Field(description="List of asset instances of this 3D asset", default=None)
    size: Optional[List[float]] = Field(description="Bounding box size of the asset (z-axis up)", default=None)
    onCeiling: bool = Field(description="Whether the asset is on the ceiling", default=False)

    def __getitem__(self, index: int) -> AssetInstance:
        "Allow indexing into placements."
        return self.placements[index]
    
    def __len__(self) -> int:
        "Allow using len() to get the number of placements."
        return len(self.placements)

class ConstraintSolver:
    def __init__(self):
        self.constraints = []
    
    def on_top_of(self, asset1: AssetInstance, asset2: AssetInstance):
        \"\"\"
        Add a constraint that asset1 should be placed on top of asset2.

        Args:
            asset1: The asset that should be placed on top of asset2. The position of this asset will be adjusted based on the constraint.
            asset2: The asset that asset1 should be placed on top of.
        \"\"\"
        pass
    
    def against_wall(self, asset1: AssetInstance, wall: Wall):
        \"\"\"
        Add a constraint that asset1 should be placed against the wall.
        This constraint does not specify the orientation of the asset. Please use the point_towards or align_with constraint to specify the orientation of the asset.
        
        Args:
            asset1: the asset that should be placed against the wall
            wall: the wall that the asset should be placed against
        \"\"\"
        pass

    def distance_constraint(self, asset1: AssetInstance, asset2: AssetInstance, min_distance, max_distance, weight=1):
        \"\"\"
        Add a distance constraint between asset1 and asset2's bounding box center. The distance between asset1 and asset2 should be between min_distance (lower bound) and max_distance (uppser bound).

        Args:
            asset1: The first asset in the distance constraint. The position of this asset will be adjusted based on the constraint.
            asset2: The second asset in the distance constraint
            min_distance: The minimum allowed distance between asset1 and asset2 in meters. If None, there is no lower bound.
            max_distance: The maximum allowed distance between asset1 and asset2 in meters. If None, there is no upper bound.
            weight: The weight of this constraint. Default is 1. If this constraint is more important, increase the weight to 10. Otherwise, decrease the weight to 0.1. 
        
        Note: If min_distance equals max_distance, the constraint enforces an exact distance between the assets. Consider that the distance might have to depend on the size of the assets.
        \"\"\"
        pass
    
    def align_with(self, asset1: AssetInstance, asset2: AssetInstance, angle=0.):
        \"\"\"
        Add a constraint that asset1 should be placed angle degrees away to asset2. Note that this constraint is only for the rotation of the asset.

        Args:
            asset1: The asset that should be aligned with asset2. The rotation of this asset will be adjusted based on the constraint.
            asset2: The asset that asset1 should be aligned with.
            angle: The angle in degrees that asset1 should be rotated away from asset2. Default is 0. If angle=0, then asset1 should be placed parallel to asset2. If angle=180, then asset1 should have an opposite orientation to asset2.
            Note that asset1 aligning with asset2 does not imply that asset1 is "facing" or "pointing towards" asset2. Use the point_towards constraint for that.
        \"\"\"
        pass

    def point_towards(self, asset1: AssetInstance, asset2: AssetInstance, angle=0.):
        \"\"\"
        Add a constraint that asset1's front should point towards asset2.

        Args:
            asset1: The asset that should point towards asset2. The rotation of this asset will be adjusted based on the constraint.
            asset2: The asset that asset1 should point towards.
            angle: The angle in degrees that asset1 should be rotated away to face asset2. If angle=0, then asset1 should point towards asset2. If angle=180, the back side of asset1 should point towards asset2.
            Usually, you won't need to specify the angle.
        \"\"\"
        pass

    def symmetric_pair(self, asset_a: AssetInstance, asset_b: AssetInstance, reference):
        \"\"\"
        Place asset_a and asset_b symmetrically about a reference. Use for pairs that should mirror each other (e.g. nightstands on either side of a bed).

        Args:
            asset_a: First asset of the pair. Both asset_a and asset_b will be optimized to be mirror-symmetric.
            asset_b: Second asset of the pair.
            reference: The symmetry axis is defined by this reference. Can be a Wall or an AssetInstance (e.g. bed[0]).
        Example: solver.symmetric_pair(night_stand[0], night_stand[1], bed[0])
        \"\"\"
        pass

solver = ConstraintSolver()
"""

program_prompt = """
Your task is to write a layout program that specifies how to arrange a set of 3D assets in the scene conditioned on the floor plan (i.e., boundary specification). 
Here are the base class definitions:
```python
{base_class_definitions}
```
The list of constraints available are:
* z-axis constraints: on_top_of.
* position-based constraints: against_wall, distance_constraint.
* orientation-based constraints: align_with, point_towards.
* symmetry: symmetric_pair(asset_a, asset_b, reference) — place two assets symmetrically about a reference (Wall or AssetInstance, e.g. bed[0]). Example: solver.symmetric_pair(night_stand[0], night_stand[1], bed[0]).
Specify at least one position-based constraint and one orientation-based constraint for each asset to be placed.
Objects by default are placed on the floor. For small objects, please remember to specify on_top_of constraints to ensure they are placed on top of a surface.

You will be given a scene with or without existing assets, and a list of assets to be added to the scene. Your program must contain these parts:
(1) Based on the rendered image of the scene and the layout criteria, calculate and specify the position and rotation for the new asset placements.
(2) Speicfy constraints: based on the layout criteria, specify the constraints for the asset placements. These constraints will be used to ensure that the layout semantics are maintained when the layout is being adjusted to be physically feasible.

Note:
* The x-y plane is the floor, the z-axis is the vertical direction, and the origin is at the center of the room. Thus, if a sofa is on the floor, the z-coordinate should be half the z-axis bounding box size of the sofa.
* A zero degree rotation implies that the object faces +x axis. ([0, 0, 90] means that the object faces +y axis (counterclockwise rotation of 90 degrees).

Here are the dos and don'ts:
* It is very important to decide on the asset placements for all the assets to be placed. Do not forget to decide on the placemnents of any asset to be placed (not the existing assets).
* DO NOT HALLUCINATE ASSETS. Only use the assets provided in the scene description.
* The provided floor plan image is marked with a coordinate system. You can use the coordinate system to specify the placement of the assets in the scene.
* Enclose your answer in ```python ... ``` tags. You don't have to output the given program again.
* Write comments to explain the reasoning behind your code. PLEASE DO NOT REPEAT THE GIVEN PROGRAM and do not re-initialize any existing assets or walls.
* Make sure to specify the constraints for all new asset instances to be placed (i.e. each asset instance in the placements list of the Assets class). 
* Do not overwrite the varaibles of the assets in the scene as we will be using those variable names to extract the final layout (e.g., if there is a variable name chair, do not write `for chair in sofa.placements: ...`)
Only the placement of the the first argument (i.e. asset1) will be updated. The ordering of the arguments effectively decide the direction of the constraint.
If you want to update both assets' placements in the constraint `align_with` for example, you need to specify the constraint twice with the arguments swapped.
For example, if you specify `solver.point_towards(chair[0], sofa[0])`, the chair will point towards the sofa. If you specify `solver.point_towards(sofa[0], chair[0])`, the sofa will be adjusted to point towards the chair.
* Please give a comprehensive list of constraints that will ensure the assets are placed correctly in the scene.
* If you want to specify a constraint bewteen an asset and an absolute coordinate (e.g., the center of the room), you can instantiate a "fixed point" asset by doing AssetInstance(position=[x, y, z]) and use that as the second argument in the constraint function.
* When there are conflicting constraints between two assets, the first constraint you specify will be used. Thus, specify important constraints first.
For example, if you want to specify that a chair should be placed at the center of the room, you can do `solver.distance_constraint(chair[0], AssetInstance(position[center_x, center_y, center_z]), 0, 0, weight=10)`.

MAKE SURE TO ALWAYS SPECIFY "on_top_of" constraints for objects that should not be placed on the floor.
""".format(base_class_definitions=BASE_CLASS_DEFINITIONS)

sample_tasks = [
    (
        "Place the bookshelf against the wall with the shelves facing the center of the room.",
    ),
    (
        "Place the table in the center of the room, cushions around the table, a rug under the table, and armchairs around the table.",
    )
]
sample_task_programs = [
"""```python
# Walls that define the boundary of the scene
walls = [
    Wall(corner1=[0, 0, 0], corner2=[8, 0, 0]),
    Wall(corner1=[8, 0, 0], corner2=[8, 9, 0]),
    Wall(corner1=[8, 9, 0], corner2=[0, 9, 0]),
    Wall(corner1=[0, 9, 0], corner2=[0, 0, 0])
]

# New assets to be placed
bookshelf = Assets(description="A tall bookshelf with multiple shelves and a curved top, containing several books and two lower cabinets with doors.", placements=[AssetInstance(position=None, rotation=None) for _ in range(4)])
```
""",
"""```python
# Walls that define the boundary of the scene
walls = [
    Wall(corner1=[0, 0, 0], corner2=[8, 0, 0]),
    Wall(corner1=[8, 0, 0], corner2=[8, 9, 0]),
    Wall(corner1=[8, 9, 0], corner2=[0, 9, 0]),
    Wall(corner1=[0, 9, 0], corner2=[0, 0, 0])
]

# Existing assets in the scene
bookshelf_1 = Assets(description="A tall bookshelf with multiple shelves and a curved top, containing several books and two lower cabinets with doors.", placements=[AssetInstance(position=None, rotation=None) for _ in range(4)])
book_1 = Assets(description="A closed, rectangular book with a hard cover and visible pages.", placements=[AssetInstance(position=None, rotation=None) for _ in range(3)])
floor_lamp_1 = Assets(description="This is a floor lamp with a tall, slender black pole, a round base, and a large upward-facing lampshade at the top. It also has two smaller, adjustable reading lights attached to the pole.", placements=[AssetInstance(position=None, rotation=None) for _ in range(2)])
mantel_clock_1 = Assets(description="A vintage-style wooden mantel clock with a rounded top and a classic clock face featuring Roman numerals and two clock hands.", placements=[AssetInstance(position=None, rotation=None) for _ in range(1)])
potted_plant_1 = Assets(description="A lush, green potted plant with dense foliage, housed in a dark cylindrical pot.", placements=[AssetInstance(position=None, rotation=None) for _ in range(1)])
potted_plant_2 = Assets(description="A potted Monstera plant with large, split leaves, placed in a white pot supported by a wooden stand.", placements=[AssetInstance(position=None, rotation=None) for _ in range(2)])
potted_plant_4 = Assets(description="A potted plant with large, elongated green leaves, placed in a simple white cylindrical pot.", placements=[AssetInstance(position=None, rotation=None) for _ in range(1)])
potted_plant_5 = Assets(description="A small potted plant with green leaves, housed in a round, gray concrete pot. The plant has several stems and appears to be well-rooted in soil.", placements=[AssetInstance(position=None, rotation=None) for _ in range(1)])

# New assets to be placed
table = Assets(description="A rectangular wooden table with a smooth, light brown surface and sturdy, white metal legs.", placements=[AssetInstance(position=None, rotation=None) for _ in range(1)])
cushion = Assets(description="A round, plush cushion with a slightly indented center, typically used for seating or lounging.", placements=[AssetInstance(position=None, rotation=None) for _ in range(2)])
rug = Assets(description="This is a rectangular rug with a repeating diamond pattern in shades of red and orange, providing a warm and inviting appearance.", placements=[AssetInstance(position=None, rotation=None) for _ in range(1)])
armchair = Assets(description="A modern, cushioned armchair with a textured fabric cover draped over the backrest and armrests. The chair features a dark woven pattern on the seat and sides.", placements=[AssetInstance(position=None, rotation=None) for _ in range(4)])
```
"""
]
sample_constraint_programs = [
"""```python
bookshelf.placements[0].position = [4.0, 0, 0]
solver.against_wall(bookshelf.placements[0], walls[0], "against the wall with the shelves facing the center of the room")
```
""",
"""```python
# CONSTRAINT PROGRAM
# Table in the center of the room
solver.distance_constraint(table.placements[0], walls[0], 1.0, 2.0)
solver.distance_constraint(table.placements[0], walls[2], 1.0, 2.0)
solver.distance_constraint(table.placements[0], walls[1], 2.0, 3.0)
solver.distance_constraint(table.placements[0], walls[3], 2.0, 3.0)
solver.align_with(table.placements[0], walls[0], "parallel to the wall")
# Cushions around the table
for i in range(2):
    solver.distance_constraint(cushion.placements[i], table.placements[0], 0.5, 1.0)
    solver.align_with(cushion.placements[i], table.placements[0], "parallel to the table")
# Rug under the table
solver.on_top_of(table.placements[0], rug.placements[0])
# Armchairs around the table
for i in range(4):
    solver.distance_constraint(armchair.placements[i], table.placements[0], 0.5, 1.0)
    solver.point_towards(armchair.placements[i], table.placements[0])
```
""",
"""```python
solver.on_top_of(chair.placements[0], rug.placements[0])
solver.align_with(chair.placements[0], tv_stand.placements[0], "tmp")
solver.distance_constraint(chair.placements[0], tv_stand.placements[0], 1.5, 1.5)
solver.point_towards(book_C.placements[1], bookshelf.placements[0])
solver.on_top_of(book_C.placements[1], tv_stand.placements[0])
"""
]


def get_layout_prompt(current_task_program, layout_criteria, add_example=False):
    if add_example:
        in_context_examples = "\nHere are some examples:\n"
        for i in range(len(sample_tasks)):
            in_context_examples += \
                f"## Example {i}:\nLAYOUT CRITERIA: {sample_tasks[i]}\nEXISTING SCENE PROGRAM:\n{sample_task_programs[i]}\nCONSTRAINT PROGRAM:\n{sample_constraint_programs[i]}\n"
    
        return f"{program_prompt}\n\n{in_context_examples}\n\nLAYOUT CRITERIA: {layout_criteria}\nEXISTING SCENE PROGRAM:\n{current_task_program}\nCONSTRAINT PROGRAM:\n"
    else:
        return f"{program_prompt}\n\nLAYOUT CRITERIA: {layout_criteria}\nEXISTING SCENE PROGRAM:\n{current_task_program}\nCONSTRAINT PROGRAM:\n"