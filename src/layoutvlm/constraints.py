import math
import torch
from .constraint_utils import *
import numpy as np
from shapely.geometry import Polygon
from .device_utils import get_device_with_index, to_device
import warnings

# Try to import oriented_iou_loss, fallback if CUDA not available
try:
    from third_party.Rotated_IoU import oriented_iou_loss
    ORIENTED_IOU_AVAILABLE = True
except ImportError as e:
    ORIENTED_IOU_AVAILABLE = False
    warnings.warn(
        "WARNING: Could not import oriented_iou_loss from third_party.Rotated_IoU. "
        "This is likely due to CUDA not being available or the CUDA extension not being compiled. "
        "The bbox_overlap_loss function will use a simplified fallback that may be less accurate. "
        f"Original error: {e}"
    )

# Only print the "no CUDA / oriented IoU fallback" warning once per process
_oriented_iou_fallback_warned = False


class Constraint:
    def __init__(self, constraint_name, constraint_func, description="", **params):
        self.constraint_name = constraint_name
        self.constraint_func = constraint_func
        self.description = description.format(**params)
        self.params = params
    
    def evaluate(self, assets: list, device=None):
        if device is None:
            device = get_device_with_index()
        # Ensure device is available
        if device.startswith('cuda') and not torch.cuda.is_available():
            device = 'cpu'
        return self.constraint_func(assets, **self.params, device=device)

def get_bounding_box(obj):
    position = obj.position
    size = obj.size
    return position[0] - size[0]/2, position[0] + size[0]/2, position[1] - size[1]/2, position[1] + size[1]/2, position[2] - size[2]/2, position[2] + size[2]/2

def get_center_position(obj):
    min_x, max_x, min_y, max_y, min_z, max_z = get_bounding_box(obj)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_z = (min_z + max_z) / 2
    return center_x, center_y, center_z


def _cpu_fallback_giou(corners1, corners2, area1, area2, device):
    """
    CPU-only fallback for GIoU calculation using Shapely for polygon intersection.
    This is less accurate than the CUDA implementation but allows the code to run without CUDA.
    
    Args:
        corners1: (B, N, 4, 2) tensor of polygon corners
        corners2: (B, N, 4, 2) tensor of polygon corners  
        area1: (B, N) tensor of areas
        area2: (B, N) tensor of areas
        device: target device
    
    Returns:
        giou_loss: (N,) tensor
        iou: (N,) tensor
    """
    B, N = corners1.shape[0], corners1.shape[1]
    
    giou_losses = []
    ious = []
    
    for i in range(N):
        try:
            # Convert to numpy for Shapely
            poly1_corners = corners1[0, i].detach().cpu().numpy()
            poly2_corners = corners2[0, i].detach().cpu().numpy()
            
            # Create Shapely polygons
            poly1 = Polygon(poly1_corners)
            poly2 = Polygon(poly2_corners)
            
            # Ensure polygons are valid
            if not poly1.is_valid:
                poly1 = poly1.buffer(0)
            if not poly2.is_valid:
                poly2 = poly2.buffer(0)
            
            # Calculate intersection
            intersection = poly1.intersection(poly2)
            intersection_area = intersection.area if hasattr(intersection, 'area') else 0.0
            
            # Calculate union
            union_area = poly1.area + poly2.area - intersection_area
            
            # Calculate IoU
            iou_val = intersection_area / union_area if union_area > 0 else 0.0
            
            # Calculate enclosing box (axis-aligned bounding box of both polygons)
            combined_coords = np.vstack([poly1_corners, poly2_corners])
            min_x, min_y = combined_coords.min(axis=0)
            max_x, max_y = combined_coords.max(axis=0)
            enclosing_area = (max_x - min_x) * (max_y - min_y)
            
            # Calculate GIoU
            giou_val = iou_val - (enclosing_area - union_area) / enclosing_area if enclosing_area > 0 else iou_val
            
            # Convert to loss (1 - GIoU)
            giou_loss_val = 1.0 - giou_val
            
        except Exception as e:
            # If Shapely fails, use zero values
            warnings.warn(f"Shapely polygon calculation failed: {e}. Using zero values.")
            giou_loss_val = 0.0
            iou_val = 0.0
        
        giou_losses.append(giou_loss_val)
        ious.append(iou_val)
    
    # Convert to tensors
    giou_loss = torch.tensor(giou_losses, dtype=torch.float32, requires_grad=True, device=device)
    iou = torch.tensor(ious, dtype=torch.float32, requires_grad=True, device=device)
    
    return giou_loss, iou


def bbox_overlap_loss(assets: list, skipped_asset_pairs: list=[], only_consider_overlapping_assets=False, detach_asset2=False, consider_z_axis=True, epsilon=1e-5, device=None):
    if device is None:
        device = get_device_with_index()
    # Ensure device is available
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    """
    This function calculates the loss for the 3D bounding boxes of the assets to not overlap
    """
    def segment_overlap(x1, y1, x2, y2):
        max_start = max(x1, x2)
        min_end = min(y1, y2)
        overlap_length = max(0, min_end - max_start)
        return overlap_length

    num_assets = len(assets)
    if num_assets < 2:
        return torch.tensor(0.0, requires_grad=True).to(device), torch.tensor(0.0, requires_grad=True).to(device)

    overlap_coefs = []
    corners1 = []
    corners2 = []
    area1 = []
    area2 = []
    for i in range(num_assets):
        asset_i = assets[i]
        area_i = asset_i.size[0] * asset_i.size[1]

        for j in range(i+1, num_assets):
            asset_j = assets[j]
            area_j = asset_j.size[0] * asset_j.size[1]
            if (asset_i.id, asset_j.id) in skipped_asset_pairs or (asset_j.id, asset_i.id) in skipped_asset_pairs:
                continue
            if only_consider_overlapping_assets:
                with torch.no_grad():
                    corner_i = asset_i.get_2dpolygon().detach().cpu().numpy()
                    corner_j = asset_j.get_2dpolygon().detach().cpu().numpy()
                    poly_i = Polygon(corner_i)
                    poly_j = Polygon(corner_j)
                    if not poly_i.intersects(poly_j):
                        continue

            if consider_z_axis:
                overlap_coef = segment_overlap(
                    asset_i.position[-1].item() - asset_i.size[-1]/2,
                    asset_i.position[-1].item() + asset_i.size[-1]/2,
                    asset_j.position[-1].item() - asset_j.size[-1]/2,
                    asset_j.position[-1].item() + asset_j.size[-1]/2
                )
                if overlap_coef < 0.05:
                    overlap_coef = 0
                overlap_coefs.append(overlap_coef)

            if detach_asset2:
                corners1.append(asset_i.get_2dpolygon())
                corners2.append(asset_j.get_2dpolygon())
                area1.append(area_i)
                area2.append(area_j)
            else:
                if abs(area_i - area_j) < epsilon:
                    corners1.append(asset_i.get_2dpolygon())
                    corners2.append(asset_j.get_2dpolygon())
                    area1.append(area_i)
                    area2.append(area_j)

                    corners1.append(asset_j.get_2dpolygon())
                    corners2.append(asset_i.get_2dpolygon())
                    area1.append(area_j)
                    area2.append(area_i)

                    if consider_z_axis:
                        overlap_coefs.append(overlap_coef)
                else:
                    small_asset, bigger_asset = (asset_i, asset_j) if area_i < area_j else (asset_j, asset_i)
                    corners1.append(small_asset.get_2dpolygon())
                    corners2.append(bigger_asset.get_2dpolygon())
                    area1.append(min(area_i, area_j))
                    area2.append(max(area_i, area_j))

    if len(corners1) == 0:
        return torch.tensor(0.0, requires_grad=False), torch.tensor(0.0, requires_grad=False)
    
    corners1 = torch.stack(corners1, dim=0).unsqueeze(0)
    corners2 = torch.stack(corners2, dim=0).unsqueeze(0)
    area1 = torch.tensor(area1, dtype=torch.float32, requires_grad=False).unsqueeze(0)
    area2 = torch.tensor(area2, dtype=torch.float32, requires_grad=False).unsqueeze(0)

    if ORIENTED_IOU_AVAILABLE:
        # Use the oriented IoU loss if available
        try:
            # Check if cal_my_giou exists, otherwise use cal_giou
            if hasattr(oriented_iou_loss, 'cal_my_giou'):
                giou_loss, iou = oriented_iou_loss.cal_my_giou(
                    corners1.to(device), corners2.to(device).detach(),
                    area1.to(device), area2.to(device).detach()
                )
            else:
                # Fallback to cal_giou if cal_my_giou doesn't exist
                # Convert corners to box format (x, y, w, h, angle) if needed
                warnings.warn("cal_my_giou not found, using simplified fallback calculation")
                giou_loss, iou = _cpu_fallback_giou(corners1, corners2, area1, area2, device)
        except Exception as e:
            warnings.warn(f"Error using oriented_iou_loss: {e}. Using simplified fallback.")
            giou_loss, iou = _cpu_fallback_giou(corners1, corners2, area1, area2, device)
    else:
        # Fallback: use a simplified overlap calculation (warn only once)
        global _oriented_iou_fallback_warned
        if not _oriented_iou_fallback_warned:
            _oriented_iou_fallback_warned = True
            print("WARNING: No CUDA device available, skipping important oriented IoU loss function. "
                  "Results may be less accurate.")
        giou_loss, iou = _cpu_fallback_giou(corners1, corners2, area1, area2, device)

    if consider_z_axis:
        overlap_coefs = torch.tensor(overlap_coefs, dtype=torch.float32, requires_grad=False).unsqueeze(0).to(device)
        giou_loss = giou_loss * overlap_coefs
        iou = iou * overlap_coefs

    return -torch.mean(giou_loss), torch.sum(iou)


################################
### distance-based
################################
def distance_constraint(assets: list, min_distance, max_distance, weight=1., device=None):
    if device is None:
        device = get_device_with_index()
    # Ensure device is available
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    assert len(assets) == 2
    coord1 = assets[0].position[:2].to(device)
    coord2 = assets[1].position[:2].to(device).detach()
    loss = distance_loss(coord1, coord2, min_distance=min_distance, max_distance=max_distance)
    return weight * torch.clamp(loss, max=1)

def distance_constraint_deterministic(assets: list, min_distance, max_distance, weight=1., device=None):
    if device is None:
        device = get_device_with_index()
    # Ensure device is available
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    assert len(assets) == 2
    distance = torch.linalg.norm(assets[0].position[:2] - assets[1].position[:2])
    if min_distance < distance < max_distance:
        return (torch.tensor(0.0), distance)
    else:
        return (torch.tensor(100000.0), distance)

################################
### top-bottom based
################################
def on_top_of_deterministic(assets: list, device=None):
    if device is None:
        device = get_device_with_index()
    # Ensure device is available
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    assert len(assets) == 2
    min_x1, max_x1, min_y1, max_y1, min_z1, max_z1 = get_bounding_box(assets[0])
    min_x2, max_x2, min_y2, max_y2, min_z2, max_z2 = get_bounding_box(assets[1])
    
    coord1 = torch.tensor(min_z1, dtype=torch.float32)
    coord2 = torch.tensor(max_z2, dtype=torch.float32)

    vertical_loss = coord1 - coord2
    if vertical_loss < 0.1:
        return torch.tensor(10)
    else:
        _, iou = bbox_overlap_loss(assets, detach_asset2=True, consider_z_axis=False, device=device)
        return -10 * iou

def on_top_of(assets: list, device=None):
    if device is None:
        device = get_device_with_index()
    # Ensure device is available
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    assert len(assets) == 2
    _, iou = bbox_overlap_loss(assets, detach_asset2=True, consider_z_axis=False, device=device)
    return torch.clamp(-10 * iou, min=-10, max=10)

################################
### orientation-based
################################
def point_towards(assets: list, angle=0, device=None):
    if device is None:
        device = get_device_with_index()
    # Ensure device is available
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    assert len(assets) == 2
    asset1, asset2 = assets
    vector1 = asset1.get_2dvector(add_radian=-math.radians(angle)).to(device)
    corners2 = asset2.get_2dpolygon().to(device)

    with torch.no_grad():
        intersects = ray_intersects_polygon(
            origin=asset1.position[:2].detach().cpu().numpy(),
            direction=vector1.detach().cpu().numpy(),
            polygon=corners2.detach().cpu().numpy()
        )

    if intersects:
        return torch.tensor(0.0, requires_grad=True, device=device)
    else:
        vector2 = (asset2.position[:2] - asset1.position[:2]).to(device).detach()
        return cosine_distance_loss(vector1, vector2)

def align_with(assets: list, angle=0, device=None):
    if device is None:
        device = get_device_with_index()
    # Ensure device is available
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    assert len(assets) == 2
    asset1, asset2 = assets
    vector1 = asset1.get_2dvector(add_radian=-math.radians(angle)).to(device)
    vector2 = asset2.get_2dvector().to(device).detach()
    return cosine_distance_loss(vector1, vector2)

################################
### others
################################
def against_wall(assets: list, device=None):
    if device is None:
        device = get_device_with_index()
    # Ensure device is available
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    assert len(assets) == 2
    asset, wall= assets
    vector = asset.get_2dvector(add_radian=-math.radians(90)).to(device)
    corners = asset.get_2dpolygon().to(device)
    segment = torch.tensor([
        [
            wall.corner1[0], wall.corner1[1],
            wall.corner2[0], wall.corner2[1]
        ]
    ], dtype=corners.dtype, requires_grad=False).to(device)
    distances = point_to_segment_batch_loss(corners[:4, ...], segment)
    angle_difference = cosine_distance_loss(vector, wall.get_2dvector())
    return torch.clamp(torch.sum(distances[:2, 0]), max=10) + 10 * angle_difference


def symmetric_pair(assets: list, orientation_weight=0.5, device=None):
    """
    Place two assets symmetrically about a reference (wall or central object).
    assets = [asset_a, asset_b, reference]. Reference can be a Wall or an AssetInstance
    (e.g. bed). Symmetry axis: through reference; for a wall, perpendicular to the wall;
    for an asset, perpendicular to its front (so left/right mirror).
    """
    if device is None:
        device = get_device_with_index()
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    assert len(assets) == 3
    asset_a, asset_b, reference = assets
    # Axis: origin (2D) and unit direction (2D)
    if hasattr(reference, 'corner1') and hasattr(reference, 'corner2'):
        # Wall: axis = perpendicular to wall through midpoint
        c1 = reference.corner1
        c2 = reference.corner2
        origin = torch.tensor(
            [(c1[0] + c2[0]) / 2.0, (c1[1] + c2[1]) / 2.0],
            dtype=torch.float32, device=device
        )
        wall_vec = torch.tensor(
            [c2[0] - c1[0], c2[1] - c1[1]],
            dtype=torch.float32, device=device
        )
        # Perpendicular (rotate 90): (-dy, dx)
        axis_dir = torch.tensor(
            [-wall_vec[1].item(), wall_vec[0].item()],
            dtype=torch.float32, device=device
        )
        axis_dir = F.normalize(axis_dir.unsqueeze(0), p=2, dim=-1).squeeze(0)
    else:
        # Asset: axis through position, perpendicular to front (left/right symmetry)
        origin = reference.position[:2].to(device).detach()
        front = reference.get_2dvector().to(device).detach()
        # Perpendicular: (-front_y, front_x)
        axis_dir = torch.stack([-front[1], front[0]], dim=0)
        axis_dir = F.normalize(axis_dir.unsqueeze(0), p=2, dim=-1).squeeze(0)

    def reflect_2d(p, c, u):
        # p, c, u: (2,) tensors; u unit. Reflect p across line through c with direction u.
        p = p.to(device)
        diff = p - c
        dot = torch.dot(diff, u)
        return c + 2 * dot * u - diff

    pos_a = asset_a.position[:2].to(device)
    pos_b = asset_b.position[:2].to(device)
    ref_a = reflect_2d(pos_a, origin, axis_dir)
    ref_b = reflect_2d(pos_b, origin, axis_dir)
    # Symmetric: reflect(a) ~ b and reflect(b) ~ a; gradient through both assets
    pos_loss = torch.sum((pos_b - ref_a) ** 2) + torch.sum((pos_a - ref_b) ** 2)

    # Mirror orientations: reflected forward vectors should align
    vec_a = asset_a.get_2dvector().to(device)
    vec_b = asset_b.get_2dvector().to(device)
    ref_vec_a = 2 * torch.dot(vec_a, axis_dir) * axis_dir - vec_a
    ref_vec_a = F.normalize(ref_vec_a.unsqueeze(0), p=2, dim=-1).squeeze(0)
    ref_vec_b = 2 * torch.dot(vec_b, axis_dir) * axis_dir - vec_b
    ref_vec_b = F.normalize(ref_vec_b.unsqueeze(0), p=2, dim=-1).squeeze(0)
    orient_loss = cosine_distance_loss(ref_vec_a, vec_b) + cosine_distance_loss(ref_vec_b, vec_a)

    return pos_loss + orientation_weight * orient_loss


################################
### deprecated
################################
# visual mark / boundary-based
#def locate_grid(assets):
#   asset1, grid = assets 
#   coord1 = asset1.position[:2]
#   coord2 = grid.position[:2]
#   assert coord1.requires_grad, "coord1 does not require gradients"
#   assert coord2.requires_grad, "coord2 does not require gradients"
#   #assert len(assets) == 2
#   return 0.01  * distance_loss(coord1, coord2, min_distance=0, max_distance=1)

#def align_x(assets):
#    assert len(assets) == 2
#    asset1, asset2 = assets
#    # Calculate the Mean Squared Error (MSE) between the x-coordinates
#    return torch.nn.functional.mse_loss(asset1.position[0], asset2.position[0])
#
#def align_y(assets):
#    assert len(assets) == 2
#    asset1, asset2 = assets
#    # Calculate the Mean Squared Error (MSE) between the x-coordinates
#    return torch.nn.functional.mse_loss(asset1.position[0], asset2.position[0])



ALL_CONSTRAINTS = {
    "distance": Constraint(
        constraint_name="distance_constraint",
        constraint_func=distance_constraint,
        description="the distance between the two objects should be within the specified range",
    ),
    "close_to": Constraint(
        constraint_name="close_to",
        constraint_func=distance_constraint,
        description="",
        min_distance=0,
        max_distance=1
    ),
    "close_to_deterministic": Constraint(
        constraint_name="close_to",
        constraint_func=distance_constraint_deterministic,
        description="",
        min_distance=0,
        max_distance=1
    ),
    "moderate_distance": Constraint(
        constraint_name="moderate_distance",
        constraint_func=distance_constraint,
        description="",
        min_distance=1,
        max_distance=3
    ),
    "moderate_distance_deterministic": Constraint(
        constraint_name="moderate_distance",
        constraint_func=distance_constraint_deterministic,
        description="",
        min_distance=1,
        max_distance=3
    ),
    "point_towards": Constraint(
        constraint_name="point_towards",
        constraint_func=point_towards,
        description="the oriented bounding box of first object should be pointing towards the second object",
    ),
    "against_wall": Constraint(
        constraint_name="against_wall",
        constraint_func=against_wall,
        description="the bounding box of the first object should overlap with the bounding box of the second object",
    ),
    "on_top_of_deterministic": Constraint(
        constraint_name="on_top_of_deterministic",
        constraint_func=on_top_of_deterministic,
        description="the first object should be on the second object within the specified distance range",
    ),
    "on_top_of": Constraint(
        constraint_name="on_top_of",
        constraint_func=on_top_of,
        description="the first object should be on the second object within the specified distance range",
    ),
    "align_with": Constraint(
        constraint_name="align_with",
        constraint_func=align_with,
        description="the first object should be aligned with the second object both in orientation and distance",
    ),
    "symmetric_pair": Constraint(
        constraint_name="symmetric_pair",
        constraint_func=symmetric_pair,
        description="place two objects symmetrically about a reference (wall or central object); use for e.g. nightstands on either side of a bed",
    ),
}
