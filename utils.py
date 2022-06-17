import torch
import numpy as np
import kornia
import torch.nn.functional as F


def meshgrid_xy(tensor1: torch.Tensor, tensor2: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """Mimick np.meshgrid(..., indexing="xy") in pytorch. torch.meshgrid only allows "ij" indexing.
	(If you're unsure what this means, safely skip trying to understand this, and run a tiny example!)

	Args:
	tensor1 (torch.Tensor): Tensor whose elements define the first dimension of the returned meshgrid.
	tensor2 (torch.Tensor): Tensor whose elements define the second dimension of the returned meshgrid.
	"""
    ii, jj = torch.meshgrid(tensor1, tensor2, indexing='ij')
    return ii.transpose(-1, -2), jj.transpose(-1, -2)

def get_camera_intrinsics(w, h, focal_length):
    x0, y0 = w / 2, h / 2
    s = 0
    camera_mat = torch.FloatTensor([
        [0, s, x0, 0],
        [0, 0, y0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]).unsqueeze(0) #.to(dev)
    camera_mat[0, 0, 0] = focal_length
    camera_mat[0, 1, 1] = focal_length
    return camera_mat


def get_camera_extrinsics():
    """
	http://www.cse.psu.edu/~rtc12/CSE486/lecture12.pdf
	https://www.scratchapixel.com/lessons/3d-basic-rendering/computing-pixel-coordinates-of-3d-point/mathematics-computing-2d-coordinates-of-3d-points
	obj_matrix_world =
	[ r11 r12 r13 tx ]
	[ r21 r22 r23 ty ]
	[ r31 r32 r33 tz ]
	[ 0   0   0   1  ]

	[r11 r21 r31].T => world X axis (1,0,0) in camera coords
	https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
	There are 3 coordinate systems involved:
	   1. The World coordinates: "world"
		  - right-handed
	   2. The Blender camera coordinates: "bcam"
		  - x is horizontal
		  - y is up
		  - right-handed: negative z look-at direction
	   3. The desired computer vision camera coordinates: "cv"
		  - x is horizontal
		  - y is down (to align to the actual pixel coordinates
			used in digital images)
		  - right-handed: positive z look-at direction
	"""
    RT = torch.eye(4)  # rotation translation matrix
    RT[2, 3] = 1  # camera is placed in (0, 0, 1)

    # camera is looking down
    RT[0, 0] = 1
    RT[1, 1] = -1
    RT[2, 2] = -1
    RT = RT  #.to(dev)
    return RT


def pnt_world_to_cam_coords(pnt):
    RT = get_camera_extrinsics()
    pnt = pnt.unsqueeze(0) if (len(pnt.shape) < 2) else pnt
    pnt_hmg = kornia.geometry.convert_points_to_homogeneous(pnt)
    return RT @ pnt_hmg.T

def pnt_world_coords_to_pixel_coords(pnt, K, RT, return_depth=False):
    pnt = pnt.unsqueeze(0) if (len(pnt.shape) < 2) else pnt
    pnt_hmg = kornia.geometry.conversions.convert_points_to_homogeneous(pnt)
    pnt_final_old = (K @ (RT @ pnt_hmg.T)).squeeze()

    pnt_final = pnt_final_old.clone()
    pnt_final /= pnt_final_old.clone()[2]
    if return_depth:
        return torch.hstack([pnt_final[0:2], pnt_final_old[2]])
    else:
        return pnt_final[0:2]


def depth_map_to_pointcloud(depth_map, K, RT, w, h):
    xyz: torch.Tensor = kornia.geometry.depth_to_3d(depth_map.reshape(1, 1, w, h), K, normalize_points=False).squeeze()

    xyz = xyz.reshape(3, -1).permute(1, 0)
    xyz = kornia.geometry.convert_points_to_homogeneous(xyz)
    xyz = (RT.inverse().squeeze() @ xyz.T).T
    xyz = kornia.geometry.convert_points_from_homogeneous(xyz)  # .permute(1, 0)
    xyz = xyz.reshape(w, h, 3)
    return xyz


def get_ray_bundle(height: int, width: int, normalize=True):
    r"""Compute the bundle of rays passing through all pixels of an image (one ray per pixel).
	Args:
	height (int): Height of an image (number of pixels).
	width (int): Width of an image (number of pixels).

	Returns:
	ray_origins (torch.Tensor): A tensor of shape :math:`(width, height, 2)` denoting the centers of
	each ray. `ray_origins[i][j]` denotes the origin of the ray passing through pixel at
	row index `j` and column index `i`.
	AK: Fixed from torch.arange(width)
	"""
    if normalize:
        ii, jj = meshgrid_xy(
            F.normalize(torch.linspace(0, width, width) - width * 0.5, dim=0, p=np.inf),
            F.normalize(torch.linspace(0, height, height) - height * 0.5, dim=0, p=np.inf)
        )

    else:
        ii, jj = meshgrid_xy(
            torch.arange(width),
            torch.arange(height)
        )
    ray_origins = torch.stack([ii, jj], dim=-1)  # returns w, h, 2
    # if normalize:
    # 	ray_origins += (torch.randn_like(ray_origins) * 0.0001)  # try to generalize better
    return ray_origins


def positional_encoding(tensor, num_encoding_functions=5, include_input=True, log_sampling=True) -> torch.Tensor:
    r"""Apply positional encoding to the input.

	Args:
	tensor (torch.Tensor): Input tensor to be positionally encoded.
	num_encoding_functions (optional, int): Number of encoding functions used to
	compute a positional encoding (default: 6).
	include_input (optional, bool): Whether or not to include the input in the
	computed positional encoding (default: True).
	log_sampling (optional, bool): Sample logarithmically in frequency space, as
	opposed to linearly (default: True).

	Returns:
	(torch.Tensor): Positional encoding of the input tensor.
	"""
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    # Now, encode the input using a set of high-frequency functions and append the
    # resulting values to the encoding.
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)


def norm_diff(normal_hat, norm_gt, silhouette=None):
    """Tensor Dim: NxCxHxW"""
    if norm_gt.ndim != 4:
        norm_gt = norm_gt.unsqueeze(0)
    if normal_hat.ndim != 4:
        normal_hat = normal_hat.unsqueeze(0)
    if norm_gt.shape[1] != 3:
        print("Warning: norm_diff received wrong shape for norm_gt")
        norm_gt = norm_gt.permute(0, 3, 1, 2)
    if normal_hat.shape[1] != 3:
        print("Warning: norm_diff received wrong shape for normal_hat")
        normal_hat = normal_hat.permute(0, 3, 1, 2)
    if silhouette is None:
        silhouette = torch.ones((1, 1, norm_gt.shape[2], norm_gt.shape[3])).to(norm_gt.device)
    elif silhouette.ndim != 4:
        silhouette = silhouette.reshape(1, 1, normal_hat.shape[2], normal_hat.shape[3])

    dot_product = (norm_gt * normal_hat).sum(1).clamp(-1, 1)
    error_map = torch.acos(dot_product) # [-pi, pi]
    angular_map = error_map * 180.0 / np.pi
    angular_map = angular_map * silhouette.narrow(1, 0, 1).squeeze(1)

    valid = silhouette.narrow(1, 0, 1).sum()
    ang_valid = angular_map[silhouette.narrow(1, 0, 1).squeeze(1).byte()]
    n_err_mean = ang_valid.sum() / valid
    return angular_map.squeeze(), n_err_mean