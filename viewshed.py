import numpy as np
import torch.nn as nn
import torch


def rad2deg(rad):
	return rad * 180 / np.pi

###################################
def calc_los_world_coords_batch(viewpoint_loc, lines_xyz, temp=-50, eps=1e-10):
	"""
	function which calculates 1d shading (los= light of sight, analog problem),
	given light location and point cloud of object (generated from depth map).
	Returns shaded / not shaded result for each pixel

	inputs:
	line_xyz : (3, N, L)
	viewpoint_loc: (3)
	"""
	# commented for speedup
	# assert (lines_xyz.shape[0] == 3)
	# assert (viewpoint_loc.shape[0] == 3)

	q = lines_xyz - viewpoint_loc.view(3, 1, 1)    # vec from l to p
	in_arcos = torch.sqrt(q[0] ** 2 + q[1] ** 2) / (torch.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2) + eps)
	in_arcos = in_arcos.clamp(-1, 1)  # numeric stability
	elevation_angle = torch.acos(in_arcos)
	elevation_angle = rad2deg(elevation_angle)
	min_angle_cumulative, _ = torch.cummin(elevation_angle, dim=0)
	shaded = 2 * torch.sigmoid((elevation_angle - min_angle_cumulative) * temp)
	shaded_real = (min_angle_cumulative - elevation_angle >= 0)

	return shaded, shaded_real

def get_los_2d_worldcoords_batch(lightsrc_loc, pixel_lines, xyz, temp=-50, return_last_points_only=False, eps=1e-6):
	"""
	All inputs should be in pixel units, in (v, u) pixels
	return_last_points_only: return the last coordinates' value along the line

	xyz.shape = (3, W, H)
	pixel_lines.shape = (num lines, 2, max line length)
	"""

	assert (xyz.shape[0] == 3)
	dev = lightsrc_loc.device

	N, _, L = pixel_lines.shape

	pixel_line_vu_coords = pixel_lines.round().long()
	pixel_line_vu_coords = pixel_line_vu_coords.permute(1, 2, 0).flatten(start_dim=1, end_dim=2)
	lines_xyz = torch.stack([xyz[ii][[*pixel_line_vu_coords]] for ii in range(3)]).to(dev)
	lines_xyz = lines_xyz.reshape(3, L, N).permute(0, 2, 1)
	temp = torch.as_tensor([temp]).to(dev)

	# Nerf way:
	shaded, shaded_real = calc_los_world_coords_batch(lightsrc_loc, lines_xyz, temp, eps)

	if return_last_points_only:
		return shaded[-1], shaded_real[-1]
	else:
		return shaded, shaded_real
###################################


def calc_los_world_coords(viewpoint_loc, line_xyz, temp=-50, eps=1e-10):
	"""
	function which calculates 1d shading (los= light of sight, analog problem),
	given light location and point cloud of object (generated from depth map).
	Returns shaded / not shaded result for each pixel

	inputs:
	line_xyz : (3, N)
	viewpoint_loc: (3)
	"""
	# commented for speedup
	# assert (line_xyz.shape[0] == 3)
	# assert (viewpoint_loc.shape[0] == 3)

	q = line_xyz - viewpoint_loc.unsqueeze(1)    # vec from l to p
	in_arcos = torch.sqrt(q[0] ** 2 + q[1] ** 2) / (torch.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2) + eps)
	in_arcos = in_arcos.clamp(-1, 1)  # for numeric stability
	elevation_angle = torch.rad2deg(torch.acos(in_arcos))
	min_angle_cumulative, _ = torch.cummin(elevation_angle, dim=0)
	shaded = 2 * torch.sigmoid((elevation_angle - min_angle_cumulative) * temp)
	shaded_real = (min_angle_cumulative - elevation_angle >= 0)
	return shaded, shaded_real


def get_los_2d_worldcoords(lightsrc_loc, pixel_line, xyz, temp=-50, return_last_points_only=False, eps=1e-6):
	"""
	All inputs should be in pixel units, in (v, u) pixels
	return_last_points_only: return the last coordinates' value along the line

	xyz.shape = (3, W, H)
	"""
	# commented for speedup
	# assert (xyz.shape[0] == 3)
	dev = lightsrc_loc.device

	pixel_line_vu_coords = pixel_line.round().long()
	line_xyz = torch.stack([xyz[ii][[*pixel_line_vu_coords]] for ii in range(3)]).to(dev)
	temp = torch.as_tensor([temp], device=dev)

	# Nerf way:
	shaded, shaded_real = calc_los_world_coords(lightsrc_loc, line_xyz, temp, eps)

	if return_last_points_only:
		return shaded[-1], shaded_real[-1]
	else:
		return shaded, shaded_real


def calc_los(viewpoint_loc, depth_map, temp=1e3, dist_lightsrc_to_boundary=torch.as_tensor([0])):
	"""
	function which calculates 1d shading (los= light of sight, analog problem),
	given viewpoint location, depth map (height map).
	Returns shaded / not shaded result for each pixel

	pixel_to_unit - how many pixels are equal to a single unit of measure (e.g. meter)
	"""
	dev = depth_map.device
	viewpoint_height = viewpoint_loc[-1]
	size_of_res = int(np.abs(len(depth_map)))

	x_dist = torch.arange(0, size_of_res, device=dev)

	# # if viewpoint is "off the grid"
	# if viewpoint_x < 0 or viewpoint_x > len(depth_map):
	# 	x_dist += torch.abs(viewpoint_x).long().to(dev)

	x_dist += dist_lightsrc_to_boundary.long()
	height_diff = (-viewpoint_height + depth_map)

	# angle between 2 points and the horizontal axis
	elevation_angle = rad2deg(torch.atan2(height_diff, x_dist))

	max_angle_cumulative, _ = torch.cummax(elevation_angle, dim=0)
	shaded = 2 * torch.sigmoid((elevation_angle - max_angle_cumulative) * temp)
	shaded_real = (elevation_angle - max_angle_cumulative >= 0)

	return shaded, shaded_real


def get_los_2d(lightsrc_loc, pixel_line, depth_map_1d, pixel_to_unit=1, temp=-1, return_last_points_only=False):
	"""
	All inputs should be in pixel units!!
	return_last_points_only: return the last coordinates' value along the line

	"""
	dev = depth_map_1d.device
	temp = torch.as_tensor([temp], device=dev)
	boundary_intersection = pixel_line[:, 0]  # point at which line from lightsrc to pixel intersects frame boundary
	dist_from_boundary = dist(lightsrc_loc[0:2], boundary_intersection)

	shaded, shaded_real = calc_los(lightsrc_loc, depth_map_1d, temp, dist_lightsrc_to_boundary=dist_from_boundary)

	if return_last_points_only:
		return shaded[-1], shaded_real[-1]
	else:
		return shaded, shaded_real


def line_intersection(line1, line2):
	"""
	line in format np.array[[x1,y1],[x2,y2]]
	returns intersection point
	"""
	try:
		t, s = torch.linalg.solve(torch.vstack([line1[1] - line1[0], line2[0] - line2[1]]).T, line2[0] - line1[0])
		return (1 - t) * line1[0] + t * line1[1]
	except:  # np.linalg.LinAlgError:
		print("no intersection")
		return None


def check_line_intersects(line1, line2):
	"""
	line in format np.array[[x1,y1],[x2,y2]]
	:return if there is an intersection
	"""
	try:
		t, s = torch.linalg.solve(torch.vstack([line1[1] - line1[0], line2[0] - line2[1]]).T, line2[0] - line1[0])
		return (0 <= t <= 1) and (0 <= s <= 1)
	except:  # torch.linalg.LinAlgError:
		# print("no intersection while checking")
		return False


# angle between two points
def angle_between_points(point1, point2):
	""" assuming (0,0) is origin """
	unit_vector_1 = point1 / (np.linalg.norm(point1) + 1e-10)
	unit_vector_2 = point2 / (np.linalg.norm(point2) + 1e-10)
	dot_product = np.dot(unit_vector_1, unit_vector_2)
	angle = np.arccos(dot_product)
	return rad2deg(angle)


# get vectors of all lines from viewpoint to square boundaries
def _lin_interp(point1, point2, num_steps):
	t = torch.linspace(0, 1, num_steps, device=point1.device)
	interp1 = point1[0] + (point2[0] - point1[0]) * t
	interp2 = point1[1] + (point2[1] - point1[1]) * t
	return torch.stack([interp1, interp2])


def dist(p1, p2):
	return torch.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def generate_square_boundaries(top_left, bottom_right, num_points):
	x_l, y_t = top_left
	x_r, y_b = bottom_right
	num_points_w = np.abs(x_l - x_r) + 1
	num_points_h = np.abs(y_b - y_t) + 1
	horizontal = torch.linspace(x_l, x_r, num_points_w)
	vertical = torch.linspace(y_b, y_t, num_points_h)
	top_line = torch.vstack([horizontal, torch.as_tensor([y_t]).expand(num_points_w)])
	bottom_line = torch.vstack([horizontal, torch.as_tensor([y_b]).expand(num_points_w)])
	left_line = torch.vstack([torch.as_tensor([x_l]).expand(num_points_h), vertical])
	right_line = torch.vstack([torch.as_tensor([x_r]).expand(num_points_h), vertical])

	# return boundary
	return torch.cat([top_line, right_line, bottom_line, left_line], dim=1).type(torch.LongTensor).T


def light_src_to_pnt_boundary_intersection(light_src, w, h, point):
	""" get intersection point between light source, which is outside
	of the frame, on the boundary of square"""
	v, u = light_src

	"""line in format np.array[[x1,y1],[x2,y2]]"""
	# line from origin to projection of light src to xy plane
	line1 = torch.as_tensor([[*point], [v, u]])

	# one of 4 square edges - calc which one
	# center of square is (0,0) for sake of calculation
	# angle_of_xy = angle_between_points(np.array([w, h]), np.array([x, y]))

	eps = 1e-3  # used to make sure no corner case of line between 2 border lines
	lineA = torch.as_tensor([[0.0 - eps, w + eps], [h + eps, w + eps]])
	lineB = torch.as_tensor([[0.0 - eps, 0.0 - eps], [0 - eps, w + eps]])
	lineC = torch.as_tensor([[0.0 - eps, 0.0 - eps], [h + eps, 0 - eps]])
	lineD = torch.as_tensor([[h, 0.0 - eps], [h + eps, w + eps]])

	if check_line_intersects(line1, lineA) and (u >= 0):  # 0 < angle_of_xy < 90
		# print("case1")
		line2 = lineA
	elif check_line_intersects(line1, lineB) and (v <= 0):  # 90 <= angle_of_xy < 180:
		# print("case2")
		line2 = lineB
	elif check_line_intersects(line1, lineC) and (u <= 0):  # 180 <= angle_of_xy < 270:
		# print("case3")
		line2 = lineC
	elif check_line_intersects(line1, lineD):
		# print("case4")
		line2 = lineD
	else:
		raise Exception(f"no intersection between lightsrc={light_src}, point={point} and frame boundary ")

	xy = line_intersection(line1, line2)
	# x, y = xy if xy else (None, None)
	# assert (0 <= x <= w and 0 <= y <= h), f"x or y have wrong values, case {case}, xy = {x},{y}, light={light_src}"
	return xy.long()


def gen_lines_from_src_to_points(boundary_points, light_src_loc, w, h, num_pts=-1):
	"""
	inputs should be in int, from 0 to width/height, given in (v,u) order!
	points outside the range [0, width/height) will be dropped.
	"""
	# assert boundary_points.shape[1] == 2, "incorrect format of points"

	all_lines = []
	light_src_in_frame = (0 <= light_src_loc[1] <= w) and (0 <= light_src_loc[0] <= h)
	if light_src_in_frame:
		print(f"light src {light_src_loc} in frame")

	for ii in range(len(boundary_points[:])):
		pnt = boundary_points[ii]
		light_src_loc_fixed = light_src_to_pnt_boundary_intersection(light_src_loc, w, h, pnt) if not light_src_in_frame else light_src_loc
		num_pts_on_line = int(dist(pnt, light_src_loc_fixed)) + 1

		# if explicitly indicated num_pts
		if num_pts > 0:
			if num_pts_on_line < 1:  # if num of pixels in generated line is too small, skip
				continue
			else:
				num_pts_on_line = num_pts

		line = _lin_interp(light_src_loc_fixed, pnt, num_pts_on_line)
		v, u = line
		filtered_line = line[:, (v >= 0) * (u >= 0)]  # get only positive coords
		v, u = filtered_line
		filtered_line = filtered_line[:, (v <= h) * (u <= w)]  # get only positive coords
		all_lines.append(filtered_line)

	return all_lines
