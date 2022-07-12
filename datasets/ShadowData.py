import cv2
import torch
import torch.utils.data as data
import json

import numpy as np
import os
try:
	import OpenEXR as exr
	import Imath
	openexr = True
except:
	os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
	openexr = False


def readEXR(filename):
	"""Read color + depth data from EXR image file.

	Parameters
	----------
	filename : str
		File path.

	Returns
	-------
	img : RGB or RGBA image in float32 format. Each color channel
		  lies within the interval [0, 1].
		  Color conversion from linear RGB to standard RGB is performed
		  internally. See https://en.wikipedia.org/wiki/SRGB#The_forward_transformation_(CIE_XYZ_to_sRGB)
		  for more information.

	Z : Depth buffer in float32 format or None if the EXR file has no Z channel.
	"""

	exrfile = exr.InputFile(filename)
	header = exrfile.header()

	dw = header['dataWindow']
	isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
	channelData = dict()

	# convert all channels in the image to numpy arrays
	for c in header['channels']:
		C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
		C = np.frombuffer(C, dtype=np.float32)
		C = np.reshape(C, isize)

		channelData[c] = C
	Z = channelData['R']
	return Z

class ShadowDataset(data.Dataset):
	def __init__(self, root, args, normalize_lights=True):
		self.root = os.path.join(root)
		self.light_dir = {}
		self.args = args
		dev = args.dev

		lines = self._read_list(os.path.join(self.root, 'all_object_lights.txt'))

		json_file = open(os.path.join(self.root, 'params.json'))
		self.params = json.load(json_file)
		json_file.close()

		for line in lines:
			name, x, y, z = line.split()
			self.light_dir[name] = np.asarray((x, y, z), dtype=float)
			if normalize_lights:
				self.light_dir[name] /= np.linalg.norm(self.light_dir[name])

		dummy_img_path = os.path.join(self.root, "0", [x for x in self.light_dir.keys()][0] + "_img.png")
		dummy_img = cv2.imread(dummy_img_path)
		self.h, self.w, self.c = dummy_img.shape  # original size

		path_base = dummy_img_path.split("_0_0")[0]
		normal_gt_path = path_base + "_normal.png"
		self.normal_gt = (cv2.cvtColor(cv2.imread(normal_gt_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0) * 2 - 1
		self.silhouette_gt = cv2.imread(path_base + "_silhouette.png")[:, :, 0] / 255.0
		self.depth_gt = None # cv2.imread(path_base + "_depth1.png")[:, :, 0] / 255.0
		if openexr:
			self.depth_exr = readEXR(path_base + "_depth.exr") if os.path.exists(path_base + "_depth.exr") else None
		else:
			self.depth_exr = cv2.imread(path_base + "_depth.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:,:,0]
		self.depth_exr = np.clip(self.depth_exr, a_min=0, a_max=500.)

		depth_img = self.depth_exr
		depth_img = depth_img - depth_img.min()
		depth_img /= depth_img.max()
		self.depth_exr_normed = depth_img

		self.normal_gt = torch.from_numpy(np.transpose(self.normal_gt, (2, 0, 1))).float().to(dev)
		self.silhouette_gt = torch.from_numpy(self.silhouette_gt).float().to(dev)

	@staticmethod
	def _read_list(list_path):
		with open(list_path) as f:
			lists = f.read().splitlines()
		return lists

	def __getitem__(self, index):
		dev = self.args.dev
		img_name = [x for x in self.light_dir.keys()][index]

		img_fname = os.path.join(self.root, "0", img_name + "_shadow1.png")
		shadow = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
		shadow = np.transpose(shadow, (2, 0, 1)).mean(axis=0)

		img_fname = os.path.join(self.root, "0", img_name + "_img.png")
		img = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
		img = np.transpose(img, (2, 0, 1))

		light_dir = torch.from_numpy(self.light_dir[img_name]).float().to(dev)
		return torch.from_numpy(img).float().to(dev), light_dir, torch.from_numpy(shadow).float().to(dev)

	def __len__(self):
		return len(self.light_dir)
