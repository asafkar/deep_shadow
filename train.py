# TODO https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html



import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

from tensorboardX import SummaryWriter

from datasets.ShadowData import ShadowDataset
from utils import *
from viewshed import *
from network import DepthNerf

import matplotlib
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn.functional as F


import numpy as np
import argparse
import kornia

matplotlib.use('pdf')
torch.manual_seed(1)

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-o', '--object', help='Description ', default='rose')
parser.add_argument('--speed', help='fast, medium or slow ', default='fast')
parser.add_argument('-lr', '--learning_rate', help='Description ', default=8e-5)
parser.add_argument('-cyclic_lr', help='Description ', default=False)
parser.add_argument('-cosine_lr', help='Description ', default=False)
parser.add_argument('-d', '--dev', help='device', required=False, default='cuda')
parser.add_argument('--model_dev', help='device', required=False, default='cuda')
parser.add_argument('--temp', help='temp of sigmoid', required=False, default=-25)
parser.add_argument('--epochs', help='epochs', required=False, type=int, default=1500)
parser.add_argument('--filter_size', help='latent var for nerf', required=False, default=128)
parser.add_argument('--num_enc_functions', help='encoding functions', required=False, default=5)
parser.add_argument('--checkpoint', help='Checkpoint to continue training from ',
    default=None)
parser.add_argument('-step_every_img', help='Description ', default=True)
parser.add_argument('--save_dir', help='dir to save models ', default='/fdata/asafk/models/')
parser.add_argument('--data_path', help='path to dataset ', default='./data/')
parser.add_argument('--alternative_light_uv_method', help='Description ', default=True)
parser.add_argument('-num_val', help='Description ', type=int, default=1)

args = parser.parse_args()
args.add_missing_points = False

task_name = f"shading C2F_depth_regl const {args.speed} {args.object} Step_gamma_0.9 {args.learning_rate} L2+L1 enc_func={args.num_enc_functions}" \
            f"filter_size={args.filter_size} learned_mult init to 10"
task_name = task_name.replace(" ", "_")

use_clearml = False

if use_clearml:
    from clearml import Task, Logger
    task = Task.init(project_name="nerf2D_shading", task_name=task_name)

writer = SummaryWriter(task_name)


print(" ---------------------------------------------------------- ")
print(f"Running: {task_name}")
print(" ---------------------------------------------------------- ")

args.use_no_light = False
args.train = False

scheme_1d = False
dev = args.dev

object = args.object  # "cube" #"snow_terrain"
train_path = f'{object}'

train_loader = ShadowDataset(root=os.path.join(args.data_path, train_path),
            args=args, train=True, num_val=args.num_val, normalize_lights=False)
c, h, w = train_loader[0][0].shape
args.w, args.h = w, h

train_loader_idx = np.arange(len(train_loader))
light_sources_xyz = torch.empty((len(train_loader), 3), device=dev)
shade_maps = torch.empty((len(train_loader), h, w), device=dev)
img_maps = torch.empty((len(train_loader), 3, h, w), device=dev)


# FIXME load with data from file
if w == 32:
    focal_length = 44.444
elif w == 64:
    focal_length = 88.888
elif w == 256:
    focal_length = 355.54
elif w==128:
    focal_length = 177.777
else:
    quit("no known focal length")

cam_location = (0, 0, 1)
img_plane_z = cam_location[2]  # FIXME - (focal_length / w)

# load all images and light sources into memory
for jj in train_loader_idx:
    target_img, target_light_src, target_shadow = train_loader[jj]
    light_sources_xyz[jj] = target_light_src.to(dev)
    shade_maps[jj] = target_shadow.to(dev)
    img_maps[jj] = target_img.to(dev)

img_mean = img_maps.mean(dim=0)
del img_maps

K_hom = get_camera_intrinsics(w, h, focal_length).to(args.model_dev)
K = K_hom[:, :3, :3]
RT = get_camera_extrinsics().to(args.model_dev)

light_sources_uv = torch.stack([pnt_world_coords_to_pixel_coords(l, K_hom, RT) for l in light_sources_xyz], dim=0)
depth_map = torch.from_numpy(train_loader.depth_exr).to(dev)

torch.manual_seed(1)
num_encoding_functions = args.num_enc_functions
depth_nerf = DepthNerf(num_encoding_functions, args=args).to(args.model_dev)

depth_params = set(depth_nerf.parameters())

criterion = torch.nn.L1Loss(reduction='none')
criterion2 = torch.nn.MSELoss(reduction='none')
optimizer = torch.optim.AdamW(depth_params, lr=args.learning_rate, weight_decay=1e-6)

if args.cyclic_lr:
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate*5, total_steps=args.epochs,
        cycle_momentum=False, final_div_factor=100, pct_start=0.1)
elif args.cosine_lr:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=1e-6, T_0=args.epochs // 5,
        T_mult=4)
else:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs / 25, gamma=0.9, verbose=False)

if args.checkpoint:
    print(f"Loading from checkpoint {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint)
    depth_nerf.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# TRAINING LOOP
pred_depth_map_l = []  # Results over time

# Speedup - draw coordinate lines to boundary points, and use
# result on all intermediate points.
# No speedup - generate coordinate lines to every pixel, calculate shading on
# that final pixel, use that result only in final shade map.
speed_up_calculation = True  #if args.speed == "fast" else False

# generate all positional encoded points for NeRF input
mesh_query_points = (get_ray_bundle(h, w, normalize=True).reshape((-1, 2))).to(dev)
mesh_query_points = mesh_query_points[:, 1].unsqueeze(1) if scheme_1d else mesh_query_points

all_depth_coords_encoded = [positional_encoding(mesh_query_points[jj, :], num_encoding_functions=num_encoding_functions)
    for jj in range(w * h)]
all_depth_coords_encoded = torch.stack(all_depth_coords_encoded).to(args.model_dev)
loss = 0.0
line_loss = False


def pre_gen_lines_arr(boundary_subsampling_factor):
    lines_arr = []
    num_points_on_square_boundary = min(w, h) / boundary_subsampling_factor
    boundary_points = generate_square_boundaries(top_left=(0, 0), bottom_right=(w - 1, h - 1),
        num_points=num_points_on_square_boundary)

    # all coordinate points (used instead of boundary points, if those aren't used)
    all_points = get_ray_bundle(h, w, normalize=False).reshape((-1, 2)).to(dev)
    points_to_calc = boundary_points if speed_up_calculation else all_points

    for idx in range(len(light_sources_uv)):  # go over all light sources
        # flip uv->vu
        curr_light_sources_vu = light_sources_uv[idx].flip(0)
        light_src_in_frame = (0 <= curr_light_sources_vu[1] <= w) and (0 <= curr_light_sources_vu[0] <= h)
        points_to_calc = points_to_calc.flip(1)

        if light_src_in_frame and args.alternative_light_uv_method:
            flat_idx = torch.argmin(torch.sqrt(
                ((light_sources_xyz[idx][:2].view(-1, 1, 1).expand(2, w, h) - xyz.squeeze()[:2]) ** 2).sum(0)))
            u_, v_ = torch.div(flat_idx, w, rounding_mode='floor'), flat_idx % h
            curr_light_sources_vu = torch.stack([u_, v_])

        lines_arr.append(gen_lines_from_src_to_points(points_to_calc.cpu(), curr_light_sources_vu.cpu(), w - 1, h - 1))

    return lines_arr


for iter in range(args.epochs):  # epochs
    # coarse to fine boundary sampling:
    if iter == 0:
        lines_arr = pre_gen_lines_arr(5)
    elif iter == args.epochs // 4:
        lines_arr = pre_gen_lines_arr(4)
    elif iter == args.epochs // 3:
        lines_arr = pre_gen_lines_arr(3)
    elif iter == args.epochs // 2:
        lines_arr = pre_gen_lines_arr(1)

    running_loss = 0.0
    for idx in range(len(light_sources_uv)):  # go over all light sources
        loss = 0.0
        res = torch.zeros_like(depth_map, device=dev, dtype=torch.float32)

        noise = torch.randn_like(all_depth_coords_encoded) * 0.0001
        # """ fast method - sample all depth points in advance"""
        depth_hat = depth_nerf(all_depth_coords_encoded + noise).reshape(w, h).to(dev)
        xyz = depth_map_to_pointcloud(depth_hat, K, RT, w, h).unsqueeze(0).permute(0, 3, 1, 2)

        # # compute the pointcloud spatial gradients
        # gradients: torch.Tensor = kornia.filters.spatial_gradient(xyz, mode='diff', order=1)  # Bx3x2xHxW
        # a, b = gradients[:, :, 0], gradients[:, :, 1]  # Bx3xHxW
        #
        # normals_kornia: torch.Tensor = torch.cross(a, b, dim=1).squeeze()  # 3xHxW
        # normals_kornia = F.normalize(normals_kornia, dim=0, p=2)
        # normals_for_loss = -1 * normals_kornia #.permute(1, 2, 0)

        lines = lines_arr[idx]

        curr_light_sources_vu = light_sources_uv[idx].flip(0)
        light_src_in_frame = (0 <= curr_light_sources_vu[1] <= w) and (0 <= curr_light_sources_vu[0] <= h)

        if line_loss:
            running_loss = 0
        for jdx, line in enumerate(lines):  # go over each line from src to boundary
            if light_sources_xyz[idx][2] > img_plane_z and not (light_src_in_frame and args.alternative_light_uv_method):  # if light is behind camera plane, reverse scan order
                line = line.flip(1)
            num_pts = line.shape[1]
            shaded_gt = shade_maps[idx]
            line_sample_coords = line.round().long()
            # sample shade map along coordinate line, nearest neighbor interpolation
            shaded_gt_line_sampled = shaded_gt.T[[*line_sample_coords]]

            pred_shading, real_shading = get_los_2d_worldcoords(light_sources_xyz[idx],
                line, xyz.squeeze(), temp=args.temp, eps=1e-4 if iter < 2000 else 1e-5,
                return_last_points_only=(not speed_up_calculation))

            if line_loss:
                # take loss between predicted shading along line and GT sampled shading
                loss = (criterion2(pred_shading, shaded_gt_line_sampled)).sum() \
                         + (criterion(pred_shading, shaded_gt_line_sampled)).sum()

                running_loss += loss

            else:
                # loss on whole image
                if speed_up_calculation:
                    res[[*line_sample_coords]] = pred_shading.float()
                else:
                    res[[*line_sample_coords[:, -1]]] = pred_shading

        if not line_loss:
            loss = criterion(shaded_gt, res).mean() + criterion2(shaded_gt, res).mean()
            # if iter > 200:
            loss += 0.1 * kornia.losses.inverse_depth_smoothness_loss(depth_hat.view(1, 1, w, h), img_mean.unsqueeze(0))
            # else:
            # if iter > 300:
            #     loss += 0.1 * (depth_hat - kornia.filters.box_blur(depth_hat.view(1, 1, w, h), (3, 3)).squeeze()).abs().mean()

            running_loss += loss

        # take backwards step on every image
        if args.step_every_img:
            optimizer.zero_grad(set_to_none=True)
            if line_loss:
                running_loss.backward()
            else:
                loss.backward()

            optimizer.step()
            # if iter > 200 and iter % 5 == 0:
            #     params_optimizer.step()

    if not args.step_every_img:
        # take backwards step after all images
        optimizer.zero_grad(set_to_none=True)

        running_loss.backward()
        optimizer.step()
        loss = 0.0

    scheduler.step()
    if use_clearml:
        Logger.current_logger().report_scalar(title="learning_rate", series="args", value=scheduler.get_last_lr()[0],
            iteration=iter)
        Logger.current_logger().report_scalar(title="loss", series="Accuracy", value=running_loss, iteration=iter)

    # every X iterations, print error between GT depth and predicted depth
    if iter % 50 == 0:

        with torch.no_grad():
            print(f"saving model iter {iter} in {args.save_dir}/{task_name}_{iter}_snapshot.pth")
            torch.save({
                'epoch': iter,
                'model_state_dict': depth_nerf.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            },
                os.path.join(args.save_dir, task_name + f"_{iter}_snapshot.pth"))

            if use_clearml:
                Logger.current_logger().report_scalar(title="sine_mult_factor", series="parameters", value=depth_nerf.factor, iteration=iter)
                Logger.current_logger().report_scalar(title="sine_bias", series="parameters", value=depth_nerf.bias, iteration=iter)
                Logger.current_logger().report_scalar(title="depth_output_max", series="parameters", value=depth_hat.max(), iteration=iter)
                Logger.current_logger().report_scalar(title="depth_output_min", series="parameters", value=depth_hat.min(), iteration=iter)

                grid_gt = torchvision.utils.make_grid([shade_maps[ii].cpu().unsqueeze(0) for ii in range(len(light_sources_uv))])
                Logger.current_logger().report_image("shading_gt", "image float", iteration=iter,
                    image=grid_gt.permute(1, 2, 0).cpu().numpy())

            # predicted depth
            pred_depth_map = depth_nerf(all_depth_coords_encoded).clamp(-1000, 1000).reshape(w, h).to(args.dev)
            err = torch.abs((pred_depth_map - depth_map)).cpu()

            if use_clearml:
                Logger.current_logger().report_scalar(title="depth absolute error", series="Accuracy", value=err.mean(), iteration=iter)
                err_scaled = torch.abs(pred_depth_map - depth_map) / (depth_map.max() - depth_map.min())
                Logger.current_logger().report_scalar(title="depth absolute error scaled by h_max_min", series="Accuracy",
                    value=err_scaled.mean().cpu() * 100, iteration=iter)

            normed_pred_depth_map = pred_depth_map - pred_depth_map.min()
            normed_pred_depth_map /= normed_pred_depth_map.max()

            normed_depth_map = depth_map - depth_map.min()
            normed_depth_map /= normed_depth_map.max()
            normed_depth_map = normed_depth_map

            if use_clearml:
                err_normed = torch.abs(((normed_pred_depth_map.reshape(w, h)) - normed_depth_map)).cpu()
                Logger.current_logger().report_image("normalized depth abs diff", "image float", iteration=iter,
                    image=err_normed.numpy())
                print(f"Normed Depth Prediction Error = {err_normed.mean()}")
                Logger.current_logger().report_scalar(title="normalized depth absolute error", series="Accuracy", value=err_normed.mean(), iteration=iter)

                Logger.current_logger().report_image("depth_GT normed", "image float", iteration=iter,
                    image=train_loader.depth_exr_normed)

                Logger.current_logger().report_image("depth pred normed", "image float", iteration=iter,
                    image=train_loader.silhouette_gt.cpu().numpy() * normed_pred_depth_map.cpu().numpy())

                fig1 = plt.figure(1)
                plt1 = plt.imshow(pred_depth_map.cpu().numpy(), cmap='hot')
                plt.colorbar(plt1)
                Logger.current_logger().report_matplotlib_figure(title="Pred depth heatmap",
                        series="depth", iteration=iter, figure=plt, report_image=True)
                fig1.clear()

                fig2 = plt.figure(2)
                plt2 =plt.imshow(train_loader.depth_exr, cmap='hot')
                plt.colorbar(plt2)
                Logger.current_logger().report_matplotlib_figure(title="GT depth heatmap",
                        series="depth", iteration=iter, figure=plt, report_image=True)
                fig2.clear()

            ###### calc all predicted shadings and save them to grid
            res_arr = []
            res_arr_real = []

            depth_hat = pred_depth_map.reshape(w, h)

            num_points_on_square_boundary = w
            boundary_points = generate_square_boundaries((0, 0), (w - 1, h - 1), num_points_on_square_boundary)
            all_points = get_ray_bundle(h, w, normalize=False).reshape((-1, 2)).to(dev)

            points_to_calc = boundary_points if speed_up_calculation else all_points

            for idx_sample in range(len(light_sources_uv)):
                shade_maps_real = []  # test

                # flip uv->vu
                curr_light_sources_vu = light_sources_uv[idx_sample].flip(0)
                light_src_in_frame = (0 <= curr_light_sources_vu[1] <= w) and (0 <= curr_light_sources_vu[0] <= h)
                points_to_calc = points_to_calc.flip(1)

                if light_src_in_frame and args.alternative_light_uv_method:
                    flat_idx = torch.argmin(torch.sqrt(
                        ((light_sources_xyz[idx_sample][:2].view(-1, 1, 1).expand(2, w, h) - xyz.squeeze()[:2]) ** 2).sum(0)))
                    u_, v_ = torch.div(flat_idx, w, rounding_mode='floor'), flat_idx % h
                    curr_light_sources_vu = torch.stack([u_, v_])

                # for light_src in light_sources:
                lines = gen_lines_from_src_to_points(points_to_calc.cpu(), curr_light_sources_vu.cpu(), w - 1,
                    h - 1)
                res = torch.zeros_like(depth_map, device=dev, dtype=torch.float32)
                res_real = torch.zeros_like(depth_map, device=dev, dtype=torch.float32)

                # add missing points
                if iter > args.epochs // 2:
                    dummy_res = torch.zeros_like(depth_map, device=dev, dtype=torch.float32)
                    all_lines = torch.cat(lines, dim=1).round().long()
                    dummy_res[[*all_lines]] += 1
                    missing_points = torch.stack((dummy_res == torch.Tensor([0]).to(dev)).nonzero(as_tuple=True)).T
                    missing_lines = gen_lines_from_src_to_points(missing_points.cpu(), curr_light_sources_vu.cpu(),
                        w - 1, h - 1)
                    lines = lines + missing_lines

                for idx, line in enumerate(lines):  # go over each line from src to boundary
                    if light_sources_xyz[idx_sample][2] > img_plane_z and not (light_src_in_frame and args.alternative_light_uv_method):  # if light is behind camera plane, reverse scan order
                        line = line.flip(1)
                    num_pts = line.shape[1]
                    if num_pts <= 1:  # boundary point, line of length 0
                        res[points_to_calc[idx][0].long(), points_to_calc[idx][1].long()] = 1
                        continue

                    line_sample_coords = line.round().long()
                    shaded_gt_line_sampled = depth_hat[[*line_sample_coords]]

                    pred_shading, real_shading = get_los_2d_worldcoords(light_sources_xyz[idx_sample],
                        line, xyz.squeeze(), temp=args.temp,
                        return_last_points_only=(not speed_up_calculation))

                    if speed_up_calculation:
                        res[[*line_sample_coords]] = pred_shading.float()
                    else:
                        res[[*line_sample_coords[:, -1]]] = pred_shading.float()

                res_arr.append(res.cpu().unsqueeze(0))

            if use_clearml:
                grid = torchvision.utils.make_grid(res_arr)
                Logger.current_logger().report_image("shading_hat sigmoid all angles", "image float", iteration=iter,
                    image=grid.permute(1, 2, 0).cpu().numpy())

                Logger.current_logger().report_image("shading difference all angles", "image float", iteration=iter,
                    image=(grid - grid_gt).abs().permute(1, 2, 0).cpu().numpy())

            ##### Compute predicted normals for this depth
            for ii, pred_depth in enumerate([pred_depth_map]):
                label = "regular" if ii == 0 else "normed"
                depth_img = pred_depth.reshape(w, h)
                xyz = depth_map_to_pointcloud(depth_img, K, RT, w, h).unsqueeze(0).permute(0, 3, 1, 2)

                # compute the pointcloud spatial gradients
                gradients: torch.Tensor = kornia.filters.spatial_gradient(xyz, mode='diff', order=1)  # Bx3x2xHxW

                # compute normals
                a, b = gradients[:, :, 0], gradients[:, :, 1]  # Bx3xHxW

                normals_kornia: torch.Tensor = torch.cross(a, b, dim=1).squeeze()  # 3xHxW
                normals_kornia = F.normalize(normals_kornia, dim=0, p=2)
                normals_kornia = -1 * normals_kornia.permute(1, 2, 0)

                angular_error, angular_error_sum = norm_diff(normals_kornia.permute(2, 0, 1), train_loader.normal_gt,
                    train_loader.silhouette_gt)

                if use_clearml:
                    Logger.current_logger().report_image(f"{label}/normal_from_depth", "image float", iteration=iter,
                        image=(train_loader.silhouette_gt.unsqueeze(2).cpu().numpy() * np.int16(((normals_kornia.cpu().numpy() + 1) / 2) * 255)))

                    Logger.current_logger().report_image(f"{label}/normal error map", "image float", iteration=iter,
                        image=angular_error.cpu().numpy())

                    Logger.current_logger().report_scalar(title=f"{label}/normals mean angle error", series="args",
                        value=angular_error_sum.cpu().numpy(), iteration=iter)

                if ii == 0:
                    print(f"Surface Normals Prediction Error = {angular_error_sum.cpu().numpy()}")

            if use_clearml:
                Logger.current_logger().report_image("normal GT", "image float", iteration=iter,
                    image=train_loader.silhouette_gt.unsqueeze(2).cpu().numpy() * 0.5*(train_loader.normal_gt+1).permute(1, 2, 0).cpu().numpy())

            # plot_pointcloud(xyz, iter)
    # profiler.step()
