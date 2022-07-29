import os
from torch.utils.tensorboard import SummaryWriter
import random

from datasets.ShadowData import ShadowDataset
from utils import norm_diff, positional_encoding, depth_map_to_pointcloud
from utils import get_camera_extrinsics, get_camera_intrinsics, get_ray_bundle, pnt_world_coords_to_pixel_coords
from viewshed import get_los_2d_worldcoords, gen_lines_from_src_to_points, generate_square_boundaries
from network import DepthNerf

import matplotlib
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import argparse
import kornia
import time


def prep_data_for_training(data_loader, args):
    train_loader_idx = np.arange(len(data_loader))
    light_sources_xyz = torch.empty((len(data_loader), 3), device=args.dev)
    shade_maps = torch.empty((len(data_loader), args.h, args.w), device=args.dev)
    img_maps = torch.empty((len(data_loader), 3, args.h, args.w), device=args.dev)

    # load all images and light sources into memory
    for jj in train_loader_idx:
        target_img, target_light_src, target_shadow = data_loader[jj]
        light_sources_xyz[jj] = target_light_src.to(args.dev)
        shade_maps[jj] = target_shadow.to(args.dev)
        img_maps[jj] = target_img.to(args.dev)

    img_mean = img_maps.mean(dim=0)
    del img_maps

    return shade_maps, light_sources_xyz, img_mean


def lights_in_frustrum(data_loader, light_sources_uv, light_sources_xyz, args):
    cam_location = (data_loader.params["cam_location_x"], data_loader.params["cam_location_y"], data_loader.params["cam_location_z"])
    light_sources_vu = light_sources_uv[:].flip(0)
    cam_location_z = cam_location[2]  # assumes cam at (0,0, Pz) - can be later generalized.

    lightsrc_in_frustrum = (light_sources_xyz[..., 2] > cam_location_z)

    for light_idx in range(len(data_loader)):
        curr_light_sources_vu = light_sources_vu[light_idx]
        light_src_in_frame = (0 <= curr_light_sources_vu[1] <= args.w) and (0 <= curr_light_sources_vu[0] <= args.h)
        lightsrc_in_frustrum[light_idx] *= (not light_src_in_frame)

    return lightsrc_in_frustrum


# if light source is above and "in" the image plane,
# find the coordinates of point in the pointcloud that is closest to the lightsrc, and use its index for uv
def nearest_pnt_to_light_src(light_source_xyz, xyz, args):
    flat_idx = torch.argmin(torch.sqrt(((light_source_xyz[:2].view(-1, 1, 1).expand(2, args.w, args.h)
                - xyz.squeeze()[:2]) ** 2).sum(0)))
    u_, v_ = torch.div(flat_idx, args.w, rounding_mode='floor'), flat_idx % args.h
    curr_light_sources_vu = torch.stack([u_, v_])
    return curr_light_sources_vu


def run(args, task_name):
    writer = SummaryWriter(task_name_)

    data_loader = ShadowDataset(root=os.path.join(args.data_path, args.object), args=args, normalize_lights=False)
    c, h, w = data_loader[0][0].shape
    args.w, args.h = w, h

    focal_length = data_loader.params["focal_length"]
    shade_maps, light_sources_xyz, img_mean = prep_data_for_training(data_loader, args)

    K_hom = get_camera_intrinsics(w, h, focal_length).to(args.dev)
    K = K_hom[:, :3, :3]
    RT = get_camera_extrinsics().to(args.dev)
    light_sources_uv = torch.stack([pnt_world_coords_to_pixel_coords(l, K_hom, RT) for l in light_sources_xyz], dim=0)
    depth_map = torch.from_numpy(data_loader.depth_exr).to(args.dev)

    lightsrc_in_frustrum = lights_in_frustrum(data_loader, light_sources_uv, light_sources_xyz, args)

    factor = data_loader.depth_exr.max()  # initialize model with maximum depth
    num_encoding_functions = args.num_enc_functions
    depth_nerf = DepthNerf(num_encoding_functions, factor=factor, args=args)
    if args.mixed:
        depth_nerf = depth_nerf.cuda()

    depth_params = set(depth_nerf.parameters())-set([depth_nerf.factor])  # don't optimize factor

    criterion = torch.nn.L1Loss()
    criterion2 = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(depth_params, lr=args.learning_rate, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.9, verbose=False)

    if args.checkpoint:
        print(f"Loading from checkpoint {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        depth_nerf.load_state_dict(checkpoint['model_state_dict'])

    # Speedup - draw coordinate lines to boundary points, and use result on all intermediate points.
    # No speedup - generate coordinate lines to every pixel, calculate shading on
    # that final pixel, use that result only in final shade map.
    speed_up_calculation = args.speed == "fast"

    # generate all positional encoded points for NeRF input
    mesh_query_points = (get_ray_bundle(h, w, normalize=True).reshape((-1, 2))).to(args.dev)

    all_depth_coords_encoded = [positional_encoding(mesh_query_points[jj, :], num_encoding_functions=num_encoding_functions)
        for jj in range(w * h)]
    all_depth_coords_encoded = torch.stack(all_depth_coords_encoded).to(args.dev)

    # pre-calculate and cache all lines for sampling the predicted depth, since these are expensive calculations
    def pre_gen_lines_arr(boundary_subsampling_factor, xyz):
        lines_arr = []
        boundary_points = generate_square_boundaries(top_left=(0, 0), bottom_right=(w - 1, h - 1),
                                                        boundary_subsampling_factor=boundary_subsampling_factor)

        # all coordinate points (used instead of boundary points, if those aren't used)
        all_points = get_ray_bundle(h, w, normalize=False).reshape((-1, 2)).to(args.dev)
        points_to_calc = boundary_points if speed_up_calculation else all_points

        for idx in range(len(light_sources_uv)):  # go over all light sources
            # flip uv->vu
            curr_light_sources_vu = light_sources_uv[idx].flip(0)
            light_src_in_frame = (0 <= curr_light_sources_vu[1] <= w) and (0 <= curr_light_sources_vu[0] <= h)
            points_to_calc = points_to_calc.flip(1)

            if light_src_in_frame:
                curr_light_sources_vu = nearest_pnt_to_light_src(light_sources_xyz[idx], xyz, args)

            lines_arr.append(gen_lines_from_src_to_points(points_to_calc.cpu(), curr_light_sources_vu.cpu(), w - 1, h - 1))
        return lines_arr

    # boundary sampling scheme
    if args.boundary_sampling:
        iter_sampling_pairs = {
                                0: 4,
                                args.epochs // 4: 3,
                                args.epochs // 3: 2,
                                args.epochs // 2: 1,
                                }
    else:
        iter_sampling_pairs = {0: 1}  # dense boundary sampling from first iteration

    def train_single_light_src(light_idx, depth_nerf, optimizer, running_loss):
        shadow_hat = torch.zeros_like(depth_map, device=args.dev, dtype=torch.float32)
        noise = torch.randn_like(all_depth_coords_encoded) * 0.0001
        if args.mixed:
            depth_hat = depth_nerf((all_depth_coords_encoded + noise).cuda()).reshape(w, h).cpu()
        else:
            depth_hat = depth_nerf((all_depth_coords_encoded + noise)).reshape(w, h)

        xyz = depth_map_to_pointcloud(depth_hat, K, RT, w, h).unsqueeze(0).permute(0, 3, 1, 2)

        lines = lines_arr[light_idx]

        for jdx, line in enumerate(lines):  # go over each line from src to boundary
            # if light is behind camera plane, reverse scan order
            if lightsrc_in_frustrum[light_idx]:
                line = line.flip(1)
            shaded_gt = shade_maps[light_idx]
            line_sample_coords = line.round().long()

            pred_shading, real_shading = get_los_2d_worldcoords(light_sources_xyz[light_idx],
                line, xyz.squeeze(), temp=args.temp, eps=1e-4 if iter < (args.epochs * 0.8) else 1e-5,
                return_last_points_only=(not speed_up_calculation))

            # sample shade map along coordinate line, nearest neighbor interpolation
            if speed_up_calculation:
                shadow_hat[[*line_sample_coords]] = pred_shading.float()
            else:
                shadow_hat[[*line_sample_coords[:, -1]]] = pred_shading

        loss = criterion(shaded_gt, shadow_hat) + criterion2(shaded_gt, shadow_hat)
        loss += 0.1 * kornia.losses.inverse_depth_smoothness_loss(depth_hat.view(1, 1, w, h), img_mean.unsqueeze(0))
        running_loss += loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        return running_loss

    def test(depth_hat, writer):

        with torch.no_grad():
            print(f"saving model iter {iter} in {args.save_dir}/{task_name}_{iter}_snapshot.pth")
            torch.save({
                'epoch': iter,
                'model_state_dict': depth_nerf.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            },
                os.path.join(args.save_dir, task_name + f"_{iter}_snapshot.pth"))

            xyz = depth_map_to_pointcloud(depth_hat, K, RT, w, h).unsqueeze(0).permute(0, 3, 1, 2)
            err = torch.abs((depth_hat - depth_map)).cpu()

            normed_pred_depth_map = depth_hat - depth_hat.min()
            normed_pred_depth_map /= normed_pred_depth_map.max()

            normed_depth_map = depth_map - depth_map.min()
            normed_depth_map /= normed_depth_map.max()
            normed_depth_map = normed_depth_map

            writer.add_scalar('Model/max_depth_value', depth_hat.max(), iter)
            writer.add_scalar('Model/min_depth_value', depth_hat.min(), iter)

            grid_gt = torchvision.utils.make_grid([shade_maps[ii].cpu().unsqueeze(0) for ii in range(len(light_sources_uv))])
            writer.add_image("Shadow_GT", grid_gt.cpu().numpy(), iter)
            writer.add_scalar('depth absolute error', err.mean(), iter)

            err_normed = torch.abs(((normed_pred_depth_map.reshape(w, h)) - normed_depth_map)).cpu()
            print(f"Normed Depth Prediction Error = {err_normed.mean()}")
            writer.add_image('normalized depth abs diff', err_normed.unsqueeze(0).numpy(), iter)
            writer.add_image('depth_GT normed', data_loader.depth_exr_normed[np.newaxis, :, :], iter)
            writer.add_image('depth pred normed', (data_loader.silhouette_gt.cpu().numpy()
                              * normed_pred_depth_map.cpu().numpy())[np.newaxis, :, :], iter)
            writer.add_scalar('normalized depth absolute error', err_normed.mean(), iter)

            if args.use_clearml:
                fig1 = plt.figure(1)
                plt1 = plt.imshow(depth_hat.cpu().numpy(), cmap='hot')
                plt.colorbar(plt1)
                Logger.current_logger().report_matplotlib_figure(title="Pred depth heatmap",
                        series="depth", iteration=iter, figure=plt, report_image=True)
                fig1.clear()

                fig2 = plt.figure(2)
                plt2 = plt.imshow(data_loader.depth_exr, cmap='hot')
                plt.colorbar(plt2)
                Logger.current_logger().report_matplotlib_figure(title="GT depth heatmap",
                        series="depth", iteration=iter, figure=plt, report_image=True)
                fig2.clear()

            # calculate all predicted shadows and save them to grid
            shadows_hat_arr = []
            num_points_on_square_boundary = w
            boundary_points = generate_square_boundaries((0, 0), (w - 1, h - 1), num_points_on_square_boundary)
            all_points = get_ray_bundle(h, w, normalize=False).reshape((-1, 2)).to(args.dev)

            points_to_calc = boundary_points if speed_up_calculation else all_points

            for idx_sample in range(len(light_sources_uv)):
                # flip uv->vu
                curr_light_sources_vu = light_sources_uv[idx_sample].flip(0)
                light_src_in_frame = (0 <= curr_light_sources_vu[1] <= w) and (0 <= curr_light_sources_vu[0] <= h)
                points_to_calc = points_to_calc.flip(1)

                if light_src_in_frame:
                    curr_light_sources_vu = nearest_pnt_to_light_src(light_sources_xyz[idx_sample], xyz, args)

                # for light_src in light_sources:
                lines = gen_lines_from_src_to_points(points_to_calc.cpu(), curr_light_sources_vu.cpu(), w - 1,
                    h - 1)
                shadow_hat = torch.zeros_like(depth_map, device=args.dev, dtype=torch.float32)

                # add missing points
                if iter > args.epochs // 2:
                    dummy_res = torch.zeros_like(depth_map, device=args.dev, dtype=torch.float32)
                    all_lines = torch.cat(lines, dim=1).round().long()
                    dummy_res[[*all_lines]] += 1
                    missing_points = torch.stack((dummy_res == torch.Tensor([0]).to(args.dev)).nonzero(as_tuple=True)).T
                    missing_lines = gen_lines_from_src_to_points(missing_points.cpu(), curr_light_sources_vu.cpu(),
                        w - 1, h - 1)
                    lines = lines + missing_lines

                for idx, line in enumerate(lines):  # go over each line from src to boundary
                    # if light is behind camera plane, reverse scan order
                    if lightsrc_in_frustrum[idx_sample]:
                        line = line.flip(1)
                    num_pts = line.shape[1]
                    if num_pts <= 1:  # boundary point, line of length 0
                        shadow_hat[points_to_calc[idx][0].long(), points_to_calc[idx][1].long()] = 1
                        continue

                    line_sample_coords = line.round().long()
                    pred_shading, real_shading = get_los_2d_worldcoords(light_sources_xyz[idx_sample],
                        line, xyz.squeeze(), temp=args.temp,
                        return_last_points_only=(not speed_up_calculation))

                    if speed_up_calculation:
                        shadow_hat[[*line_sample_coords]] = pred_shading.float()
                    else:
                        shadow_hat[[*line_sample_coords[:, -1]]] = pred_shading.float()

                shadows_hat_arr.append(shadow_hat.cpu().unsqueeze(0))

            grid = torchvision.utils.make_grid(shadows_hat_arr)
            writer.add_image('predicted_shadows', grid.cpu().numpy(), iter)
            writer.add_image('shadow_error', (grid - grid_gt).abs().cpu().numpy(), iter)

            # Compute predicted normals for this depth
            depth_img = depth_hat.reshape(w, h)
            xyz = depth_map_to_pointcloud(depth_img, K, RT, w, h).unsqueeze(0).permute(0, 3, 1, 2)

            # compute the pointcloud spatial gradients
            gradients: torch.Tensor = kornia.filters.spatial_gradient(xyz, mode='diff', order=1)  # Bx3x2xHxW

            # compute normals
            a, b = gradients[:, :, 0], gradients[:, :, 1]  # Bx3xHxW

            normals_kornia: torch.Tensor = torch.cross(a, b, dim=1).squeeze()  # 3xHxW
            normals_kornia = F.normalize(normals_kornia, dim=0, p=2)
            normals_kornia = -1 * normals_kornia.permute(1, 2, 0)

            angular_error, angular_error_sum = norm_diff(normals_kornia.permute(2, 0, 1), data_loader.normal_gt,
                data_loader.silhouette_gt)

            writer.add_image('predicted_normals',
                (data_loader.silhouette_gt.unsqueeze(0).cpu().numpy()
                 * ((normals_kornia.cpu().numpy().transpose(2, 0, 1) + 1) / 2)), iter)

            writer.add_image('normals_err_map', angular_error.cpu().numpy()[np.newaxis, ...], iter)  # fixme!!
            writer.add_image('normals_GT', data_loader.silhouette_gt.unsqueeze(0).cpu().numpy()
                    * 0.5 * (data_loader.normal_gt + 1).cpu().numpy(), iter)

            writer.add_scalar('normals mean angle error', angular_error_sum.cpu().numpy(), iter)
            print(f"Surface Normals Prediction Error = {angular_error_sum.cpu().numpy()}")

    for iter in range(args.epochs):  # epochs
        epoch_start_time = time.time()
        if iter < 100:
            args.temp = -10
        elif iter < 300:
            args.temp = -30
        elif iter < 500:
            args.temp = -50
        else:
            args.temp = -80

        # coarse to fine boundary sampling:
        if iter in iter_sampling_pairs.keys():
            noise = torch.randn_like(all_depth_coords_encoded) * 0.0001

            if args.mixed:
                depth_hat = depth_nerf((all_depth_coords_encoded + noise).cuda()).reshape(w, h).to("cpu")
            else:
                depth_hat = depth_nerf((all_depth_coords_encoded + noise)).reshape(w, h)

            xyz = depth_map_to_pointcloud(depth_hat, K, RT, w, h).unsqueeze(0).permute(0, 3, 1, 2)
            lines_arr = pre_gen_lines_arr(iter_sampling_pairs[iter], xyz)

        running_loss = 0.0
        for idx in range(len(light_sources_uv)):  # go over all light sources
            running_loss += train_single_light_src(idx, depth_nerf, optimizer, running_loss)

        scheduler.step()

        writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], iter)
        writer.add_scalar('Train_loss', running_loss, iter)

        epoch_end_train_time = time.time()
        # print(f"epoch {iter} train time took {epoch_end_train_time - epoch_start_time}")
        if iter % 10 == 0:
            print(f"epoch {iter} running loss = {running_loss}")

        # every X iterations, print error between GT depth and predicted depth
        if iter % 50 == 0:
            if args.mixed:
                depth_hat = depth_nerf(all_depth_coords_encoded.cuda()).reshape(w, h).cpu()
            else:
                depth_hat = depth_nerf(all_depth_coords_encoded).reshape(w, h)
            test(depth_hat, writer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-o', '--object', help='Description ', default='cactus')
    parser.add_argument('--speed', help='fast, medium or slow ', default='fast')
    parser.add_argument('-lr', '--learning_rate', help='Description ', default=5e-5)
    parser.add_argument('-d', '--dev', help='device: cpu, cuda or mixed', required=False, default='mixed')
    parser.add_argument('--temp', help='temp of sigmoid', required=False, default=-25)
    parser.add_argument('--epochs', help='epochs', required=False, type=int, default=1300)
    parser.add_argument('--filter_size', help='latent var for nerf', required=False, default=128)
    parser.add_argument('--num_enc_functions', help='encoding functions', required=False, default=5)
    parser.add_argument('--checkpoint', help='Checkpoint to continue training from', default=None)
    parser.add_argument('--save_dir', help='dir to save models ', default='/tmp/deep_shadow/models/')
    parser.add_argument('--data_path', help='path to dataset ', default='./data/')
    parser.add_argument('--boundary_sampling', help='Sub-sample image boundary ', default=False)
    parser.add_argument('--use_clearml', help='Use clear-ml for logging ', default=False)

    args_ = parser.parse_args()

    task_name_ = f"{args_.object} LR={args_.learning_rate} dev={args_.dev}"
    task_name_ = task_name_.replace(" ", "_")

    if args_.dev == "mixed":
        args_.dev = "cpu"
        args_.mixed = True
    else:
        args_.mixed = False

    if not os.path.isdir(args_.save_dir):
        os.makedirs(args_.save_dir, mode=0o777)

    if args_.use_clearml:
        from clearml import Task, Logger
        task = Task.init(project_name="nerf2D_shading", task_name=task_name_)
        matplotlib.use('pdf')

    # seed everything
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    print(" ---------------------------------------------------------- ")
    print(f"Running: {task_name_}")
    print(" ---------------------------------------------------------- ")

    run(args_, task_name_)