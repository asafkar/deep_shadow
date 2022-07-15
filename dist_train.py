# TODO https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
# WIP should be fine-tuned

import os
from tensorboardX import SummaryWriter
import random

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
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def run(rank, args, task_name, use_clearml):
    # seed everything
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    if use_clearml and rank == 0:
        from clearml import Task, Logger
        task = Task.init(project_name="nerf2D_shading", task_name=task_name)
        matplotlib.use('pdf')

    print(f"running from rank = {rank}")
    world_size = args.world_size

    # setup the process groups
    setup(rank, world_size)

    dev = args.dev
    train_loader = ShadowDataset(root=os.path.join(args.data_path, args.object), args=args, normalize_lights=False)
    c, h, w = train_loader[0][0].shape
    args.w, args.h = w, h

    all_light_idx = np.arange(len(train_loader))
    relevant_light_idx = np.array_split(all_light_idx, args.world_size)[rank]
    # print(f"rank={rank} relevant_light_idx={relevant_light_idx}")

    light_sources_xyz = torch.empty((len(relevant_light_idx), 3), device=dev)
    shade_maps = torch.empty((len(relevant_light_idx), h, w), device=dev)
    img_maps = torch.empty((len(train_loader), 3, h, w), device=dev)

    focal_length = train_loader.params["focal_length"]

    cam_location = (train_loader.params["cam_location_x"], train_loader.params["cam_location_y"], train_loader.params["cam_location_z"])
    img_plane_z = cam_location[2]  # FIXME - (focal_length / w)

    # TODO DDP DataLoader?
    # load all images and light sources into memory
    ii = 0
    for jj in all_light_idx:
        target_img, target_light_src, target_shadow = train_loader[jj]
        if jj in relevant_light_idx:
            light_sources_xyz[ii] = target_light_src.to(dev)
            shade_maps[ii] = target_shadow.to(dev)
            ii += 1
        img_maps[jj] = target_img.to(dev)

    img_mean = img_maps.mean(dim=0)
    del img_maps

    K_hom = get_camera_intrinsics(w, h, focal_length).to(args.dev)
    K = K_hom[:, :3, :3]
    RT = get_camera_extrinsics().to(args.dev)

    light_sources_uv = torch.stack([pnt_world_coords_to_pixel_coords(l, K_hom, RT) for l in light_sources_xyz], dim=0)
    depth_map = torch.from_numpy(train_loader.depth_exr).to(dev)

    num_encoding_functions = args.num_enc_functions
    depth_nerf = DepthNerf(num_encoding_functions, args=args).to(args.dev)
    depth_nerf = DDP(depth_nerf)

    depth_params = set(depth_nerf.parameters())

    criterion = torch.nn.L1Loss(reduction='none')
    criterion2 = torch.nn.MSELoss(reduction='none')
    optimizer = torch.optim.AdamW(depth_params, lr=args.learning_rate, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs / 25, gamma=0.9, verbose=False)

    if args.checkpoint:
        print(f"Loading from checkpoint {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        depth_nerf.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Speedup - draw coordinate lines to boundary points, and use result on all intermediate points.
    # No speedup - generate coordinate lines to every pixel, calculate shading on
    # that final pixel, use that result only in final shade map.
    speed_up_calculation = args.speed == "fast"

    # generate all positional encoded points for NeRF input
    mesh_query_points = (get_ray_bundle(h, w, normalize=True).reshape((-1, 2))).to(dev)

    all_depth_coords_encoded = [positional_encoding(mesh_query_points[jj, :], num_encoding_functions=num_encoding_functions)
        for jj in range(w * h)]
    all_depth_coords_encoded = torch.stack(all_depth_coords_encoded).to(args.dev)

    # Whether to calculate loss over each line, or accumulate all lines and then calculate the loss
    line_loss = False

    # pre-calculate and cache all lines for sampling the predicted depth, since these are expensive calculations
    def pre_gen_lines_arr(boundary_subsampling_factor, xyz):
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

    # boundary sampling scheme
    iter_sampling_pairs = {
                            0: 5,
                            args.epochs // 4: 4,
                            args.epochs // 3: 3,
                            args.epochs // 2: 1,
                            }

    def train_single_light_src(light_idx, depth_nerf, optimizer, running_loss):
        shadow_hat = torch.zeros_like(depth_map, device=dev, dtype=torch.float32)

        if args.step_every_img:
            noise = torch.randn_like(all_depth_coords_encoded) * 0.0001
            depth_hat = depth_nerf(all_depth_coords_encoded + noise).reshape(w, h)
            xyz = depth_map_to_pointcloud(depth_hat, K, RT, w, h).unsqueeze(0).permute(0, 3, 1, 2)

        lines = lines_arr[light_idx]

        curr_light_sources_vu = light_sources_uv[light_idx].flip(0)
        light_src_in_frame = (0 <= curr_light_sources_vu[1] <= w) and (0 <= curr_light_sources_vu[0] <= h)

        if line_loss:
            running_loss = 0
        for jdx, line in enumerate(lines):  # go over each line from src to boundary
            if light_sources_xyz[light_idx][2] > img_plane_z and not (
                        light_src_in_frame and args.alternative_light_uv_method):  # if light is behind camera plane, reverse scan order
                line = line.flip(1)
            shaded_gt = shade_maps[light_idx]
            line_sample_coords = line.round().long()
            # sample shade map along coordinate line, nearest neighbor interpolation
            shaded_gt_line_sampled = shaded_gt.T[[*line_sample_coords]]

            pred_shading, real_shading = get_los_2d_worldcoords(light_sources_xyz[light_idx],
                line, xyz.squeeze(), temp=args.temp, eps=1e-4 if iter < (args.epochs * 0.8) else 1e-5,
                return_last_points_only=(not speed_up_calculation))

            if line_loss:
                # take loss between predicted shading along line and GT sampled shading
                loss = (criterion2(pred_shading, shaded_gt_line_sampled)).sum() \
                       + (criterion(pred_shading, shaded_gt_line_sampled)).sum()
                running_loss += loss

            else:  # loss on whole image
                if speed_up_calculation:
                    shadow_hat[[*line_sample_coords]] = pred_shading.float()
                else:
                    shadow_hat[[*line_sample_coords[:, -1]]] = pred_shading

        if not line_loss:
            loss = criterion(shaded_gt, shadow_hat).mean() + criterion2(shaded_gt, shadow_hat).mean()
            loss += 0.1 * kornia.losses.inverse_depth_smoothness_loss(depth_hat.view(1, 1, w, h), img_mean.unsqueeze(0))
            running_loss += loss

        # take backwards step on every image
        if args.step_every_img:
            optimizer.zero_grad(set_to_none=True)
            if line_loss:
                running_loss.backward()
            else:
                loss.backward()

            optimizer.step()

        return running_loss

    def test(depth_hat):
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

            if use_clearml:
                Logger.current_logger().report_scalar(title="sine_mult_factor", series="parameters",
                    value=depth_nerf.module.factor, iteration=iter)
                # Logger.current_logger().report_scalar(title="sine_bias", series="parameters", value=depth_nerf.bias, iteration=iter)
                Logger.current_logger().report_scalar(title="depth_output_max", series="parameters",
                    value=depth_hat.max(), iteration=iter)
                Logger.current_logger().report_scalar(title="depth_output_min", series="parameters",
                    value=depth_hat.min(), iteration=iter)

                grid_gt = torchvision.utils.make_grid(
                    [shade_maps[ii].cpu().unsqueeze(0) for ii in range(len(light_sources_uv))])
                Logger.current_logger().report_image("shading_gt", "image float", iteration=iter,
                    image=grid_gt.permute(1, 2, 0).cpu().numpy())

                Logger.current_logger().report_scalar(title="depth absolute error", series="Accuracy", value=err.mean(),
                    iteration=iter)
                err_scaled = torch.abs(depth_hat - depth_map) / (depth_map.max() - depth_map.min())
                Logger.current_logger().report_scalar(title="depth absolute error scaled by h_max_min",
                    series="Accuracy",
                    value=err_scaled.mean().cpu() * 100, iteration=iter)

                err_normed = torch.abs(((normed_pred_depth_map.reshape(w, h)) - normed_depth_map)).cpu()
                Logger.current_logger().report_image("normalized depth abs diff", "image float", iteration=iter,
                    image=err_normed.numpy())
                print(f"Normed Depth Prediction Error = {err_normed.mean()}")
                Logger.current_logger().report_scalar(title="normalized depth absolute error", series="Accuracy",
                    value=err_normed.mean(), iteration=iter)

                Logger.current_logger().report_image("depth_GT normed", "image float", iteration=iter,
                    image=train_loader.depth_exr_normed)

                Logger.current_logger().report_image("depth pred normed", "image float", iteration=iter,
                    image=train_loader.silhouette_gt.cpu().numpy() * normed_pred_depth_map.cpu().numpy())

                fig1 = plt.figure(1)
                plt1 = plt.imshow(depth_hat.cpu().numpy(), cmap='hot')
                plt.colorbar(plt1)
                Logger.current_logger().report_matplotlib_figure(title="Pred depth heatmap",
                    series="depth", iteration=iter, figure=plt, report_image=True)
                fig1.clear()

                fig2 = plt.figure(2)
                plt2 = plt.imshow(train_loader.depth_exr, cmap='hot')
                plt.colorbar(plt2)
                Logger.current_logger().report_matplotlib_figure(title="GT depth heatmap",
                    series="depth", iteration=iter, figure=plt, report_image=True)
                fig2.clear()

            ###### calc all predicted shadings and save them to grid
            shadows_hat_arr = []
            num_points_on_square_boundary = w
            boundary_points = generate_square_boundaries((0, 0), (w - 1, h - 1), num_points_on_square_boundary)
            all_points = get_ray_bundle(h, w, normalize=False).reshape((-1, 2)).to(dev)

            points_to_calc = boundary_points if speed_up_calculation else all_points

            for idx_sample in range(len(light_sources_uv)):
                # flip uv->vu
                curr_light_sources_vu = light_sources_uv[idx_sample].flip(0)
                light_src_in_frame = (0 <= curr_light_sources_vu[1] <= w) and (0 <= curr_light_sources_vu[0] <= h)
                points_to_calc = points_to_calc.flip(1)

                if light_src_in_frame and args.alternative_light_uv_method:
                    flat_idx = torch.argmin(torch.sqrt(
                        ((light_sources_xyz[idx_sample][:2].view(-1, 1, 1).expand(2, w, h) - xyz.squeeze()[
                        :2]) ** 2).sum(0)))
                    u_, v_ = torch.div(flat_idx, w, rounding_mode='floor'), flat_idx % h
                    curr_light_sources_vu = torch.stack([u_, v_])

                # for light_src in light_sources:
                lines = gen_lines_from_src_to_points(points_to_calc.cpu(), curr_light_sources_vu.cpu(), w - 1,
                    h - 1)
                shadow_hat = torch.zeros_like(depth_map, device=dev, dtype=torch.float32)

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
                    if light_sources_xyz[idx_sample][2] > img_plane_z and not (
                                light_src_in_frame and args.alternative_light_uv_method):  # if light is behind camera plane, reverse scan order
                        line = line.flip(1)
                    num_pts = line.shape[1]
                    if num_pts <= 1:  # boundary point, line of length 0
                        shadow_hat[points_to_calc[idx][0].long(), points_to_calc[idx][1].long()] = 1
                        continue

                    line_sample_coords = line.round().long()
                    shaded_gt_line_sampled = depth_hat[[*line_sample_coords]]

                    pred_shading, real_shading = get_los_2d_worldcoords(light_sources_xyz[idx_sample],
                        line, xyz.squeeze(), temp=args.temp,
                        return_last_points_only=(not speed_up_calculation))

                    if speed_up_calculation:
                        shadow_hat[[*line_sample_coords]] = pred_shading.float()
                    else:
                        shadow_hat[[*line_sample_coords[:, -1]]] = pred_shading.float()

                shadows_hat_arr.append(shadow_hat.cpu().unsqueeze(0))

            if use_clearml:
                grid = torchvision.utils.make_grid(shadows_hat_arr)
                Logger.current_logger().report_image("shading_hat sigmoid all angles", "image float", iteration=iter,
                    image=grid.permute(1, 2, 0).cpu().numpy())

                Logger.current_logger().report_image("shading difference all angles", "image float", iteration=iter,
                    image=(grid - grid_gt).abs().permute(1, 2, 0).cpu().numpy())

            ##### Compute predicted normals for this depth
            for ii, pred_depth in enumerate([depth_hat]):
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
                        image=(train_loader.silhouette_gt.unsqueeze(2).cpu().numpy() * np.int16(
                            ((normals_kornia.cpu().numpy() + 1) / 2) * 255)))

                    Logger.current_logger().report_image(f"{label}/normal error map", "image float", iteration=iter,
                        image=angular_error.cpu().numpy())

                    Logger.current_logger().report_scalar(title=f"{label}/normals mean angle error", series="args",
                        value=angular_error_sum.cpu().numpy(), iteration=iter)

                if ii == 0:
                    print(f"Surface Normals Prediction Error = {angular_error_sum.cpu().numpy()}")

            if use_clearml:
                Logger.current_logger().report_image("normal GT", "image float", iteration=iter,
                    image=train_loader.silhouette_gt.unsqueeze(2).cpu().numpy() * 0.5 * (
                                    train_loader.normal_gt + 1).permute(1, 2, 0).cpu().numpy())

            # plot_pointcloud(xyz, iter)

    for iter in range(args.epochs):  # epochs
        epoch_start_time = time.time()

        # if accumulating grads (take loss after all images, predict depth_hat only once per epoch)
        if (not args.step_every_img) or (iter in iter_sampling_pairs.keys()):
            noise = torch.randn_like(all_depth_coords_encoded) * 0.0001
            depth_hat = depth_nerf(all_depth_coords_encoded + noise).reshape(w, h)
            xyz = depth_map_to_pointcloud(depth_hat, K, RT, w, h).unsqueeze(0).permute(0, 3, 1, 2)

        # coarse to fine boundary sampling:
        if iter in iter_sampling_pairs.keys():
            lines_arr = pre_gen_lines_arr(iter_sampling_pairs[iter], xyz)

        running_loss = 0.0

        for idx in range(len(light_sources_uv)):  # go over all light sources
            running_loss += train_single_light_src(idx, depth_nerf, optimizer, running_loss)

        if not args.step_every_img:
            # take backwards step after all images
            optimizer.zero_grad(set_to_none=True)

            running_loss.backward()
            optimizer.step()

        scheduler.step()
        if use_clearml and rank == 0:
            Logger.current_logger().report_scalar(title="learning_rate", series="args", value=scheduler.get_last_lr()[0],
                iteration=iter)
            Logger.current_logger().report_scalar(title="loss", series="Accuracy", value=running_loss, iteration=iter)

        epoch_end_train_time = time.time()
        print(f"epoch {iter} train time took {epoch_end_train_time - epoch_start_time}")
        # print(f"epoch {iter} running loss = {running_loss}")

        # dist.barrier()
        # every X iterations, print error between GT depth and predicted depth
        if iter % 50 == 0 and rank == 0:
            depth_hat = depth_nerf(all_depth_coords_encoded).reshape(w, h)
            test(depth_hat)
            # profiler.step()

    def cleanup():
        dist.destroy_process_group()

    cleanup()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-o', '--object', help='Description ', default='rose')
    parser.add_argument('--speed', help='fast, medium or slow ', default='fast')
    parser.add_argument('-lr', '--learning_rate', help='Description ', default=8e-5)
    parser.add_argument('-d', '--dev', help='device', required=False, default='cpu')
    parser.add_argument('--temp', help='temp of sigmoid', required=False, default=-25)
    parser.add_argument('--epochs', help='epochs', required=False, type=int, default=1000)
    parser.add_argument('--filter_size', help='latent var for nerf', required=False, default=128)
    parser.add_argument('--num_enc_functions', help='encoding functions', required=False, default=5)
    parser.add_argument('--checkpoint', help='Checkpoint to continue training from', default=None)
    parser.add_argument('--step_every_img', help='Description ', default=True)
    parser.add_argument('--save_dir', help='dir to save models ', default='/tmp/deep_shadow/models/')
    parser.add_argument('--data_path', help='path to dataset ', default='./data/')
    parser.add_argument('--alternative_light_uv_method', help='Description ', default=True)
    parser.add_argument('--world_size', help='Description ', type=int, default=1)
    args_ = parser.parse_args()

    task_name_ = f"DISTRIBUTED_num_nodes{args_.world_size} shading dev={args_.dev} {args_.object} Step_gamma_0.9 {args_.learning_rate} L2+L1 enc_func={args_.num_enc_functions}" \
                f"filter_size={args_.filter_size} learned_mult init to 10"
    task_name_ = task_name_.replace(" ", "_")

    use_clearml_ = True
    writer = SummaryWriter(task_name_)

    print(" ---------------------------------------------------------- ")
    print(f"Running: {task_name_}")
    print(" ---------------------------------------------------------- ")

    mp.spawn(run,
             args=(args_, task_name_, use_clearml_,),
             nprocs=args_.world_size,
             join=True)