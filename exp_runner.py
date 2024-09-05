import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer
import matplotlib.pyplot as plt
import sys
from models.embedder import get_embedder

class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'])
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = SummaryWriter(self.base_exp_dir)

        # Networks
        params_to_train = []
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     **self.conf['model.neus_renderer'])

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

    def train(self):
        # self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()

        for iter_i in tqdm(range(res_step)):
            data = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size)

            rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)


            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)

            mask_sum = mask.sum() + 1e-5
            render_out = self.renderer.render(rays_o, rays_d, near, far, self.iter_step,
                                              background_rgb=background_rgb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio())

            color_fine = render_out['color_fine']
            uncert_fine = render_out['uncert_fine']
            s_val = render_out['s_val']
            cdf_fine = render_out['cdf_fine']
            gradient_error = render_out['gradient_error']
            weight_max = render_out['weight_max']
            weight_sum = render_out['weight_sum']
            s_loss = render_out['s_loss']

            rggbmask = true_rgb.clone()
            rggbmask[rggbmask > 0] = 1
            color_error = (torch.mul(color_fine, rggbmask) - true_rgb) * mask
            

            color_fine_loss =torch.mean((1 / (2*(uncert_fine+1e-9).unsqueeze(-1))) *((color_error) ** 2)) + 0.5*torch.mean(torch.log(uncert_fine+1e-9))

            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            eikonal_loss = gradient_error

            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

            if self.iter_step < 100001:
                s_loss_weight = 0.1
            else:
                s_loss_weight = 0.0

            loss = color_fine_loss +\
                   eikonal_loss * self.igr_weight +\
                   mask_loss * self.mask_weight +\
                   s_loss * s_loss_weight


            if self.iter_step > 100001:
                snumber = 256
                sphi = np.random.uniform(0, 2 * np.pi, snumber)
                stheta = np.arccos(np.random.uniform(-1, 1, snumber))
                su = np.random.uniform(0, 1, snumber)

                r = su ** (1 / 3)
                x = (r * np.sin(stheta) * np.cos(sphi))
                y = (r * np.sin(stheta) * np.sin(sphi))
                z = (r * np.cos(stheta))
                x = torch.tensor(x)
                y = torch.tensor(y)
                z = torch.tensor(z)
                spts = torch.zeros(snumber, 3)
                spts[:, 0] = x
                spts[:, 1] = y
                spts[:, 2] = z

                embedview_fn, input_ch = get_embedder(3)
                t_un_val = embedview_fn(spts)
                seq_model = self.seq_model
                p_unc = 1 - seq_model(t_un_val)
                p_unc = p_unc.squeeze()

                s_sdf_nn_output = self.sdf_network(spts)
                a_unc = s_sdf_nn_output[:, -1] - 0.1

                pa_loss = 0.5*torch.mean(torch.log(p_unc/a_unc)+a_unc/p_unc-1)


                loss = loss + pa_loss
                self.writer.add_scalar('Loss/pa_loss', pa_loss, self.iter_step)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            self.writer.add_scalar('Loss/s_loss', s_loss*0.1, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
            self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                self.validate_image()

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh()

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()

            if self.iter_step == 100001:
                self.get_view_prior()
                self.priorNN()


    # 1113 光线步进
    def sphere_tracing(self, ray_o, ray_d):
        """Run sphere tracing algorithm for max iterations"""
        min_dis = 1e-3
        iters = 0
        sdf_threshold = 1e-3
        sphere_tracing_iters = 32
        work_mask = ray_o[:,0] > -100
        unfinished_mask_start = work_mask.clone()
        acc_start_dis = min_dis * torch.ones([(self.dataset.n_images + 0)])
        curr_start_points = ray_o + ray_d * acc_start_dis.unsqueeze(-1)
        curr_sdf_start = self.sdf_network.sdf(curr_start_points).squeeze(-1)
        curr_sdf_start =torch.t(curr_sdf_start)

        while True:
            unfinished_mask_start = (
                unfinished_mask_start & (curr_sdf_start.abs() > sdf_threshold))# & (acc_start_dis < max_dis)

            if iters == sphere_tracing_iters or unfinished_mask_start.sum() == 0:
                break
            iters += 1

            # Make step
            tmp = curr_sdf_start[unfinished_mask_start]
            acc_start_dis[unfinished_mask_start] += tmp
            curr_start_points[unfinished_mask_start] += ray_d[unfinished_mask_start] * tmp.unsqueeze(-1)
            curr_sdf_start[unfinished_mask_start] = self.sdf_network.sdf(curr_start_points[unfinished_mask_start]).squeeze(-1)

        convergent_mask = (
            work_mask
            & ~unfinished_mask_start
            & (curr_sdf_start.abs() <= sdf_threshold)
        )

        count = torch.count_nonzero(convergent_mask)
        return convergent_mask

    def get_view_prior(self):
        view = self.dataset.pose_all[:, None, :3, 3].numpy()

        psnumber = 1000
        psphi = np.random.uniform(0.0*np.pi, 2.0 * np.pi, psnumber)
        pstheta = np.arccos(np.random.uniform(-1, 1, psnumber))
        psu = np.random.uniform(0, 1, psnumber)
        pr = psu ** (1 / 3)
        px = (pr * np.sin(pstheta) * np.cos(psphi))
        py = (pr * np.sin(pstheta) * np.sin(psphi))
        pz = (pr * np.cos(pstheta))
        pspts = np.zeros([psnumber,3])
        pspts[:, 0] = px
        pspts[:, 1] = py
        pspts[:, 2] = pz

        pointa = view[0:self.dataset.n_images, 0, :]
        pointa = torch.tensor(pointa)
        ray_o = pointa

        pcount = torch.zeros([psnumber])
        for num in tqdm(range(0, psnumber)):
            pointb = pspts[num, :]
            pointb = torch.tensor(pointb)
            pointb = pointb.float()
            ray_d = pointb - pointa
            pr = self.sphere_tracing(ray_o, ray_d)

            pointaa = pointa[~pr]

            linea = pointaa-pointb

            anglemax = torch.tensor(0)
            for line1 in linea:
                for line2 in linea:
                    angle12 = torch.arccos(torch.clamp(torch.dot(line1,line2)/torch.linalg.norm(line1)/torch.linalg.norm(line2), -1, 1))
                    anglemax = torch.max(anglemax,angle12)

            pointuncp = 1/(self.dataset.n_images - torch.count_nonzero(pr)+2) + 1/(5*anglemax + 2)-0.01
            pcount[num] = 1-pointuncp

        self.pspts = pspts
        self.pcount = pcount


    def priorNN(self):
        import numpy as np
        import torch
        import torch.optim as optim
        import torch.nn as nn
        from collections import OrderedDict
        from matplotlib import pyplot as plt


        def training_loop(n_epochs, optimizer, model, loss_fn, t_u_train, t_u_val,
                          t_c_train, t_c_val):
            embedview_fn, input_ch = get_embedder(3)
            t_u_train = embedview_fn(t_u_train)
            t_u_val = embedview_fn(t_u_val)

            for epoch in range(1, n_epochs + 1):
                t_p_train = model(t_u_train)
                t_p_train = t_p_train.squeeze(1)
                loss_train = loss_fn(t_p_train, t_c_train)

                t_p_val = model(t_u_val)
                t_p_val = t_p_val.squeeze(1)
                loss_val = loss_fn(t_p_val, t_c_val)

                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()

                if epoch == 1 or epoch % 100 == 0:
                    print(f"Epoch {epoch}, Training loss {loss_train.item():.4f},"
                          f" Validation loss {loss_val.item():.4f}")


        t_u = self.pspts
        t_c = self.pcount

        t_c = torch.tensor(t_c).unsqueeze(1)
        t_u = torch.tensor(t_u).unsqueeze(1)
        t_c = t_c.float()
        t_u = t_u.float()

        n_samples = t_u.shape[0]
        n_val = int(0.10 * n_samples)

        shuffled_indices = torch.randperm(n_samples)
        train_indices = shuffled_indices[:-n_val]
        val_indices = shuffled_indices[-n_val:]

        t_u_train = t_u[train_indices]
        t_c_train = t_c[train_indices]

        t_u_val = t_u[val_indices]
        t_c_val = t_c[val_indices]

        t_un_train = 1 * t_u_train
        t_un_val = 1 * t_u_val

        self.seq_model = nn.Sequential(
            nn.Linear(in_features=21, out_features=50, bias=True),
            nn.Softplus(beta=100),
            nn.Linear(in_features=50, out_features=50, bias=True),
            nn.Softplus(beta=100),
            nn.Linear(in_features=50, out_features=50, bias=True),
            nn.Softplus(beta=100),
            nn.Linear(50, 1, bias=True),
            nn.Sigmoid()
        )


        optimizer = optim.Adam(self.seq_model.parameters(), lr=1e-4)

        training_loop(
            n_epochs=1000,
            optimizer=optimizer,
            model=self.seq_model,
            loss_fn=nn.MSELoss(),
            t_u_train=t_un_train,
            t_u_val=t_un_val,
            t_c_train=t_c_train,
            t_c_val=t_c_val)

    def innerprior(self):
        psnumber = 2000
        psphi = np.random.uniform(0.0 * np.pi, 2.0 * np.pi, psnumber)
        pstheta = np.arccos(np.random.uniform(-1, 1, psnumber))
        psu = np.random.uniform(0, 1, psnumber)
        pr = psu ** (1 / 3)
        px = (pr * np.sin(pstheta) * np.cos(psphi))
        py = (pr * np.sin(pstheta) * np.sin(psphi))
        pz = (pr * np.cos(pstheta))

        pspts = torch.zeros(psnumber, 3)
        px = torch.tensor(px)
        py = torch.tensor(py)
        pz = torch.tensor(pz)

        pspts[:, 0] = px
        pspts[:, 1] = py
        pspts[:, 2] = pz


        pspts = torch.tensor(pspts)

        s_sdf_nn_output = self.sdf_network(pspts)
        a_unc = s_sdf_nn_output[:, -1] - 0.1

        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = Axes3D(fig)
        pcount = a_unc.cpu().detach().numpy()

        pspts = torch.tensor(pspts).cpu().detach().numpy()
        punc = pspts

        surf = ax.scatter(punc[:, 0], punc[:, 1], punc[:, 2], c=1-pcount, cmap='bwr')
        plt.axis('off')
        plt.show()
        sys.exit()

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []
        uncertainty_fine = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              self.iter_step,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            def feasible(key):
                return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
                uncertaintys = render_out['uncert_fine']
                uncertaintys = uncertaintys.detach().cpu().numpy()
                uncertainty_fine.append(uncertaintys)

            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        uncertainty_img = None
        if len(uncertainty_fine) > 0:
            uncertainty_img = np.concatenate(uncertainty_fine, axis=0)
            uncertainty_img = uncertainty_img*50
            uncertainty_img = (uncertainty_img[:, :, None]
                               .reshape([H, W, 1, -1]) * 255).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'unc_inv'), exist_ok=True)  # UCS

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           normal_img[..., i])
            if len(uncertainty_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'unc_inv',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           uncertainty_img[..., i])

    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              self.iter_step,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine

    def validate_mesh(self, world_space=False, resolution=512, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)


        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')

    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        for i in range(n_frames):
            print(i)
            images.append(self.render_novel_image(img_idx_0,
                                                  img_idx_1,
                                                  np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                          resolution_level=4))
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(video_dir,
                                             '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),
                                fourcc, 30, (w, h))

        for image in images:
            writer.write(image)

        writer.release()


if __name__ == '__main__':
    print('Hello Wooden')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/womask.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=True, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='SZ')

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == 'train':
        runner.train()
        runner.validate_mesh(world_space=True, resolution=512, threshold=args.mcube_threshold)
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(world_space=True, resolution=512, threshold=args.mcube_threshold)
    elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
        _, img_idx_0, img_idx_1 = args.mode.split('_')
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1)
