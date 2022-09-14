import torch.utils.data

from utils import *
import time
from networks import *
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torchvision
from functools import partial
from metric.cal_fid import InceptionV3, calculate_fid

print = partial(print, flush=True)

def run_fn(rank, args, world_size):
    device = torch.device('cuda', rank)
    torch.backends.cudnn.benchmark = True

    model = StyleGAN2(args, world_size)
    model.build_model(rank, device)
    model.train_model(rank, device)

class StyleGAN2():
    def __init__(self, args, NUM_GPUS):
        super(StyleGAN2, self).__init__()
        """ Model """
        self.model_name = 'StyleGAN2'
        self.phase = args['phase']
        self.checkpoint_dir = args['checkpoint_dir']
        self.result_dir = args['result_dir']
        self.log_dir = args['log_dir']
        self.sample_dir = args['sample_dir']
        self.dataset_name = args['dataset']
        self.NUM_GPUS = NUM_GPUS


        """ Training parameters """
        self.img_size = args['img_size']
        self.batch_size = args['batch_size']
        self.global_batch_size = self.batch_size * self.NUM_GPUS
        self.n_total_image = args['n_total_image'] * 1000
        self.iteration = self.n_total_image // self.global_batch_size

        self.g_reg_every = args['g_reg_every']
        self.d_reg_every = args['d_reg_every']
        self.lr = args['lr']


        """ Network parameters """
        self.channel_multiplier = args['channel_multiplier']
        self.lazy_regularization = args['lazy_regularization']
        self.r1_gamma = 10.0
        self.path_batch_shrink = 2
        self.path_weight = 2.0
        self.path_decay = 0.01
        self.mean_path_length = 0

        self.latent_dim = 512
        self.mixing_prob = args['mixing_prob']


        """ Print parameters """
        self.print_freq = args['print_freq']
        self.save_freq = args['save_freq']
        self.log_template = 'step [{}/{}]: elapsed: {:.2f}s, d_loss: {:.3f}, g_loss: {:.3f}, fid: {:.2f}, best_fid: {:.2f}, best_fid_iter: {}'
        self.n_sample = args['n_sample']

        """ Directory """
        self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
        check_folder(self.sample_dir)
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        check_folder(self.checkpoint_dir)
        self.log_dir = os.path.join(self.log_dir, self.model_dir)
        check_folder(self.log_dir)


        """ MISC """
        self.nsml_flag = args['nsml']

        if self.nsml_flag:
            import nsml
            self.nsml = nsml
            self.dataset_name = os.path.basename(self.nsml.DATASET_PATH)
            dataset_path = os.path.join(self.nsml.DATASET_PATH, 'train')
            self.dataset_path = dataset_path

        else:
            dataset_path = './dataset'
            self.dataset_path = os.path.join(dataset_path, self.dataset_name)

    ##################################################################################
    # Model
    ##################################################################################
    def build_model(self, rank, device):
        if self.phase == 'train':
            """ Init process """
            build_init_procss(rank, world_size=self.NUM_GPUS, device=device)

            """ Dataset Load """
            dataset = ImageDataset(dataset_path=self.dataset_path, img_size=self.img_size)
            self.dataset_num = dataset.__len__()
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=4,
                                                 sampler=distributed_sampler(dataset, rank=rank, num_replicas=self.NUM_GPUS, shuffle=True),
                                                 drop_last=True, pin_memory=True)
            self.dataset_iter = infinite_iterator(loader)

            """ Calculate FID metric """
            self.fid_dataset = ImageDataset(dataset_path=self.dataset_path, img_size=299, fid_transform=True)
            self.fid_loader = torch.utils.data.DataLoader(self.fid_dataset, batch_size=self.batch_size, num_workers=4,
                                                 sampler=distributed_sampler(dataset, rank=rank, num_replicas=self.NUM_GPUS, shuffle=False),
                                                 drop_last=False, pin_memory=True)
            self.inception = InceptionV3().to(device)
            self.inception = dataparallel_and_sync(self.inception, rank)

            """ Network """
            self.generator = Generator(size=self.img_size, channel_multiplier=self.channel_multiplier).to(device)
            self.discriminator = Discriminator(size=self.img_size, channel_multiplier=self.channel_multiplier).to(device)
            self.g_ema = deepcopy(self.generator).to(device)

            """ Optimizer """
            g_reg_ratio = self.g_reg_every / (self.g_reg_every + 1)
            d_reg_ratio = self.d_reg_every / (self.d_reg_every + 1)
            self.g_optim = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio))
            self.d_optim = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio))

            """ Distributed Learning """
            self.generator = dataparallel_and_sync(self.generator, rank)
            self.discriminator = dataparallel_and_sync(self.discriminator, rank)
            self.g_ema = dataparallel_and_sync(self.g_ema, rank)


            """ Checkpoint """
            latest_ckpt_name, start_iter = find_latest_ckpt(self.checkpoint_dir)
            if latest_ckpt_name is not None:
                print('Latest checkpoint restored!! ', latest_ckpt_name)
                print('start iteration : ', start_iter)
                self.start_iteration = start_iter

                latest_ckpt = os.path.join(self.checkpoint_dir, latest_ckpt_name)
                ckpt = torch.load(latest_ckpt, map_location=device)

                self.generator.load_state_dict(ckpt["generator"])
                self.discriminator.load_state_dict(ckpt["discriminator"])
                self.g_ema.load_state_dict(ckpt["g_ema"])

                self.g_optim.load_state_dict(ckpt["g_optim"])
                self.d_optim.load_state_dict(ckpt["d_optim"])

            else:
                if rank == 0:
                    print('Not restoring from saved checkpoint')
                self.start_iteration = 0

        else:
            """ Init process """
            build_init_procss(rank, world_size=self.NUM_GPUS, device=device)

            """ Network """
            self.g_ema = Generator(size=self.img_size, channel_multiplier=self.channel_multiplier).to(device)
            self.g_ema = dataparallel_and_sync(self.g_ema, rank)

            """ Checkpoint """
            latest_ckpt_name, start_iter = find_latest_ckpt(self.checkpoint_dir)
            if latest_ckpt_name is not None:
                print('Latest checkpoint restored!! ', latest_ckpt_name)
                print('start iteration : ', start_iter)
                self.start_iteration = start_iter

                latest_ckpt = os.path.join(self.checkpoint_dir, latest_ckpt_name)
                ckpt = torch.load(latest_ckpt, map_location=device)

                self.g_ema.load_state_dict(ckpt["g_ema"])

            else:
                print('Not restoring from saved checkpoint')
                self.start_iteration = 0

    def d_train_step(self, real_images, d_regularize=False, device=torch.device('cuda')):
        # gradient check
        requires_grad(self.discriminator, True)
        requires_grad(self.generator, False)

        # forward pass
        noise = mixing_noise(self.batch_size, self.latent_dim, self.mixing_prob, device)
        fake_images = self.generator(noise)

        real_logit = self.discriminator(real_images)
        fake_logit = self.discriminator(fake_images)

        # loss
        d_loss = d_logistic_loss(real_logit, fake_logit)

        if d_regularize:
            real_images.requires_grad = True
            real_logit = self.discriminator(real_images)
            r1_penalty = d_r1_loss(real_logit, real_images)
            r1_penalty = (self.r1_gamma / 2 * r1_penalty * self.d_reg_every + 0 * real_logit[0]).mean()

            d_loss += r1_penalty

        apply_gradients(d_loss, self.d_optim)

        return d_loss

    def g_train_step(self, g_regularize, device=torch.device('cuda')):
        # gradient check
        requires_grad(self.discriminator, False)
        requires_grad(self.generator, True)

        # forward pass
        noise = mixing_noise(self.batch_size, self.latent_dim, self.mixing_prob, device)
        fake_images = self.generator(noise)

        fake_logit = self.discriminator(fake_images)

        # loss
        g_loss = g_nonsaturating_loss(fake_logit)

        if g_regularize:
            path_batch_size = max(1, self.batch_size // self.path_batch_shrink)
            noise = mixing_noise(path_batch_size, self.latent_dim, self.mixing_prob, device)
            fake_img, latents = self.generator(noise, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(fake_img, latents, self.mean_path_length)
            self.mean_path_length = mean_path_length

            weighted_path_loss = self.path_weight * self.g_reg_every * path_loss

            g_loss += weighted_path_loss

        apply_gradients(g_loss, self.g_optim)

        return g_loss

    def train_model(self, rank, device):
        start_time = time.time()
        fid_start_time = time.time()

        # setup tensorboards
        train_summary_writer = SummaryWriter(self.log_dir)


        # start training
        if rank == 0:
            print()
            print(self.dataset_path)
            print("Dataset number : ", self.dataset_num)
            print("GPUs : ", self.NUM_GPUS)
            print("Each batch size : ", self.batch_size)
            print("Global batch size : ", self.global_batch_size)
            print("Target image size : ", self.img_size)
            print("Print frequency : ", self.print_freq)
            print("Save frequency : ", self.save_freq)
            print("PyTorch Version :", torch.__version__)
            print('max_steps: {}'.format(self.iteration))
            print()
        losses = {'g/loss': 0.0, 'd/loss': 0.0}
        fid_dict = {'metric/fid': 0.0, 'metric/best_fid': 0.0, 'metric/best_fid_iter': 0}


        fid = 0
        best_fid = 1000
        best_fid_iter = 0

        for idx in range(self.start_iteration, self.iteration):
            iter_start_time = time.time()

            real_img = next(self.dataset_iter)
            real_img = real_img.to(device)

            if idx == 0:
                if rank == 0:
                    print("count params")
                    g_params = count_parameters(self.generator)
                    d_params = count_parameters(self.discriminator)
                    print("G network parameters : ", format(g_params, ','))
                    print("D network parameters : ", format(d_params, ','))
                    print("Total network parameters : ", format(g_params + d_params, ','))
                    print()

            # update discriminator
            if (idx + 1) % self.d_reg_every == 0:
                d_loss = self.d_train_step(real_img, d_regularize=True, device=device)
            else:
                d_loss = self.d_train_step(real_img, d_regularize=False, device=device)

            losses['d/loss'] = d_loss

            # update generator
            if (idx + 1) % self.g_reg_every == 0:
                g_loss = self.g_train_step(g_regularize=True, device=device)
            else:
                g_loss = self.g_train_step(g_regularize=False, device=device)

            losses['g/loss'] = g_loss

            # moving average
            moving_average(self.g_ema, self.generator, decay=0.999)

            losses = reduce_loss_dict(losses)

            if np.mod(idx, self.save_freq) == 0 or idx == self.iteration - 1 :
                if rank == 0:
                    print("calculate fid ...")
                    fid_start_time = time.time()

                fid = calculate_fid(self.fid_loader, self.g_ema, self.inception, self.dataset_name, rank, device,
                                    self.latent_dim, fake_samples=50000, batch_size=self.batch_size)

                if rank == 0:
                    fid_end_time = time.time()
                    fid_elapsed = fid_end_time - fid_start_time
                    print("calculate fid finish: {:.2f}s".format(fid_elapsed))
                    if fid < best_fid:
                        print("BEST FID UPDATED")
                        best_fid = fid
                        best_fid_iter = idx
                        self.torch_save(idx)

                        fid_dict['metric/best_fid'] = best_fid
                        fid_dict['metric/best_fid_iter'] = best_fid_iter
                    fid_dict['metric/fid'] = fid


            if rank == 0:
                # save to tensorboard
                if self.nsml_flag:
                    if np.mod(idx, self.save_freq) == 0 or idx == self.iteration - 1:
                        self.nsml.report(**losses, scope=locals(), step=idx)
                        self.nsml.report(**fid_dict, scope=locals(), step=idx)
                    else:
                        self.nsml.report(**losses, scope=locals(), step=idx)

                    for k, v in losses.items():
                        train_summary_writer.add_scalar(k, v, global_step=idx)

                    if np.mod(idx, self.save_freq) == 0 or idx == self.iteration - 1:
                        train_summary_writer.add_scalar('fid', fid, global_step=idx)
                else:
                    for k, v in losses.items():
                        train_summary_writer.add_scalar(k, v, global_step=idx)

                    if np.mod(idx, self.save_freq) == 0 or idx == self.iteration - 1:
                        train_summary_writer.add_scalar('fid', fid, global_step=idx)

                # save every self.save_freq
                if np.mod(idx + 1, self.save_freq) == 0:
                    print("ckpt save")
                    self.torch_save(idx)

                if np.mod(idx + 1, self.print_freq) == 0:
                    with torch.no_grad():
                        partial_size = int(self.n_sample ** 0.5)
                        sample_z = [torch.randn([self.n_sample, self.latent_dim], device=device)]
                        self.g_ema.eval()

                        sample = self.g_ema(sample_z)

                        torchvision.utils.save_image(sample, './{}/fake_{:06d}.png'.format(self.sample_dir, idx + 1),
                                                     nrow=partial_size,
                                                     normalize=True, range=(-1, 1))
                        # normalize = set to the range (0, 1) by range(min, max)

                elapsed = time.time() - iter_start_time
                print(self.log_template.format(idx, self.iteration, elapsed, losses['d/loss'], losses['g/loss'],
                                               fid_dict['metric/fid'], fid_dict['metric/best_fid'], fid_dict['metric/best_fid_iter']))

        if rank == 0:
            # save model for final step
            self.torch_save(self.iteration)

            print("LAST FID: ", fid)
            print("BEST FID: {}, {}".format(best_fid, best_fid_iter))
            print("Total train time: %4.4f" % (time.time() - start_time))

    def torch_save(self, idx):
        torch.save(
            {
                'generator': self.generator.state_dict(),
                'discriminator': self.discriminator.state_dict(),
                'g_ema': self.g_ema.state_dict(),
                'g_optim': self.g_optim.state_dict(),
                'd_optim': self.d_optim.state_dict()
            },
            os.path.join(self.checkpoint_dir, '{}.pt'.format(idx))
        )

    @property
    def model_dir(self):
        return "{}_{}_{}".format(self.model_name, self.dataset_name, self.img_size)