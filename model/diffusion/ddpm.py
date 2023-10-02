import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from functools import partial
from tqdm import tqdm
from model.diffusion.ddim import DDIMSampler
from model.diffusion.dpm_solver import DPMSolverSampler
import math
# from contextlib import contextmanager

def exists(x):
    return x is not None

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

#warmup_trick
def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


class DDPM(nn.Module):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 denoise_net:nn.Module,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 gpu_ids=[0],
                 # ignore_keys=[],
                 # load_only_unet=False,
                 #monitor="val/loss",
                 use_ema=False,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 conditional=True
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.num_timesteps = None
        self.use_positional_encodings = use_positional_encodings
        self.model = denoise_net
        self.conditional=conditional
        self.gpu_ids=gpu_ids
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.gpu_ids else "cpu")
        # count_params(self.model, verbose=True)
        self.use_ema = use_ema#False
        # if self.use_ema:#要看一下
        #     self.model_ema = LitEma(self.model)
        #     print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        # if monitor is not None:
        #     self.monitor = monitor
        # if ckpt_path is not None:
        #     self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type
        #可学习方差最好是false？
        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)


    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3,device=None):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        betas=betas.detach().cpu().numpy() if isinstance(betas,torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32,device=device)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        #eps是原始的形式，需要保留
        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    def set_loss(self,device:torch.device):
        '''set loss function
        '''
        if self.loss_type == "l1":
            self.loss_func=nn.L1Loss(reduction="sum").to(device)
        elif self.loss_type=="l2":
            self.loss_func=nn.MSELoss(reduction="sum").to(device)
        else:
            raise NotImplementedError("Specify loss_type attribute to be either \'l1\' or \'l2\'.")
    # @contextmanager
    # def ema_scope(self, context=None):
    #     if self.use_ema:
    #         self.model_ema.store(self.model.parameters())
    #         self.model_ema.copy_to(self.model)
    #         if context is not None:
    #             print(f"{context}: Switched to EMA weights")
    #     try:
    #         yield None
    #     finally:
    #         if self.use_ema:
    #             self.model_ema.restore(self.model.parameters())
    #             if context is not None:
    #                 print(f"{context}: Restored training weights")

    # def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
    #     sd = torch.load(path, map_location="cpu")
    #     if "state_dict" in list(sd.keys()):
    #         sd = sd["state_dict"]
    #     keys = list(sd.keys())
    #     for k in keys:
    #         for ik in ignore_keys:
    #             if k.startswith(ik):
    #                 print("Deleting key {} from state_dict.".format(k))
    #                 del sd[k]
    #     missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
    #         sd, strict=False)
    #     print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
    #     if len(missing) > 0:
    #         print(f"Missing Keys: {missing}")
    #     if len(unexpected) > 0:
    #         print(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool,condition_x: torch.Tensor = None):
        if condition_x is not None:
        	model_out=self.model(torch.cat([condition_x,x],dim=1),t)
        else:
        	model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-5., 5.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False,condition_x: torch.Tensor = None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised,condition_x=condition_x)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, x_in:torch.Tensor, return_intermediates=False):
        device = self.betas.device
        b = x_in.size(0)
        shape=(b,1, x_in.size(2),x_in.size(3))
        img = torch.randn(shape, device=device)
        intermediates = [img]
        if self.conditional:
        	condition_x=x_in
        else:
        	condition_x=None
        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised,condition_x=condition_x)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        else:
            return img

    @torch.no_grad()
    def sample(self, x_in: torch.Tensor, return_intermediates=False,ddim=True,use_dpm_solver=True,ddim_steps=200):#等价于super reoslution
        if ddim:
            if use_dpm_solver:
                sampler=DPMSolverSampler(self)
                shape = (1, x_in.size(2),x_in.size(3))
                batch_size=x_in.size(0)
                return sampler.sample(x_in,ddim_steps,batch_size,shape)
            else:
                ddim_sampler = DDIMSampler(self)
                shape = (1, x_in.size(2),x_in.size(3))
                batch_size=x_in.size(0)
                return ddim_sampler.sample(ddim_steps,batch_size,
                                                        shape,x_T=x_in,verbose=False,return_intermediates=return_intermediates)
        else:
            return self.p_sample_loop(x_in,
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise=torch.randn_like(x_start)
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_in, t, noise=None):
        x_start=x_in["HR"]#label
        if noise is None:
            noise=torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        if not self.conditional:
            model_out = self.model(x_noisy, t)
        else:
            model_out = self.model(torch.cat([x_in["INTERPOLATED"],x_noisy],dim=1), t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])#mean(dim=[1, 2, 3])
        # print(loss.shape)
        # log_prefix = 'train' if self.training else 'val'

        # loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss * self.l_simple_weight

        loss_vlb = self.lvlb_weights[t] * loss
        # loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb
        # loss_dict.update({f'{log_prefix}/loss': loss})

        return loss#, loss_dict

    def forward(self, x:dict, *args, **kwargs):
        
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (x["INTERPOLATED"].shape[0],), device=self.device).long()
        # print("t:",t.shape)
        return self.p_losses(x, t, *args, **kwargs)

