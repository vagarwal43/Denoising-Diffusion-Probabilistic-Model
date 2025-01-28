
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import wandb

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
    )

def extract(a, t, x_shape):
    """
    Extracts the tensor at the given time step.
    Args:
        a: A tensor contains the values of all time steps.
        t: The time step to extract.
        x_shape: The reference shape.
    Returns:
        The extracted tensor.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_schedule(timesteps, s=0.008):
    """
    Defines the cosine schedule for the diffusion process
    Args:
        timesteps: The number of timesteps.
        s: The strength of the schedule.
    Returns:
        The computed alpha.
    """
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clip(alphas, 0.001, 1)


# normalization functions
def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# DDPM implementation
class Diffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        channels=3,
        timesteps=1000,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.model = model
        self.num_timesteps = int(timesteps)

        """
        Initializes the diffusion process.
            1. Setup the schedule for the diffusion process.
            2. Define the coefficients for the diffusion process.
        Args:
            model: The model to use for the diffusion process.
            image_size: The size of the images.
            channels: The number of channels in the images.
            timesteps: The number of timesteps for the diffusion process.
        """
        ## TODO: Implement the initialization of the diffusion process ##
        # 1. define the scheduler here
        # 2. pre-compute the coefficients for the diffusion process

        #scheduler
        self.alphas = cosine_schedule(self.num_timesteps)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim= 0).to(device) # sqrt of cummalative product
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.one_minus_sqrt_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod).to(device)
        # self.sqrt_alphas_cumprod_prev = torch.sqrt(self.alphas_cumprod_prev)
        
        # ###########################################################

    def noise_like(self, shape, device):
        """
        Generates noise with the same shape as the input.
        Args:
            shape: The shape of the noise.
            device: The device on which to create the noise.
        Returns:
            The generated noise.
        """
        noise = lambda: torch.randn(shape, device=device)
        return noise()

    # backward diffusion
    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        """
        Samples from the reverse diffusion process at time t_index.
        Args:
            x: The initial image.
            t: a tensor of the time index to sample at.
            t_index: a scalar of the index of the time step.
        Returns:
            The sampled image.
        """
        ####### TODO: Implement the p_sample function #######
        # sample x_{t-1} from the gaussian distribution wrt. posterior mean and posterior variance
        # Hint: use extract function to get the coefficients at time t
        # Hint: use self.noise_like function to generate noise. DO NOT USE torch.randn
        # Begin code here
        ...
        ...
        ...

        # coef1 = extract(self.x_0_pred_coef_1, t, x.shape)
        # coef2 = extract(self.x_0_pred_coef_2, t, x.shape)

        # output = self.model(x, t)
        # # x_0 = coef1 * x - coef2 * output
        # x_0 = (x - coef2*output)/(coef1)
        # x_0 = torch.clamp(x_0, -1., 1.)

        # posterior_var = extract(self.posterior_variance, t, x.shape)
        # posterior_mean = (
        #     extract(self.posterior_mean_coef1, t, x.shape) * x_0 +
        #     extract(self.posterior_mean_coef2, t, x.shape) * x
        # )
        # noise = self.noise_like(x.shape, x.device) if t_index > 0 else torch.zeros_like(x)

        
        # if t_index == 0:
        #     return x_0
        # else:
        #     return posterior_mean + torch.sqrt(posterior_var)*noise

        alphas_cumprod_t = extract(self.alphas_cumprod,t,x.shape)
        alphas_t = extract(self.alphas,t,x.shape)
        sqrt_alphas_t = extract(torch.sqrt(self.alphas),t,x.shape)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod,t,x.shape)
        one_minus_alphas_cumprod_t = extract(1. - self.alphas_cumprod,t,x.shape)

        if t_index > 0:
            alphas_cumprod_prev = extract(self.alphas_cumprod,t-1,x.shape)
            sqrt_alphas_cumprod_prev = extract(torch.sqrt(self.alphas_cumprod),t-1,x.shape)
        else:
            alphas_cumprod_prev = torch.ones_like(alphas_cumprod_t,device = x.device)
            sqrt_alphas_cumprod_prev = torch.ones_like(alphas_cumprod_t, device= x.device)

        model_output = self.model(x,t)
        x_0 = (x - torch.sqrt(one_minus_alphas_cumprod_t)*model_output) / sqrt_alphas_cumprod_t
        x0 = torch.clamp(x_0, -1. ,1.)

        # print(f"x_0 size: {x_0.size()}")
        # print(f"x0 size: {x0.size()}")

        coef1 = sqrt_alphas_t*(1. - alphas_cumprod_prev) / (1. - alphas_cumprod_t)
        coef2 = (sqrt_alphas_cumprod_prev)*(1. - alphas_t) / (1. - alphas_cumprod_t)

        # print(f"coef1 size: {coef1.size()}")
        # print(f"coef2 size: {coef2.size()}")

        mu = coef1 * x + coef2 * x0

        var = torch.sqrt((1. - alphas_cumprod_prev)*(1. - alphas_t) / (1. - alphas_cumprod_t))
        # print(f"mu size: {mu.size()}")
        # print(f"var size: {var.size()}")

        if t_index==0:
            return mu
        else:
            noise = self.noise_like(x.shape, device=x.device)
            # print(f"noise size: {noise.size()}")
            return mu + var * noise
        
        # ####################################################

    @torch.no_grad()
    def p_sample_loop(self, img):
        """
        Samples from the noise distribution at each time step.
        Args:
            img: The initial image that randomly sampled from the noise distribution.
        Returns:
            The sampled image.
        """
        b = img.shape[0]
        #### TODO: Implement the p_sample_loop function ####
        # 1. loop through the time steps from the last to the first
        # 2. inside the loop, sample x_{t-1} from the reverse diffusion process
        # 3. clamp and unnormalize the generted image to valid pixel range
        # Hint: to get time index, you can use torch.full()
        for i in reversed(range(self.num_timesteps)):
            img = self.p_sample(img, torch.full((b,),i, dtype = torch.long,device=device), i)
        img = torch.clamp(img, -1, 1)
        img = unnormalize_to_zero_to_one(img)
        return img
        # ####################################################

    @torch.no_grad()
    def sample(self, batch_size):
        """
        Samples from the noise distribution at each time step.
        Args:
            batch_size: The number of images to sample.
        Returns:
            The sampled images.
        """
        self.model.eval()
        #### TODO: Implement the p_sample_loop function ####
        # Hint: use self.noise_like function to generate noise. DO NOT USE torch.randn
        
        shape = (batch_size,self.channels,self.image_size,self.image_size)
        img = self.p_sample_loop(
            self.noise_like(
                shape, device=device
                )
            )
        return img

    # forward diffusion
    def q_sample(self, x_0, t, noise):
        """
        Samples from the noise distribution at time t. Simply apply alpha interpolation between x_0 and noise.
        Args:
            x_0: The initial image.
            t: The time index to sample at.
            noise: The noise tensor to sample from.
        Returns:
            The sampled image.
        """
        ###### TODO: Implement the q_sample function #######

        alphas_t = extract(self.sqrt_alphas_cumprod,t,x_0.shape)
        one_minus_alphas_t = extract(self.one_minus_sqrt_alphas_cumprod, t, x_0.shape)
        x_t = alphas_t * x_0 + noise * one_minus_alphas_t
        return x_t

    def p_losses(self, x_0, t, noise):
        """
        Computes the loss for the forward diffusion.
        Args:
            x_0: The initial image.
            t: The time index to compute the loss at.
            noise: The noise tensor to use.
        Returns:
            The computed loss.
        """
        ###### TODO: Implement the p_losses function #######
        # define loss function wrt. the model output and the target
        # Hint: you can use pytorch built-in loss functions: F.l1_loss

        x_t = self.q_sample(x_0,t,noise)
        output = self.model(x_t,t)
        loss = F.l1_loss(noise, output)

        return loss
        # ####################################################

    def forward(self, x_0, noise):
        """
        Computes the loss for the forward diffusion.
        Args:
            x_0: The initial image.
            noise: The noise tensor to use.
        Returns:
            The computed loss.
        """
        b, c, h, w, device, img_size, = *x_0.shape, x_0.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        ###### TODO: Implement the forward function #######
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        loss_c = self.p_losses(x_0,t,noise)
        return loss_c