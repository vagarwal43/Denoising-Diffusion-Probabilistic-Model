import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from wandb_api import wandb_api_key
import wandb

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

# Eqn 14 from handout, s = epsilon
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
        self.image_size = image_size # 512x512
        self.model = model # unet
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
        self.alphas = cosine_schedule(self.num_timesteps)
        self.alphas_bar = torch.cumprod(self.alphas, dim=-1)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.inv_sqrt_alphas_bar = 1./self.sqrt_alphas_bar
        self.sigma = torch.sqrt(1. - self.alphas_bar) # std      


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
            x: The initial image. (x_{t})
            t: a tensor of the time index to sample at.
            t_index: a scalar of the index of the time step.
        Returns:
            The sampled image. (x_{t-1})
        """
        ####### TODO: Implement the p_sample function #######
        # sample x_{t-1} from the gaussian distribution wrt. posterior mean and posterior variance
        # Hint: use extract function to get the coefficients at time t

        # Hint: use self.noise_like function to generate noise. DO NOT USE torch.randn
        # Begin code here
        # Algorithm 2 steps 4-9
        eps_theta = self.noise_like(x.shape(), x.device())
        alpha_bar = extract(self.alphas_bar, t_index, x.shape())
        alpha = extract(self.alphas, t_index, x.shape())
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        mean = 1 / (alpha ** 0.5) * (x - eps_coef * eps_theta)
        std = extract(self.alpha, t)
        eps = torch.randn(xt.shape, device=xt.device)
        
        
        new_x =  mean + (std) * eps
        
        if t_index == 0:
            return x_0
        else:
            return new_x
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
        for i in range(self.num_timesteps - 1, -1, -1):
            t = torch.full((b,), i, device=img.device)
            img = self.p_sample(img, t, i)
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
        img = self.noise_like((batch_size, self.channels, self.image_size, self.image_size), batch_size.device)
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
        x_t = ...
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
        loss = ...

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
        t = torch.randint(self.num_timesteps) # generate in range of timesetps
        # Algorithm 1 steps 2-6
        return ...
