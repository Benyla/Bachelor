import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) with a convolutional encoder and decoder.
    Args:
        in_channels (int): Number of channels in the input image.
        latent_dim (int): Dimensionality of the latent space.
        image_size (int): Height/width of the (square) input image.
    """
    def __init__(self, in_channels=3, latent_dim=128):
        super(VAE, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # -----------------------
        # ENCODER
        # -----------------------
        self.encoder = nn.Sequential(
            # Input: (in_channels, 64, 64)
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # -> (32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),           # -> (64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),          # -> (128, 8, 8)
            nn.ReLU(),
            nn.Flatten()  # -> (128*8*8)
        )

        # Fully-connected layers to produce the latent Gaussian parameters
        self.fc_mu = nn.Linear(128*8*8, latent_dim)
        self.fc_logvar = nn.Linear(128*8*8, latent_dim)

        # -----------------------
        # DECODER
        # -----------------------
        # Map latent vector back to flattened feature map
        self.decoder_input = nn.Linear(latent_dim, 128*8*8)
        self.decoder = nn.Sequential(
            # Reshape to (128, 8, 8)
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> (64, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # -> (32, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),  # -> (in_channels, 64, 64)
            nn.Sigmoid()  # Normalizes the output to [0, 1]
        )

    def reparameterize(self, mu, logvar):
        """
        Applies the reparameterization trick to sample from the latent Gaussian.
        Args:
            mu (Tensor): Mean of the latent Gaussian.
            logvar (Tensor): Log variance of the latent Gaussian.
            
        Returns:
            Tensor: Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar) # Standard deviation from log variance
        eps = torch.randn_like(std) # Sample epsilon from standard normal with matching dim as std
        return mu + eps * std # Return reparameterized sample

    def forward(self, x):
        """
        Defines the forward pass of the VAE.
        
        Args:
            x (Tensor): Input image tensor.
        
        Returns:
            Tuple[Tensor, Tensor, Tensor]: Reconstructed image, mean, and log-variance.
        """
        # Encode the input image to a latent representation
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        # Sample a latent vector using the reparameterization trick
        z = self.reparameterize(mu, logvar)
        # Decode the latent vector to reconstruct the image
        decoder_input = self.decoder_input(z)
        x_recon = self.decoder(decoder_input)
        return x_recon, mu, logvar

    def loss(self, x, mu, logvar, num_samples=1, sigma=1.0):
        """
        Computes the VAE loss using a Gaussian likelihood for reconstruction,
        with the option to sample more than one z.
        
        Args:
            x (Tensor): Original input image.
            mu (Tensor): Mean from the encoder's latent Gaussian.
            logvar (Tensor): Log variance from the encoder's latent Gaussian.
            num_samples (int): Number of samples to draw from q(z|x) (default is 1).
            sigma (float): Standard deviation for the Gaussian likelihood.
        
        Returns:
            Tensor: Total VAE loss.
        """
        # KL Divergence term (analytical solution for Gaussians)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Compute reconstruction loss over num_samples samples
        recon_loss_samples = []
        for _ in range(num_samples):
            # Sample z and generate reconstruction for each sample
            z = self.reparameterize(mu, logvar)
            decoder_input = self.decoder_input(z)
            recon_x = self.decoder(decoder_input)

            # Create a Normal distribution with mean=recon_x and std=sigma
            normal_dist = dist.Normal(recon_x, sigma)
            # Compute log likelihood: log p(x|z)
            log_px_given_z = normal_dist.log_prob(x)
            # Sum over all elements (pixels) in x
            recon_loss_samples.append(torch.sum(log_px_given_z))
        
        # Average the log likelihood over all samples and take the negative for loss
        recon_loss = -torch.mean(torch.stack(recon_loss_samples))
        
        return recon_loss, kl_loss

    def decode(self, z): # used in sample_generation.py when we have to decode a random z
        decoder_input = self.decoder_input(z)
        x_recon = self.decoder(decoder_input)
        return x_recon
    
    def encode(self, x, return_stats=False):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        if return_stats:
            return z, mu, logvar
        return z

