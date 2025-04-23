import neptune 
import neptune.types
import os
import random
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()

class NeptuneLogger:
    def __init__(self, config):
        self.run = neptune.init_run(
            project=config["experiment"]["neptune_project"],
            api_token = os.getenv("NEPTUNE_API_TOKEN"),
            name=config["experiment"]["name"],
            tags=config["experiment"]["tags"],
            capture_hardware_metrics=False,   # <- disable hardware metrics
            capture_stderr=False,
            capture_stdout=False
        )
        self.run["parameters"] = config # Log the configuration parameters as metadata


    def log_metrics(self, metrics: dict, step: int = None, prefix: str = None):
        for key, value in metrics.items():
            metric_key = f"{prefix}/{key}" if prefix else key
            self.run[metric_key].log(value, step=step)

    def log_time(self, timings: dict, step: int = None, prefix: str = "timing"):
        """
        Log timing-related metrics. For example: {"epoch_time": 12.34}
        """
        for key, value in timings.items():
            tag = f"{prefix}/{key}" if prefix else key
            self.run[tag].log(value, step=step)


    def log_images(self, x, recon_x, step):
        idx = random.randint(0, x.size(0) - 1)
        original = x[idx].cpu()
        reconstructed = recon_x[idx].detach().cpu()

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(to_pil_image(original))
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(to_pil_image(reconstructed))
        axes[1].set_title("Reconstruction")
        axes[1].axis("off")

        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)

        self.run[f"visuals/comparison_epoke_{step}"] = neptune.types.File.from_content(buf.getvalue(), extension="png")
    
    def stop(self):
        self.run.stop()
    
    def log_latent_distribution(z_batch, epoch, save_path):
        """
        z_batch: Tensor [batch, z_dim]
        """
        # use matplotlib to histogram each latent dim (or e.g. corner plot/KDE)
        # save fig to disk or return fig

    def log_top_k_images_w_biggest_loss(inputs, reconstructions, losses, epoch, save_dir, k=3):
        """
        inputs: [batch, C, H, W]
        losses: [batch] per-sample combined loss
        """
        # sort losses descending, pick top-k indices
        # save or log those input / recon pairs
    