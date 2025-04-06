import neptune 
import neptune.types
import os
import random
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
from io import BytesIO

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
        if prefix:
            # Log the entire dictionary under the specified prefix.
            # This creates a single metric "loss" with subkeys for different series.
            self.run[prefix].log(metrics, step=step)
        else:
            # If no prefix is provided, log each metric individually.
            for key, value in metrics.items():
                self.run[key].log(value, step=step)


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
    


