import os
import pandas as pd
import matplotlib.pyplot as plt

# Define file paths and model names (add .csv if needed)
folder_path = './data'
files = {
    'VAE+ (256)': 'val_recon_VAE+_256.csv',
    'VAE (256)': 'val_recon_VAE_256.csv',
    'VAE+ (128)': 'val_recon_VAE+_128.csv',
    'VAE (128)': 'val_recon_VAE_128.csv'
}

# Subtract constant
CONSTANT = 11292

# Load data and prepare for plotting
plot_data = {}
for model_name, filename in files.items():
    filepath = os.path.join(folder_path, filename)
    df = pd.read_csv(filepath, header=None)
    recon_loss = df.iloc[:, -1] - CONSTANT
    plot_data[model_name] = recon_loss

# Plotting
plt.figure(figsize=(10, 6))
for model_name, losses in plot_data.items():
    plt.plot(losses.index, losses.values, label=model_name)

plt.xlabel('Epoch')
plt.ylabel('Reconstruction Loss')
plt.title('Reconstruction Loss per Epoch')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot
output_folder = './experiments/plots'
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, 'recon_loss_plot.png')
plt.savefig(output_path, dpi=300)
plt.close()
