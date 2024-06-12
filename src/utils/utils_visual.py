import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

# Function to display reconstructed images
def display_reconstructed_images(epoch, vae_model, data, n_samples=10, dim=[1, 28, 28], is_flip=False):
    vae_model.eval()
    with torch.no_grad():
        data = data[:n_samples]
        recon_x, z, _, _ = vae_model(data)
        recon_x = recon_x[:n_samples]
        comparison = torch.cat([data.view(-1, dim[0], dim[1], dim[2]), recon_x.view(-1, dim[0], dim[1], dim[2])])
        comparison = make_grid(comparison, nrow=data.size(0))
        comparison = comparison.cpu().numpy().transpose(1, 2, 0)

        plt.figure(figsize=(15, 5))
        plt.imshow(comparison, cmap='gray')
        plt.axis('off')
        plt.title(f'Reconstructed Images at Epoch {epoch}')
        plt.show()


def display_reconstructed_and_flip_images(epoch, vae_model, flip_vae_model, data, n_samples=10, dim=[1, 28, 28],
                                          flip_dim=[3, 32, 32], is_mnist=True, is_both=True):
    vae_model.eval()
    with torch.no_grad():
        data = data[:n_samples]
        recon_x, z, _, _ = vae_model(data)
        recon_x_flip = flip_vae_model.decode(z)
        data = data[:n_samples]
        recon_x = recon_x[:n_samples]
        recon_x_flip = recon_x_flip[:n_samples]

        data = data.view(n_samples, dim[0], dim[1], dim[2])
        recon_x = recon_x.view(n_samples, dim[0], dim[1], dim[2])
        recon_x_flip = recon_x_flip.view(n_samples, flip_dim[0], flip_dim[1], flip_dim[2])
        z = z[:n_samples]
        fig, axes = plt.subplots(3, n_samples, figsize=(n_samples * 3 / 2, 4.5))
        if is_mnist:
            main_color = 'gray'
            flip_color = None
        elif is_both:
            main_color = 'gray'
            flip_color = 'gray'
        else:
            flip_color = 'gray'
            main_color = None

        for i in range(n_samples):
            axes[0, i].imshow(np.transpose(data[i].detach().cpu().numpy(), (1, 2, 0)), cmap=main_color)
            axes[0, i].axis('off')

            axes[1, i].imshow(np.transpose(recon_x[i].detach().cpu().numpy(), (1, 2, 0)), cmap=main_color)
            axes[1, i].axis('off')

            axes[2, i].imshow(np.transpose(recon_x_flip[i].detach().cpu().numpy(), (1, 2, 0)), cmap=flip_color)
            axes[2, i].axis('off')

    return plt