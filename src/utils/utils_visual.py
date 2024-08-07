import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import seaborn as sns
import umap
import pandas as pd


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


def display_reconstructed_and_flip_images(axes, epoch, vae_model, flip_vae_model, data, n_samples=10, dim=[1, 32, 32],
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
        # fig, axes = plt.subplots(3, n_samples, figsize=(n_samples * 3 / 2, 4.5))
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


def plot_dist_matrix(similarity_score, title):

    similarity_np = similarity_score.detach().cpu().numpy()

    # Plot the heatmap
    plt.figure(figsize=(10, 8))  # Adjust figure size as needed
    sns.heatmap(similarity_np, annot=False, fmt=".2f", cmap="YlGnBu")
    plt.title(f"{title} (Heatmap)")
    plt.xlabel("Batch 2 Data Points")
    plt.ylabel("Batch 1 Data Points")
    plt.show()


def display_umap_for_latent_old(epoch, vae_1, vae_2, data_1, data_2, label_1, label_2):

    vae_1.eval()
    vae_2.eval()
    with torch.no_grad():
        _, z1, _, _ = vae_1(data_1)
        _, z2, _, _ = vae_2(data_2)

        # Combine the datasets
        data = np.vstack((z1.cpu(), z2.cpu()))
        labels = np.concatenate((label_1.cpu(), label_2.cpu()))
        domains = np.concatenate(
            (
                np.array(['domain1']*len(z1.cpu())),
                np.array(['domain2']*len(z2.cpu()))
            )
        )
        #
        # # print the shape of all three variables with descriptive messages
        # print(f"Shape of data: {data.shape}")
        # print(f"Shape of labels: {labels.shape}")
        # print(f"Shape of domains: {domains.shape}")

        # Fit and transform the data using UMAP
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(data)

        # Create a DataFrame for easier handling
        df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
        df['label'] = labels
        df['domain'] = domains

        # Define markers and colors
        markers = {'domain1': 'o', 'domain2': 'x'}
        unique_labels = np.unique(labels)
        palette = sns.color_palette("hsv", len(unique_labels))
        color_map = {label: palette[i] for i, label in enumerate(unique_labels)}

        # Plot the data
        fig, ax = plt.subplots(figsize=(12, 10))
        for domain in markers:
            for label in unique_labels:
                subset = df[(df['domain'] == domain) & (df['label'] == label)]
                ax.scatter(subset['UMAP1'], subset['UMAP2'], c=[color_map[label]], label=f'{domain}-{label}',
                           marker=markers[domain], alpha=0.6, edgecolors='w', linewidth=0.5)

        # Add a legend for domains only
        domain_handles = [
            plt.Line2D([0], [0], marker=markers[domain], color='w', markerfacecolor='k', markersize=10, label=domain)
            for domain in markers]
        domain_legend = ax.legend(handles=domain_handles, title='Domains', loc='best', bbox_to_anchor=(1.05, 1), ncol=1)

        # Add a legend for labels
        label_handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[label], markersize=10, label=label)
            for label in unique_labels]
        label_legend = ax.legend(handles=label_handles, title='Labels', loc='upper right', bbox_to_anchor=(1.05, 0.5),
                                 ncol=1)

        # Add the legends to the plot
        ax.add_artist(domain_legend)
        ax.add_artist(label_legend)

        ax.set_title(f'UMAP Visualization of Latent Space at Epoch {epoch}')

    return plt

def display_umap_for_latent(axes, epoch, vae_1, vae_2, data_1, data_2, label_1, label_2):
    vae_1.eval()
    vae_2.eval()
    with torch.no_grad():
        _, z1, _, _ = vae_1(data_1)
        _, z2, _, _ = vae_2(data_2)

        # Combine the datasets
        data = np.vstack((z1.cpu(), z2.cpu()))
        labels = np.concatenate((label_1.cpu(), label_2.cpu()))
        domains = np.concatenate((['domain1'] * len(z1.cpu()), ['domain2'] * len(z2.cpu())))

        # Fit and transform the data using UMAP
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(data)

        # Create a DataFrame for easier handling
        df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
        df['label'] = labels
        df['domain'] = domains

        # Define markers and colors
        colors = {'domain1': 'b', 'domain2': 'r'}
        unique_labels = np.unique(labels)
        palette = sns.color_palette("hsv", len(unique_labels))
        color_map = {label: palette[i] for i, label in enumerate(unique_labels)}

        # Plot the data
        for domain in colors:
            subset = df[(df['domain'] == domain)]
            axes[1].scatter(subset['UMAP1'], subset['UMAP2'], c=[colors[domain]], marker="s",
                            alpha=1,
                            edgecolors='w', linewidth=0.5, label=f'{domain}')

        # Create a combined legend
        axes[1].set_title(f'UMAP Visualization of Latent Space at Epoch {epoch}')

        # Define markers and colors
        markers = {'domain1': 'X', 'domain2': 's'}
        unique_labels = np.unique(labels)
        palette = sns.color_palette("hsv", len(unique_labels))
        color_map = {label: palette[i] for i, label in enumerate(unique_labels)}

        # Plot the data
        for domain in markers:
            for label in unique_labels:
                subset = df[(df['domain'] == domain) & (df['label'] == label)]
                axes[0].scatter(subset['UMAP1'], subset['UMAP2'], c=[color_map[label]], marker=markers[domain], alpha=0.6,
                           edgecolors='w', linewidth=0.5, label=f'{domain}-{label}')

        # Create a combined legend
        handles, _ = axes[0].get_legend_handles_labels()
        by_label = dict(zip(handles, _))
        axes[0].legend(by_label.values(), by_label.keys(), title='Domain-Label', loc='best', bbox_to_anchor=(1.05, 1))

        axes[0].set_title(f'UMAP Visualization of Latent Space at Epoch {epoch}')

    return axes
