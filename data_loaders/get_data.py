from torch.utils.data import DataLoader

def get_dataset_class():
    from data_loaders.spectral.dataset import Spactral
    return Spactral


def get_dataset(mode, path, nb_freqs, offset, size_window, means_stds):
    DATA = get_dataset_class()
    dataset = DATA(mode, path, nb_freqs, offset, size_window, means_stds)
    return dataset


def get_dataset_loader(mode, path, batch_size, nb_freqs, offset, size_window, means_stds=None):
    dataset = get_dataset(mode, path, nb_freqs, offset, size_window, means_stds)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, drop_last=True
    )

    return loader, dataset.means_stds