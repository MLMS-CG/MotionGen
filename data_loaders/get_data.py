from torch.utils.data import DataLoader
from data_loaders.spectral.shape_dataset import ShapeSpec
from data_loaders.tensors import t2m_collate

def get_dataset_class():
    from data_loaders.spectral.dataset import Spactral
    return Spactral


def get_dataset(mode, path, nb_freqs, offset, size_window, means_stds, return_gender=False, rot_aug=False, used_id=-1):
    DATA = get_dataset_class()
    dataset = DATA(mode, path, nb_freqs, offset, size_window, means_stds, return_gender=return_gender, rot_aug=rot_aug, used_id=used_id)
    return dataset


def get_dataset_loader(mode, path, batch_size, nb_freqs, offset, size_window, used_id=-1, means_stds=None, return_gender=False, rot_aug=False):
    dataset = get_dataset(mode, path, nb_freqs, offset, size_window, means_stds, return_gender=return_gender, rot_aug=rot_aug, used_id=used_id)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, drop_last=True, collate_fn=t2m_collate
    )

    return loader, dataset.means_stds

def get_dataset_classifier(batch_size, status):
    dataset = ShapeSpec(status)
    loader = DataLoader(
        dataset, batch_size=batch_size, 
        num_workers=8, drop_last=True
    )
    return loader