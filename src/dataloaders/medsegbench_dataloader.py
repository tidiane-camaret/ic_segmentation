"""Simple dataloader for MedSegBench datasets."""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import medsegbench


# Available MedSegBench dataset classes
DATASET_CLASSES = {
    "medsegbench_abdomenus": medsegbench.AbdomenUSMSBench,
    "medsegbench_dca1": medsegbench.Dca1MSBench
}

"""
"busi": medsegbench.BUSIMSBench,
"camus": medsegbench.CAMUSMSBench,
"chasedb1": medsegbench.CHASEDB1MSBench,
"covid_qu_ex": medsegbench.CovidQUExMSBench,
"cvc_clinicdb": medsegbench.CVCClinicDBMSBench,
"dca1": medsegbench.DCA1MSBench,
"drive": medsegbench.DRIVEMSBench,
"drishti_gs": medsegbench.DrishtiGSMSBench,
"fetal_head": medsegbench.FetalHeadMSBench,
"glas": medsegbench.GLASMSBench,
"hc18": medsegbench.HC18MSBench,
"idrid": medsegbench.IDRiDMSBench,
"isic2018": medsegbench.ISIC2018MSBench,
"kvasir_seg": medsegbench.KvasirSEGMSBench,
"nci_isbi2013": medsegbench.NCIISBI2013MSBench,
"nucseg": medsegbench.NuCSegMSBench,
"pannuke": medsegbench.PanNukeMSBench,
"promise12": medsegbench.Promise12MSBench,
"refuge": medsegbench.REFUGEMSBench,
"siim_acr": medsegbench.SIIMACRMSBench,
"stare": medsegbench.STAREMSBench,
"tn3k": medsegbench.TN3KMSBench,
"tnbc": medsegbench.TNBCMSBench,
"""
class MedSegBenchDataset(Dataset):
    """Wrapper for MedSegBench datasets with configurable transforms."""

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        root: str = None,
        image_size: int = 224,
        download: bool = False,
    ):
        """
        Args:
            dataset_name: Name of the dataset (e.g., 'abdomenus', 'busi')
            split: 'train', 'val', or 'test'
            root: Root directory for dataset storage
            image_size: Target size for images (default 224 for ViT)
            download: Whether to download the dataset if not present
        """
        if dataset_name.lower() not in DATASET_CLASSES:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {list(DATASET_CLASSES.keys())}"
            )

        self.dataset_name = dataset_name
        self.split = split
        self.image_size = image_size

        # Image transform: resize and convert to tensor
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        # Load the underlying medsegbench dataset
        dataset_cls = DATASET_CLASSES[dataset_name.lower()]
        self.dataset = dataset_cls(
            split=split,
            root=root,
            download=download,
            transform=self.image_transform,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Returns dict with image, mask, and case_id."""
        image, mask = self.dataset[idx]

        # mask is already a tensor from medsegbench, resize it
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask)

        # set values > 1 to 1
        mask[mask>1]=1

        # Resize mask to match image size
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # Add channel dim for interpolation
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0).float(),
            size=(self.image_size, self.image_size),
            mode="nearest"
        ).squeeze(0).squeeze(0).long()

        return {
            "image": image,
            "label": mask,
            "case_id": f"s{idx:04d}",
        }

    @property
    def num_classes(self):
        """Returns number of segmentation classes."""
        if hasattr(self.dataset, 'num_classes'):
            return self.dataset.num_classes
        # Infer from first few samples if not available
        classes = set()
        for i in range(min(10, len(self))):
            _, mask = self[i]
            classes.update(mask.unique().tolist())
        return len(classes)


def get_dataloader(
    dataset_name: str,
    split: str = "train",
    root: str = None,
    batch_size: int = 8,
    image_size: int = 224,
    shuffle: bool = None,
    num_workers: int = 4,
    download: bool = False,
) -> DataLoader:
    """
    Get a DataLoader for a MedSegBench dataset.

    Args:
        dataset_name: Name of the dataset
        split: 'train', 'val', or 'test'
        root: Root directory for dataset storage
        batch_size: Batch size
        image_size: Target image size
        shuffle: Whether to shuffle (defaults to True for train)
        num_workers: Number of data loading workers
        download: Whether to download if not present

    Returns:
        DataLoader instance
    """
    dataset = MedSegBenchDataset(
        dataset_name=dataset_name,
        split=split,
        root=root,
        image_size=image_size,
        download=download,
    )

    if shuffle is None:
        shuffle = (split == "train")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def list_datasets():
    """List available MedSegBench datasets."""
    return list(DATASET_CLASSES.keys())
