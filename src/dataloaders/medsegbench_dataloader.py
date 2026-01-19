"""Simple dataloader for MedSegBench datasets."""

import random
import warnings
from typing import List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import medsegbench


# Available MedSegBench dataset classes
DATASET_CLASSES = {
    "medsegbench_abdomenus": medsegbench.AbdomenUSMSBench,
    "medsegbench_dca1": medsegbench.Dca1MSBench,
    "medsegbench_covid19radio": medsegbench.Covid19RadioMSBench
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
        root: Optional[str] = None,
        image_size: int = 224,
        download: bool = False,
        context_size: int = 0,
        label_ids: Optional[List[int]] = None,
    ):
        """
        Args:
            dataset_name: Name of the dataset (e.g., 'abdomenus', 'busi')
            split: 'train', 'val', or 'test'
            root: Root directory for dataset storage
            image_size: Target size for images (default 224 for ViT)
            download: Whether to download the dataset if not present
            context_size: Number of context examples to sample (0 = no context)
            label_ids: List of label IDs to use for binary segmentation.
                       If None, uses all non-zero labels found in masks.
        """
        if dataset_name.lower() not in DATASET_CLASSES:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {list(DATASET_CLASSES.keys())}"
            )

        self.dataset_name = dataset_name
        self.split = split
        self.image_size = image_size
        self.context_size = context_size
        self.label_ids = label_ids

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

        # Build label_to_cases mapping for efficient context sampling
        self.label_to_cases = self._build_label_to_cases_mapping()

    def _build_label_to_cases_mapping(self) -> dict:
        """Scan all masks and build mapping: label_id -> list of case indices."""
        label_to_cases = {}
        print(f"Building label-to-cases mapping for {len(self.dataset)} samples...")

        for idx in range(len(self.dataset)):
            _, mask = self.dataset[idx]
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask)

            # Get unique labels in this mask
            unique_labels = mask.unique().tolist()
            for label in unique_labels:
                if label not in label_to_cases:
                    label_to_cases[label] = []
                label_to_cases[label].append(idx)

        print(f"Found {len(label_to_cases)} unique labels: {sorted(label_to_cases.keys())}")
        return label_to_cases

    def __len__(self):
        return len(self.dataset)

    def _process_sample(self, idx, label_id=None):
        """Load and process a single sample.

        Args:
            idx: Sample index
            label_id: If provided, create binary mask for this label only.
                      If None, binarize the full mask (any label > 0).
        """
        image, mask = self.dataset[idx]

        # mask is already a tensor from medsegbench, resize it
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask)

        # Resize mask to match image size
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # Add channel dim for interpolation
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0).float(),
            size=(self.image_size, self.image_size),
            mode="nearest"
        ).squeeze(0).squeeze(0).long()

        # Create binary mask for specific label_id or binarize all
        if label_id is not None:
            binary_mask = (mask == label_id).long()
        else:
            binary_mask = (mask > 0).long()

        return image, binary_mask

    def __getitem__(self, idx):
        """Returns dict with image, mask, case_id, and optionally context examples."""
        # Sample a label_id for this sample (target and context will use the same)
        if self.label_ids is not None and len(self.label_ids) > 0:
            label_id = random.choice(self.label_ids)
        else:
            label_id = None  # Will binarize full mask

        image, mask = self._process_sample(idx, label_id=label_id)

        result = {
            "image": image,
            "label": mask,
            "case_id": f"s{idx:04d}",
            "label_id": label_id,
        }

        # Sample context examples if context_size > 0
        if self.context_size > 0:
            # Get cases that have this label (exclude current sample)
            if label_id is not None and label_id in self.label_to_cases:
                available_indices = [i for i in self.label_to_cases[label_id] if i != idx]
            else:
                # Fallback: all indices except current
                available_indices = [i for i in range(len(self)) if i != idx]

            # Sample context indices
            n_context = min(self.context_size, len(available_indices))
            if n_context < self.context_size:
                warnings.warn(
                    f"Only {n_context}/{self.context_size} cases have label {label_id}"
                )

            context_indices = random.sample(available_indices, n_context) if n_context > 0 else []

            context_images = []
            context_labels = []

            for ctx_idx in context_indices:
                ctx_image, ctx_mask = self._process_sample(ctx_idx, label_id=label_id)
                context_images.append(ctx_image)
                context_labels.append(ctx_mask)

            if len(context_images) > 0:
                # Stack context examples: [k, C, H, W]
                result["context_in"] = torch.stack(context_images, dim=0)
                result["context_out"] = torch.stack(
                    [m.unsqueeze(0).float() for m in context_labels], dim=0
                )
                result["context_case_ids"] = [f"s{i:04d}" for i in context_indices]

        return result

    @property
    def num_classes(self):
        """Returns number of segmentation classes."""
        if hasattr(self.dataset, 'num_classes'):
            return self.dataset.num_classes
        # Use precomputed label_to_cases mapping
        return len(self.label_to_cases)


def collate_fn(batch):
    """Collate function that handles optional context examples."""
    has_context = "context_in" in batch[0]

    result = {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "label": torch.stack([item["label"] for item in batch], dim=0),
        "case_id": [item["case_id"] for item in batch],
        "label_id": [item["label_id"] for item in batch],
    }

    if has_context:
        result["context_in"] = torch.stack([item["context_in"] for item in batch], dim=0)
        result["context_out"] = torch.stack([item["context_out"] for item in batch], dim=0)
        result["context_case_ids"] = [item["context_case_ids"] for item in batch]

    return result


def get_dataloader(
    dataset_name: str,
    split: str = "train",
    root: Optional[str] = None,
    batch_size: int = 8,
    image_size: int = 224,
    shuffle: Optional[bool] = None,
    num_workers: int = 4,
    download: bool = False,
    context_size: int = 0,
    label_ids: Optional[List[int]] = None,
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
        context_size: Number of context examples (0 = no context)
        label_ids: List of label IDs for binary segmentation

    Returns:
        DataLoader instance
    """
    dataset = MedSegBenchDataset(
        dataset_name=dataset_name,
        split=split,
        root=root,
        image_size=image_size,
        download=download,
        context_size=context_size,
        label_ids=label_ids,
    )

    if shuffle is None:
        shuffle = (split == "train")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )


def list_datasets():
    """List available MedSegBench datasets."""
    return list(DATASET_CLASSES.keys())
