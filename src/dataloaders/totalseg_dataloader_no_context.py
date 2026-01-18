from typing import Dict, List
import torch

from src.totalseg_dataloader import TotalSegmentatorDataset

class TotalSegDatasetNoContext(TotalSegmentatorDataset):
    """Modified TotalSegmentatorDataset that handles zero context size."""

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.context_size == 0:
            target_case_id, sampled_label_id = self.case_label_id_pairs[idx]
            target_case_folder = self.root_dir / target_case_id
            target_data = self._load_single_case(target_case_folder, sampled_label_id)
            return {
                "image": target_data["image"],
                "label": target_data["label"],
                "case_id": target_data["case_id"],
                "label_id": sampled_label_id,
            }
        else:
            data = super().__getitem__(idx)
            return {
                "image": data["target_in"],
                "label": data["target_out"],
                "case_id": data["target_case_id"],
                "label_id": data["label_id"],
            }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for the dataloader."""
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    return {
        "image": images,
        "label": labels,
        "case_ids": [item["case_id"] for item in batch],
        "label_ids": [item["label_id"] for item in batch],
    }
