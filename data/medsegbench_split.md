# MedSegBench Train/Val Split for In-Context Segmentation

Split designed to test **within-modality, cross-task generalization**: every validation
modality appears in training, but validation datasets segment different anatomical
structures than training datasets in the same modality.

Reference: [MedSegBench (Nature Scientific Data, 2024)](https://www.nature.com/articles/s41597-024-04159-2)

## Available Datasets

32 working datasets from MedSegBench (3 excluded due to loading errors: `abdomenus`,
`bkai-igh`, `idrib`). All images resized to 256x256. 17 grayscale, 15 RGB.
7 multi-class datasets, 25 binary.

## Training (24 datasets, ~39K train-split images)

| Dataset | Modality | Segmentation target | Classes | Train imgs |
|---|---|---|---|---|
| fhpsaop | Ultrasound | Fetal head, pubic symphysis | 3 | 2,800 |
| ultrasoundnerve | Ultrasound | Brachial plexus nerves | 2 | 1,651 |
| usforkidney | Ultrasound | Kidney | 2 | 3,210 |
| chasedb1 | Fundus | Retinal vessels | 2 | 19 |
| chuac | Fundus | Retinal vessels | 2 | 21 |
| dca1 | Fundus | Optic discs | 2 | 93 |
| covid19radio | Chest X-Ray | Lung | 2 | 14,814 |
| pandental | X-Ray | Mandible | 2 | 81 |
| isic2018 | Dermoscopy | Skin lesions | 2 | 2,594 |
| uwaterlooskincancer | Dermoscopy | Skin cancer | 2 | 143 |
| polypgen | Endoscopy | Colon polyps | 2 | 984 |
| m2caiseg | Endoscopy | Surgical tools/tissues | 19 | 245 |
| robotool | Endoscopy | Surgical tools | 2 | 350 |
| bbbc010 | Microscopy | C. elegans | 2 | 70 |
| brifiseg | Microscopy | Lung/cervix/breast/eye fibers | 3 | 1,005 |
| deepbacs | Microscopy | Bacterial cells | 2 | 17 |
| yeaz | Microscopy | Yeast cells | 2 | 360 |
| dynamicnuclear | Nuclear cell | Nuclear cells | 2 | 4,950 |
| nuset | Nuclear cell | Nuclear cells | 2 | 2,385 |
| mosmedplus | CT | Lung | 2 | 1,910 |
| promise12 | MRI | Prostate | 2 | 1,031 |
| cystoidfluid | OCT | Cystoid macular edema | 2 | 703 |
| monusac | Pathology | Lung/prostate/kidney/breast | 5 | 188 |
| tnbcnuclei | Pathology | Histopathology nuclei | 2 | 35 |

## Validation (8 datasets, ~4.5K val-split images)

| Dataset | Modality | Segmentation target | Classes | Val imgs | Why held out |
|---|---|---|---|---|---|
| busi | Ultrasound | Breast lesions | 2 | 64 | Different anatomy than train US (kidney/nerve/fetal) |
| covidquex | Chest X-Ray | Lung | 2 | 466 | Different dataset than covid19radio |
| isic2016 | Dermoscopy | Skin lesions | 2 | 90 | Classic benchmark, separate from isic2018 |
| kvasir | Endoscopy | GI polyps | 2 | 100 | Standard polyp segmentation benchmark |
| cellnuclei | Microscopy | Cell nuclei | 2 | 67 | Different target than train microscopy |
| wbc | Microscopy | White blood cells | 3 | 40 | Multi-class validation dataset |
| drive | Fundus | Retinal vessels | 2 | 2 | Classic retinal vessel benchmark |
| nuclei | Pathology | Cell nuclei | 2 | 14 | Different from multi-organ monusac |

## Design Rationale

1. **Cross-task within modality**: Every validation modality has training coverage, but
   validation datasets target different structures (e.g., breast US vs kidney/nerve/fetal
   US). This tests the core IC-segmentation capability: adapting to new tasks via context.

2. **Multi-class coverage**: 5 multi-class datasets in training (fhpsaop, brifiseg,
   m2caiseg, monusac, wbc-like), 1 in validation (wbc). Ensures the model handles
   multi-label scenarios.

3. **Classic benchmarks in validation**: DRIVE, ISIC2016, and Kvasir are established
   benchmarks that enable comparison with published results.

4. **Size balance**: Training covers ~39K images across 24 datasets. Validation is
   ~4.5K images across 8 datasets (~10% of images, 25% of datasets). Note that
   `covid19radio` (14K) dominates training by volume; use `max_samples_per_dataset`
   to cap if needed.

5. **Excluded datasets**: `abdomenus`, `bkai-igh`, `idrib` have loading errors in the
   current .npz files and are excluded from both splits.

## Config

See `configs/experiment/30_medsegbench.yaml` for the corresponding Hydra config with
`train_datasets` and `val_datasets` fields.
