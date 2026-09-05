# Failure-Aware Robustness Evaluation of Deep Learning Models for Tuberculosis Detection Under Synthetic Chest X-ray Degradation

Code and evaluation outputs for the manuscript submitted to the *Egyptian Journal of Radiology and Nuclear Medicine*.

This repository reproduces every table reported in the paper, from the fold assignments through to the supplementary confusion-matrix counts. It does not redistribute the chest radiographs; see [Data](#data).

**Release used for the reported analyses:** `v1.0.0`
**Commit:** `TODO\_COMMIT\_HASH`
**Archive:** `TODO\_ZENODO\_DOI`
**Licence:** MIT (code). The datasets carry their own terms.

\---

## Study in brief

Three ImageNet-pretrained CNNs — ResNet-50, DenseNet-121 and MobileNetV2 — are fine-tuned on the combined Montgomery and Shenzhen chest X-ray sets (800 images, 394 TB-positive) using non-repeated stratified five-fold cross-validation. Five synthetic degradations are then applied **to the held-out test images only**, at three severity levels each, and discrimination, sensitivity, predictive values and calibration are compared against the clean baseline.

|Degradation|Mild|Moderate|Severe|
|-|-|-|-|
|Gaussian noise|σ = 0.01|σ = 0.02|σ = 0.03|
|Motion blur|k = 5|k = 7|k = 9|
|JPEG compression|Q = 90|Q = 75|Q = 50|
|Downsampling|0.75×|0.67×|0.50×|
|Contrast reduction|γ = 0.9|γ = 0.8|γ = 0.7|

Model weights and the classification threshold are fixed before degradation is applied, so the reported changes reflect input degradation rather than re-fitting or threshold re-selection.

\---

## Environment

Reported analyses were produced with **PyTorch 2.1** and **torchvision 0.16** on a single **NVIDIA RTX A5000 (24 GB)**.

```bash
python -m venv .venv \&\& source .venv/bin/activate
pip install -r requirements.txt
```

Exact versions of all dependencies are pinned in `requirements.txt`.

A single global seed (42) is set for Python, NumPy and PyTorch at the start of every run. One training run was performed per architecture per fold, 15 runs in total, without repetition across seeds.

\---

## Data

The radiographs are **not** included in this repository. Both sets are distributed by the U.S. National Library of Medicine and must be obtained from the original source.

|Set|Images|TB-positive|Non-TB|
|-|-|-|-|
|Montgomery County|138|58|80|
|Shenzhen|662|336|326|
|**Combined**|**800**|**394**|**406**|

Expected layout:

```
data/
├── montgomery/CXR\_png/\*.png
└── shenzhen/CXR\_png/\*.png
```

All 800 images are used. No images are excluded and no resampling is applied. Fold assignments are fixed and recorded in `results/folds.csv`, so the split does not depend on your environment. Each held-out test fold contains 160 images (78–79 TB-positive).

\---

## Reproducing the paper

Full pipeline — fold assignment, training, clean and degraded evaluation, table generation:

```bash
bash run\_all.sh
```

Tables only, from the prediction files already in this repository (no training required):

```bash
python src/make\_tables.py --val results/predictions/val\_predictions.csv \\
                          --test results/predictions/test\_predictions.csv \\
                          --outdir results/tables
```

### Manuscript table → output file

|Manuscript|Output|
|-|-|
|Table 1 — clean baseline|`results/tables/table1\_clean.csv`|
|Table 3 — PPV and NPV|`results/tables/table3\_ppv\_npv.csv`|
|Table 4 — relative sensitivity drops|`results/tables/table4\_sensitivity.csv`|
|Table 5 — AUC and ECE|`results/tables/table5\_auc\_ece.csv`|
|Table 6 — error rates under severe blur|`results/tables/table6\_error\_rates.csv`|
|Table 7 — relative and absolute sensitivity|`results/tables/table7\_robustness.csv`|
|Supplementary Table S1|`results/tables/supplementary\_table\_s1.csv`|
|Supplementary Table S2|`results/tables/supplementary\_table\_s2.csv`|
|Per-fold thresholds|`results/tables/thresholds.csv`|

Table 2 compares against published studies and is not generated from these data.

\---

## How the numbers are computed

**Preprocessing.** Images are loaded as float32, resized to 224×224 by bilinear interpolation, and normalised to \[0,1] by per-image min–max scaling. Degradations are applied at that point, to test images only, after which images are replicated to three channels. Noise is added within the \[0,1] range and clipped; downsampled images are resized back to 224×224 so the network input size is constant across conditions; JPEG requires conversion to 8-bit and back, so that condition includes quantisation as well as compression. No augmentation, denoising, sharpening or contrast enhancement is applied at any stage.

**Training.** End-to-end fine-tuning with no frozen layers. Binary cross-entropy loss, Adam optimizer, initial learning rate 1×10⁻⁴ reduced by a factor of 0.1 after five consecutive epochs without improvement in validation loss, batch size 16, up to 50 epochs with early stopping on validation AUC (patience 10). No image augmentation and no class weighting. Hyperparameters were not selected by search: learning rate, batch size and Adam settings were fixed a priori and applied identically to all three architectures and all five folds.

**Threshold.** One threshold per fold, chosen by maximising Youden's J statistic on that fold's validation subset, then applied unchanged to the clean and degraded test images of the same fold. The threshold is never re-optimised on test data. Per-fold thresholds are written to `results/tables/thresholds.csv`.

**Aggregation.** Sensitivity, specificity, PPV and NPV are computed within each fold and averaged across folds with equal weight, not pooled. Because PPV and NPV are ratio statistics, fold-averaged values differ slightly from values computed from pooled counts; the fold-level counts are in `supplementary\_table\_s2.csv`.

**Calibration.** ECE with 10 equal-width probability bins, each weighted by its sample count. Point estimates are computed within each fold and averaged across folds; confidence intervals come from percentile bootstrap resampling of the test predictions pooled across folds, with 1000 resamples, stratified by TB status and reusing the same images across clean and degraded conditions to preserve pairing. Intervals are 95% percentile intervals, unadjusted for multiplicity across the 45 model × degradation × severity combinations.

\---

## Repository layout

```
├── README.md
├── LICENSE
├── requirements.txt
├── run\_all.sh                    # end-to-end reproduction
├── data/README.md                # how to obtain the images
├── src/
│   ├── data.py                   # loading, fold assignment, preprocessing
│   ├── degradations.py           # five degradations, three severities
│   ├── train.py                  # fine-tuning
│   ├── evaluate.py               # clean and degraded inference
│   └── make\_tables.py            # manuscript and supplementary tables
└── results/
    ├── folds.csv                 # image\_id, source, label, fold
    ├── predictions/              # per-image predictions (validation and test)
    └── tables/                   # generated tables
```

### Prediction file format

`val\_predictions.csv` — `model, fold, image\_id, y\_true, y\_prob`
`test\_predictions.csv` — `model, condition, severity, fold, image\_id, y\_true, y\_prob`

`condition` is one of Clean, Gaussian noise, Motion blur, JPEG compression, Downsampling, Contrast reduction. `severity` is `-` for Clean, otherwise Mild, Moderate or Severe. `y\_true` is 1 for TB-positive.

\---

## Scope and limitations of this code

Degradations are applied to images already resized to 224×224, so the parameters describe perturbations in the network input space rather than calibrated measurements at the detector. Downsampling and motion blur are both low-pass operations and are therefore not fully independent. Each noise condition uses a single fixed realisation. No quantitative image-quality measures are computed on the degraded images, and no naturally degraded radiographs, combined artefacts, or images from external devices or institutions are evaluated. The degradation parameters are nominal severity levels spanning a plausible range; they are not calibrated against measured image-quality metrics from clinical acquisitions.

\---

## Citation

```bibtex
@article{TODO\_KEY,
  title   = {Failure-Aware Robustness Evaluation of Deep Learning Models for
             Tuberculosis Detection Under Synthetic Chest X-ray Degradation},
  author  = {TODO\_AUTHORS},
  journal = {Egyptian Journal of Radiology and Nuclear Medicine},
  year    = {TODO\_YEAR},
  doi     = {TODO\_DOI}
}
```

Please also cite the source datasets:

Jaeger S, Candemir S, Antani S, Wáng YX, Lu PX, Thoma G (2014). Two public chest X-ray datasets for computer-aided screening of pulmonary diseases. *Quantitative Imaging in Medicine and Surgery* 4:475–477.

