# BacUNet

## 1. Introduction

This repository implements the training and inference pipeline for BacUNet, a multi-channel fluorescence image segmentation network designed for bacterial identification and segmentation in microbial communities.

The training strategy follows the progressive learning framework described in the BacUNet method. It transfers the model from single-bacterial segmentation to mixed-bacterial segmentation and, after adapting the data loader, can be extended to real microbial-community data.

The main training files are:

| File | Function |
|---|---|
| `train_mix11_ddp.py` | Main training entry point. Handles dataset initialization, model construction, distributed training, validation, logging, and checkpoint saving. |
| `train_eval_mix.py` | Training and validation functions. |
| `datasets.py` | Dataset definitions and data loading. |
| `predict_Bac11.py` | Inference and prediction-result export. |

The network architecture is imported from the `model` package. Utility functions, loss functions, and image transformations are imported from `utils` and `transforms`.

---

## 2. Environment and Dependencies

The code uses PyTorch and NVIDIA CUDA for GPU training. Distributed training is implemented using PyTorch Distributed Data Parallel (DDP) with the NCCL backend.

The main dependencies include:

- Python
- PyTorch
- torchvision
- NumPy
- tifffile
- OpenCV
- Pillow
- Matplotlib
- pandas
- tqdm
- TensorBoard
- h5py

The exact versions should be selected according to the CUDA version and the existing project environment.

A typical environment can be prepared using:

```bash
pip install numpy tifffile opencv-python pillow matplotlib pandas tqdm h5py tensorboard
```

Install a CUDA-compatible version of PyTorch and torchvision separately.

---

## 3. Dataset Organization

### 3.1 Single-bacterial dataset

The `Bac11Dataset` class is used for single-bac training and validation.

The expected directory structure is:

```text
data_dir/
└── dataset_name/
    ├── split_info_cate.json
    ├── im_crop/
    │   ├── sample_001.tif
    │   ├── sample_002.tif
    │   └── ...
    └── cp_sf_gt_crop/
        ├── sample_001.tif
        ├── sample_002.tif
        └── ...
```

The directory names can be changed through:

- `--data_dir`: Dataset root directory.
- `--dataset`: Dataset folder name.
- `--im_name`: Fluorescence image directory.
- `--gt_name`: Ground-truth directory.

The default directory names are:

```bash
--im_name im_crop
--gt_name cp_sf_gt_crop
```

The `split_info_cate.json` file provides sample lists grouped by bacterial category. It should contain category-specific entries for training and validation, for example:

```json
{
    "cate_train": {
        "Bacteria_A": ["sample_001", "sample_002"],
        "Bacteria_B": ["sample_003"]
    },
    "cate_val": {
        "Bacteria_A": ["sample_004"],
        "Bacteria_B": ["sample_005"]
    }
}
```

### 3.2 Multi-bacterial dataset

Multi-bac curriculum learning is implemented through category-based sample selection in Bac11Dataset. The selection is controlled by select, tr_num, and ignore_bac:
- select=True: Enables category-based sample selection. The dataset reads the category-specific sample lists (cate_train or cate_val) and filters samples according to ignore_bac and tr_num.
- select=False: Disables category-based filtering and uses the direct train or val sample list.
- ignore_bac: Specifies an ordered list of bacterial categories that can be excluded during curriculum learning.
- tr_num: Specifies how many categories to exclude from the end of ignore_bac. For example, tr_num=3 excludes the last three categories in the list, whereas tr_num=0 excludes none.
When select=True, changing tr_num changes the bacterial categories retained in the selected dataset. The order of categories in ignore_bac determines the exclusion order. When select=False, tr_num does not perform category-based filtering.

In train_mix11_ddp.py, the corresponding command-line argument is --tr_bac_num, which is passed to Bac11Dataset as tr_num in the applicable data-loader branch. 

### 3.3 Mixed-bacterial dataset

The `Mix11Dataset` class constructs mixed-bacterial samples from individual bacterial images and their corresponding annotations.

Its expected directory structure is:

```text
data_dir/
└── dataset_name/
    ├── im_crop/
    ├── cp_sf_gt_crop/
    ├── mix_2_train.json
    ├── mix_2_val.json
    ├── mix_3_train.json
    ├── mix_3_val.json
    └── ...
```

Each mixing-degree JSON file contains two important fields:

- `mix_bacs`: Bacterial categories included in each mixed sample.
- `mix_fnames`: The corresponding individual sample names used to construct the mixed sample.

For example:

```json
{
    "mix_bacs": [
        ["Bacteria_A", "Bacteria_B"]
    ],
    "mix_fnames": [
        ["sample_001", "sample_002"]
    ]
}
```

The loader generates each mixed sample online. The mixed fluorescence image is calculated by channel-wise summation:

\[
I_{\mathrm{mix}} = \sum_{i=1}^{N} I_i
\]

The mixed annotation is generated by taking the logical OR of the corresponding foreground masks. 

`--mix_num 2` is the mixing dfegree, requires `mix_2_train.json` and `mix_2_val.json`. The corresponding JSON files must exist before training or inference.

---

## 4. Data Preprocessing

The training and validation transformations are defined in the training script.

### 4.1 Training preprocessing

The training transformation applies:

1. Conversion of the input image and mask to tensors.
2. Image intensity preprocessing through `ToTensor`.
3. Spatial cropping using `Cropxy(x=68, y=68, slen=512)`.
4. Random horizontal flipping with probability 0.5.
5. Random rotation with a maximum angle of 30 degrees and probability 0.5.

The default training crop size is 512 × 512. The input intensity preprocessing is controlled by `--in_compress`; the exact compression behavior is defined in `transforms.py`.

### 4.2 Validation preprocessing

The validation transformation converts the image and mask to tensors and applies the validation crop configured in the training code. Random flipping and rotation are not applied during validation.

---

## 5. Model Configuration

The model is constructed in `train_mix11_ddp.py`.

The supported model names are:

```bash
--model_name u2net_full
--model_name u2net_lite
--model_name unet
```

The input channel number is set to 60 for these model configurations.

The default number of output classes is configured by:

```bash
--n_classes 97
```

It represents the background channel and N bacterial channels in the model output.

The MCML or MCSL mode is controlled by `--msk_type`. The supported modes include:

- `label`
- `one-hot-wo`

The default is:

```bash
--msk_type one-hot-wo
```

For `one-hot-wo`, the training code constructs an additional background channel from the foreground masks and uses sigmoid-based multi-label prediction.



The model output channels, mask representation, activation, loss, and evaluation procedure must be consistent. 

---

## 6. Distributed Training

The training script initializes distributed training using the NCCL backend and obtains the local GPU index from the `LOCAL_RANK` environment variable. The model is wrapped with `DistributedDataParallel`.

Therefore, launch the training script using `torchrun`, rather than directly using `python`.

For example, to train using eight GPUs:

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 \
../u2net/train_mix11_ddp.py \
--batch_size=2 \
--num_workers=4 \
--model_name=u2net_full \
--in_compress="clip" \
--data_loader='v5' \
--bg_weight=0.05 \
--epochs=2000 \
--tr_bac_num=9 \
--output_dir="/CX/neuro_tracking/ruifanghai/test_train/"   \
--dataset='global_view_20250727_new' \
--mix_num=11 \
--gt_name='cp_sf_gt_crop'  \
--im_name='im_crop'  \
--schedule='cnst'  \
--n_classes=12 \
--data_dir='/CX/neuro_tracking/ruifanghai/'  \
--msk_type='label'


```

The batch size is the batch size used by each process. Thus, the effective global batch size is approximately:

\[
B_{\mathrm{global}} =
B_{\mathrm{per\ GPU}} \times N_{\mathrm{GPU}}
\]

The training data uses `DistributedSampler` to distribute samples across processes. Validation is executed by rank 0 in the current implementation.

---


---

## 7. Checkpoints and Logging

The output directory is organized according to the model name, data-loading mode, compression configuration, loss type, and other training parameters.

The main output structure is:

```text
output_dir/
├── ckpt/
│   ├── checkpoint_best.pth
│   ├── model_<epoch>.pth
│   └── ...
└── log/
    └── <training-run>/
        ├── events.out.tfevents...
        └── output.log
```

The training script saves:

- `checkpoint_best.pth`: Checkpoint selected according to the validation criterion implemented in the script.
- `model_<epoch>.pth`: Periodic/final-epoch checkpoints according to the saving logic.

The checkpoint contains model parameters and, depending on the checkpoint-saving code, training-state information such as optimizer state, scheduler state, epoch, global step, and training arguments.

TensorBoard records training loss, learning rate, and the segmentation metrics implemented by the training and evaluation functions, including MAE, IoU, precision, and recall.

To visualize the training curves:

```bash
tensorboard --logdir /path/to/output_dir/log
```

Then open the TensorBoard address displayed in the terminal.

### Resume training

The `--resume` parameter loads model parameters from a checkpoint. In the supplied implementation, optimizer, scheduler, and epoch restoration are not fully enabled. Therefore, `--resume` should be treated as model-weight initialization rather than guaranteed restoration of the complete previous training state. The training epoch starts from `--start_epoch` unless the code is modified to restore it.

---

## 8. Inference

Inference is performed using `predict_Bac11.py`. The script constructs the selected network, loads the checkpoint, creates a test dataset, runs prediction without gradient calculation, and saves the predicted masks and probability maps.

### 8.1 Example inference commands


```bash
cd /CX/neuro_tracking/ruifanghai/code/u2net
python predict_Bac11.py  --split_json 'split_info_cated.json' --data_loader 'v5' --ckpt '/CX/neuro_tracking/ruifanghai/bacteriaML_12round11bac_train/mix_11/ddp_crop68_u2net_full_v5_clip/bce_mix_11_trBac_9_lr_1.0e-04_rsm_mix10_bw0.05_bs_4/ckpt/checkpoint_best.pth' --gt_name 'view_gt_crop_merge' --im_name 'view_im_merge' --output_dir '/CX/neuro_tracking/ruifanghai/bacteriaML_12round11bacview1_result'
```

### 8.2 Inference outputs

The script creates a timestamped result directory under:

```text
output_dir/result/
```

The main outputs include:

```text
result/
└── <model>_<dataset>_<group>_<timestamp>/
    ├── pred_msk/
    ├── vis/
    ├── prob/
    ├── prob_np/
    └── pred_<dataset>_<group>.csv
```

- `pred_msk/`: Predicted binary masks. For `one-hot-wo`, each sample has a subdirectory containing one PNG mask per output channel.
- `vis/`: Visualization comparing predicted masks with the supplied ground-truth masks.
- `prob/`: Probability heatmaps generated from the sigmoid probabilities in the `one-hot-wo` branch.
- `prob_np/`: Probability arrays saved as TIFF files in the `one-hot-wo` branch.
- `pred_<dataset>_<group>.csv`: Overall mean MAE, Jaccard/IoU, precision, and recall.
- When `--ch_metric` is enabled, the script also exports per-channel mean metrics.

**Implementation notes for the supplied `predict_Bac11.py`:**

1. The inference transform uses `PresetEval`; unlike the training transform, it does not apply random flipping or rotation. The shown `PresetEval` definition also does not apply the training crop.
2. The `v5` branch creates a `Bac11Dataset`, while `v6` creates a `Mix11Dataset`. 

---

## 9. Summary

The repository provides the main infrastructure for BacUNet training and inference, including single-bacterial and mixed-bacterial dataset loading, category-based sample selection, mixing-degree selection, distributed training, validation, checkpoint management, and prediction-result export.


The intended progressive training workflow is:

```text
Single-bacterial fluorescence images
              |
              v
      Single-bac pre-training
              |
              v
   Multi-bac curriculum learning
   (category selection via select,
        tr_num and ignore_bac)
              |
              v
      Mix-bac curriculum
       (mix-2, mix-3, ...)
              |
              v
  Real microbial-community fine-tuning
              |
              v
       Final BacUNet model
              |
              v
            inference
```
