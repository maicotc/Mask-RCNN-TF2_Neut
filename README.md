# Neutrophil Detection with Mask R-CNN (TensorFlow 2)

Training pipeline for a **Mask R-CNN** model that detects and classifies neutrophil
polarization states (**N1** and **N2**) in microscopy images.

This work was developed as part of an undergraduate research scholarship and my
final undergraduate thesis (*Trabalho de Conclusão de Curso*):

> **Artificial intelligence in prognosis determination for cancer patients:
> analysis of peripheral immune cell plasticity**

The goal is to support the study of **peripheral immune cell plasticity** —
specifically the N1/N2 phenotypic shift of neutrophils — as a potential factor in
prognosis determination for cancer patients, by automating cell detection and
classification from imaging data.

---

## Background

Neutrophils can adopt different functional phenotypes in the tumor
microenvironment, commonly described as **N1** (anti-tumoral) and **N2**
(pro-tumoral). Quantifying the balance between these populations from microscopy
images by hand is slow and subjective. This repository trains an instance
segmentation model to detect each cell and assign it an N1 or N2 label
automatically.

---

## Built on Mask-RCNN-TF2

This project relies on **[Mask-RCNN-TF2](https://github.com/ahmedfgad/Mask-RCNN-TF2)**
by [Ahmed Fawzy Gad](https://github.com/ahmedfgad), a port of the original
[Matterport Mask R-CNN](https://github.com/matterport/Mask_RCNN) implementation to
**TensorFlow 2**.

That project provides the `mrcnn` package (config, model, utils, visualization).
**This repository contains the custom training script** I used to fine-tune the
model on my own neutrophil dataset — it is not a fork of the framework itself, but
an application of it.

All credit for the TF2 Mask R-CNN implementation goes to the original author.

---

## Repository contents

| File | Description |
|------|-------------|
| `neutrot-maskrcnn_training.py` | Custom dataset loader + training entry point for the N1/N2 model. |

Key pieces in the script:

- **`CustomDataset`** — extends `mrcnn.utils.Dataset`; loads images and their
  annotations for the two classes (`N1`, `N2`).
- **`extract_boxes` / `load_mask`** — parse [LabelMe](https://github.com/wkentaro/labelme)-style
  JSON annotations (`shapes` with `points`, `imageWidth`, `imageHeight`) and build
  bounding-box-derived masks.
- **`NeutConfig`** — extends `mrcnn.config.Config` (3 classes = 2 + background,
  1 image/GPU, `LEARNING_RATE = 2e-5`, `STEPS_PER_EPOCH = 64`).
- **Training** — starts from COCO weights, excludes the classification/mask heads,
  and trains the `heads` layers.

---

## Requirements

- Python 3.x
- TensorFlow 2.x
- The `mrcnn` package from [Mask-RCNN-TF2](https://github.com/ahmedfgad/Mask-RCNN-TF2)
- `numpy`, `scikit-image`, `matplotlib`
- [LabelMe](https://github.com/wkentaro/labelme) — used to annotate the images (produces the JSON annotation files)

Install the `mrcnn` framework by following the instructions in the
[Mask-RCNN-TF2 repository](https://github.com/ahmedfgad/Mask-RCNN-TF2), then make it
importable from this project.

---

## Dataset layout

Images were annotated with **[LabelMe](https://github.com/wkentaro/labelme)**, which
exports one `.json` file per image (with `shapes`, `points`, `imageWidth` and
`imageHeight`). The script expects the following structure, with each annotation
file matching its image filename:

```
Dataset/
├── train/
│   ├── N1/
│   │   ├── images/      # e.g. cell_001.png
│   │   └── annots/      # e.g. cell_001.json
│   └── N2/
│       ├── images/
│       └── annots/
└── validation/
    ├── N1/
    │   ├── images/
    │   └── annots/
    └── N2/
        ├── images/
        └── annots/
```

---

## Pre-trained weights

The training script initializes from COCO weights:

```
mask_rcnn_coco.h5
```

Download it from the [Matterport Mask R-CNN releases](https://github.com/matterport/Mask_RCNN/releases)
and place it in the project root.

---

## Usage

1. Prepare the `Dataset/` folder as shown above.
2. Place `mask_rcnn_coco.h5` in the project root.
3. Run the training script:

```bash
python neutrot-maskrcnn_training.py
```

Trained weights are saved to the project root as
`Maskrcnn_neut-<timestamp>`.

> **Note:** the committed script trains for a single epoch on the `heads` layers as
> a starting point. Increase `epochs` and adjust `NeutConfig` for full training runs.

---

## Citation / Acknowledgements

- **Mask R-CNN (TF2):** Ahmed Fawzy Gad — https://github.com/ahmedfgad/Mask-RCNN-TF2
- **Original Mask R-CNN:** Matterport — https://github.com/matterport/Mask_RCNN
- **LabelMe (image annotation):** Kentaro Wada — https://github.com/wkentaro/labelme
- He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). *Mask R-CNN.* ICCV.

Developed as part of the undergraduate thesis *"Artificial intelligence in prognosis
determination for cancer patients: analysis of peripheral immune cell plasticity"*.
