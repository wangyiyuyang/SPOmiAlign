# SPOmiAlign: a modality-agnostic framework for multimodal spatial omics alignment

## Pipeline

![SPOmiAlign pipeline](docs/_static/pipeline.png)

SPOmiAlign aligns spatial omics datasets across modalities, samples, and reference images by converting molecular measurements into spatial structural images (SSIs) and registering them with image-matching based transformations. The framework supports omic-to-image, omic-to-omic, and image-to-image workflows, including global alignment and non-rigid refinement, and can propagate the resulting transforms back to spatial coordinates for downstream analysis and reassignment.

## Directory structure

```text
SPOmiAlign-main/
|- SPOmiAlign/                 Core alignment and reassignment modules
|  `- software/                Bundled local dependencies
|     |- Roma/
|     `- fused-local-corr-master/
|- Tutorial/                   Tutorial notebooks and scripts
|  |- Tutorial 1 omic-to-image (spatial transcriptomics to CCF)/
|  |- Tutorial 2 omic-to-omic (spatial multi-omics alignment without paired images)/
|  `- Tutorial 3 image-to-image (spatial multi-omics alignment with paired images)/
|- docs/
|  `- _static/pipeline.png     Pipeline figure
|- env/
|  `- SPOmiAlign.yml           Conda environment file
|- Data/                       Downloaded tutorial data, not tracked by git
|- output/                     Generated tutorial results, not tracked by git
`- README.md
```

## Installation

1. Create and activate the conda environment:

```bash
conda env create -f env/SPOmiAlign.yml
conda activate SPOmiAlign
```

We run SPOmiAlign with A100 and torch2.6.0+cu124. If using GPU acceleration, install the PyTorch build that matches your GPU driver and CUDA version.

2. Install the bundled local dependencies:

```bash
cd SPOmiAlign/software/fused-local-corr-master/fused-local-corr-master
pip install -e .

cd ../../Roma
pip install -e .

cd ../../..
```

## Tutorial

Tutorial files are organized by application scenario. Each case keeps its notebook and Python script together.

### Tutorial 1: omic-to-image (spatial transcriptomics to CCF)

- spatial transcriptomic section (Slide-seq_29) to Allen Brain Atlas: [notebook](Tutorial/Tutorial%201%20omic-to-image%20%28spatial%20transcriptomics%20to%20CCF%29/spatial%20transcriptomic%20section%20%28Slide-seq_29%29%20to%20Allen%20Brain%20Atlas.ipynb) / [script](Tutorial/Tutorial%201%20omic-to-image%20%28spatial%20transcriptomics%20to%20CCF%29/spatial%20transcriptomic%20section%20%28Slide-seq_29%29%20to%20Allen%20Brain%20Atlas.py)

### Tutorial 2: omic-to-omic (spatial multi-omics alignment without paired images)

- Spatial multi-omics alignment for kidney sections: [notebook](Tutorial/Tutorial%202%20omic-to-omic%20%28spatial%20multi-omics%20alignment%20without%20paired%20images%29/spatial%20multi-omics%20alignment%20for%20kidney%20sections.ipynb) / [script](Tutorial/Tutorial%202%20omic-to-omic%20%28spatial%20multi-omics%20alignment%20without%20paired%20images%29/spatial%20multi-omics%20alignment%20for%20kidney%20sections.py)
- Spatial multi-omics alignment for mouse brain sections: [notebook](Tutorial/Tutorial%202%20omic-to-omic%20%28spatial%20multi-omics%20alignment%20without%20paired%20images%29/spatial%20multi-omics%20alignment%20for%20mouse%20brain%20sections.ipynb) / [script](Tutorial/Tutorial%202%20omic-to-omic%20%28spatial%20multi-omics%20alignment%20without%20paired%20images%29/spatial%20multi-omics%20alignment%20for%20mouse%20brain%20sections.py)

### Tutorial 3: image-to-image (spatial multi-omics alignment with paired images)

- Spatial multi-omics alignment with paired images: [notebook](Tutorial/Tutorial%203%20image-to-image%20%28spatial%20multi-omics%20alignment%20with%20paired%20images%29/spatial%20multi-omics%20alignment%20with%20paired%20images.ipynb) / [script](Tutorial/Tutorial%203%20image-to-image%20%28spatial%20multi-omics%20alignment%20with%20paired%20images%29/spatial%20multi-omics%20alignment%20with%20paired%20images.py)

## Data

The prepared tutorial data are available from Google Drive. Download the archive from the link below and use it as the input data in `Data/`. Users can put your data in `Data/` when using SPOmiAlign for your task.

[https://drive.google.com/file/d/17j39rTAISwuH-kL3H0hnvzTG15Zo_xSK/view?usp=sharing](https://drive.google.com/file/d/1Yd4iRdPewefABQdpaVC10noy0DNl3T_r/view?usp=sharing)
