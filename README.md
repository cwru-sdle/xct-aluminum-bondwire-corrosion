# xct-aluminum-bondwire-corrosion

## Automated Image Segmentation and Processing Pipeline Applied to X-ray Computed Tomography Studies of Pitting Corrosion in Aluminum Wires

This is the official repository for "Automated Image Segmentation and Processing Pipeline Applied to X-ray Computed Tomography Studies of Pitting Corrosion in Aluminum Wires". This respository contains the code and pipeline necessary to reproduce the work from this paper or to apply the approach to your own dataset.

## Repository Design

## Downloading Data

A version of the dataset can be downloaded by running the `/src/data_download.py` file. This script downloads the image and mask files from an OSF.io project hosted at: https://osf.io/k27v4/. All downloads will be placed in `/data/raw`. From here the `/src/preprocess_data.py` script can be run to preprocess the image and masks as described in the paper for the deep learning pipeline. The preprocessed versions will be placed in `/data/preprocessed`.

Alternative preprocessing techniques can be used if desired. However, the model scripts operate under the assumption the directory is structured as such:

```
xct-aluminum-bondwire-corrosion
├── data
│   ├── processed
│   │   ├── images
│   │   └── masks
│   └── raw
│       ├── images
│       └── masks
├── models
└── src
    ├── data
    └── modeling
```

The default modeling scripts assume the image mask pairs are inside of `/data/preprocessed`, however, these directories can easily be changed by adjusting the paths inside the the modeling scripts.

## Model Training (Optional)

A pretrained version of the model is contained in...

