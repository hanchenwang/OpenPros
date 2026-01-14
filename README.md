# OpenPros: A Large-Scale Dataset for Limited View Prostate Ultrasound Computed Tomography

This repository is the official implementation of [OpenPros: A Large-Scale Dataset for Limited View Prostate Ultrasound Computed Tomography](https://arxiv.org/abs/2030.12345). 

![teaser_4](https://github.com/user-attachments/assets/bbc5153b-9c96-4b77-9eea-79f5221eecb1)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the InversionNet in the paper, run this command:

```train
python -u train.py -n baseline -s run2_inv -m FCN4_Deep -mc config/inv.yaml -g1v 1 -g2v 1 \
  --lr 1e-4 --lr-warmup-epochs 5 -b 128 -j 8  -nb 30 -eb 8 --k 1e5 \
  -t prostate_train_new.txt \
  --sync-bn --dist-url tcp://$MASTER:$MASTERPORT --world-size $SLURM_NTASKS 
```

To train the ViT-Inversion in the paper, run this command:
```train
python -u train.py -n baseline -s run2 -m ViT -mc config/inv.yaml -g1v 1 -g2v 1 \
  --lr 1e-4 --lr-warmup-epochs 5 -b 128 -j 8  -nb 30 -eb 8 --k 1e5 \
  -t prostate_train_new.txt \
  --sync-bn --dist-url tcp://$MASTER:$MASTERPORT --world-size $SLURM_NTASKS 
```

## Evaluation

To evaluate InversionNet on ImageNet, run:

```eval
python -u test.py -n baseline -s run2_inv -m FCN4_Deep -mc config/inv.yaml \
   -b 64 -j 8 --k 1e5 -r model_240.pth -v prostate_test_new.txt --vis -vsu 240 -vb 1 -vsa 20
```


To evaluate ViT-Inversion on ImageNet, run:

```eval
python -u test.py -n baseline -s run2_vit -m ViT -mc config/inv.yaml \
   -b 64 -j 8 --k 1e5 -r model_240.pth -v prostate_test_new.txt --vis -vsu 240 -vb 1 -vsa 20
```

## Pre-trained Models

You can download pretrained models here:

InversionNet checkpoint can be downloaded at: [https://drive.google.com/file/d/1TVrrzmVzUQvUyDTZ4il-EGH5EG60yFBL/view?usp=share_link](https://drive.google.com/file/d/1TVrrzmVzUQvUyDTZ4il-EGH5EG60yFBL/view?usp=share_link)
ViT-Inversion checkpoint can be downloaded at: [https://drive.google.com/file/d/1TVrrzmVzUQvUyDTZ4il-EGH5EG60yFBL/view?usp=share_link](https://drive.google.com/file/d/1u7nLsy-a7lsiG3gcK67HMeH_R4aUL3-Z/view?usp=share_link)

## Results

Our model achieves the following performance:

| Method         | MAE ↓  | RMSE ↓ | SSIM ↑ | PCC ↑  |
|----------------|--------|--------|--------|--------|
| InversionNet   | 0.0089 | 0.0433 | 0.9845 | 0.9798 |
| ViT-Inversion  | 0.0123 | 0.0566 | 0.9774 | 0.9728 |



## Contributing

First and foremost, the reproducibility of \textsc{OpenPros} benchmarks is guaranteed by a number of public resources, listed below. Remarkably, we have a group (link available below) where any related discussion is welcome. Our team also promises to maintain the platform and support further developments based on the community feedback.

Website: [https://open-pros.github.io](https://open-pros.github.io)
Google Group: [https://groups.google.com/g/openfwi](https://groups.google.com/g/openfwi)

The codes are released on Github under OSS license and BSD-3 license. We also attach the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License to the data.
