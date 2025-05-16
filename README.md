# OpenPros: A Large-Scale Dataset for Limited View Prostate Ultrasound Computed Tomography

This repository is the official implementation of [OpenPros: A Large-Scale Dataset for Limited View Prostate Ultrasound Computed Tomography](https://arxiv.org/abs/2030.12345). 

![teaser_3](https://github.com/user-attachments/assets/d7f79054-2203-4059-a726-9c47c73ccc66)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
python -u train.py -n baseline -s debug2 -m FCN4_Deep -mc config/inv.yaml -g1v 1 -g2v 1 --lr 1e-4 -b 128 -j 12 \
  --sync-bn --dist-url tcp://$MASTER:$MASTERPORT --world-size $SLURM_NTASKS -nb 1 -eb 4 -pf 1
```

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

## Results

Our model achieves the following performance:

Method	|MAE ↓	|RMSE ↓	|SSIM ↑	|PCC ↑
--------|-------|-------|-------|------
InversionNet	|0.0089	|0.0433	|0.9845	|0.9798
ViT-Inversion	|0.0123	|0.0566	|0.9774	|0.9728



## Contributing

First and foremost, the reproducibility of \textsc{OpenPros} benchmarks is guaranteed by a number of public resources, listed below. Remarkably, we have a group (link available below) where any related discussion is welcome. Our team also promises to maintain the platform and support further developments based on the community feedback.

\begin{itemize}
\item \textbf{Website: } \url{https://open-pros.github.io}
\item \textbf{Google Group: } \url{https://groups.google.com/g/openfwi}
\end{itemize}

The codes are released on Github under \textbf{OSS} license and \textbf{BSD-3} license. We also attach the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License to the data.
