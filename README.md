# Histopathology Vision Transformers with DINO

PyTorch implementation and histopathology pretrained models for DINO. For details, see **Emerging Properties in Self-Supervised Vision Transformers**.  
[[`blogpost`](https://ai.facebook.com/blog/dino-paws-computer-vision-with-self-supervised-transformers-and-10x-more-efficient-training)] [[`arXiv`](https://arxiv.org/abs/2104.14294)] [[`Yannic Kilcher's video`](https://www.youtube.com/watch?v=h3ij3F3cPIk)]

## Training

### Documentation
Please install [PyTorch](https://pytorch.org/). This codebase has been developed with python version 3.6, PyTorch version 1.7.1, CUDA 11.0 and torchvision 0.8.2. For a glimpse at the full documentation of DINO training please run:
```
python main_dino.py --help
```

```
python -m torch.distributed.launch --nproc_per_node=3 main_dino.py --arch vit_small --data_path /path/to/train_file_list_pickle/ --output_dir /path/to/saving_dir
```

## Self-attention visualization
You can look at the self-attention of the [CLS] token on the different heads of the last layer by running:
```
python visualize_attention.py
```


## Extract features:
To extract features from feature extractor on histopathology patches, run:

If you choose not to specify `--pretrained_weights`, then DINO reference weights are used by default. If you want instead to evaluate checkpoints from a run of your own, you can run for example:
```
python -m torch.distributed.launch --nproc_per_node=3 extract_features.py --pretrained_weights /path/to/checkpoint.pth --checkpoint_key teacher --data_path /path/to/train_test_file_list_pickle
```

## TCGA Lung Cancer weights for LUAD vs. LUSC:
5X weights :- https://drive.google.com/drive/folders/1cOshjIzrzuYN61lGAxyQ7f8-qGsG8GBs?usp=sharing

20X weights :- https://drive.google.com/drive/folders/1NSBKYt8oPtS7hK5dxNUTCKBjOBee0OIc?usp=sharing

## Citation
If you find this repository useful, please consider giving a star :star: and cite the following
```
Kapse, S., Das, S. and Prasanna, P., 2022. CD-Net: Histopathology Representation Learning using Pyramidal Context-Detail Network. arXiv preprint arXiv:2203.15078.
```
