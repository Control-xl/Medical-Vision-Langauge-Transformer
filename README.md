# Medical Vision-Language Transformer (MVLT)

## Requirement
torch >= 1.11.0

torchvision >= 0.12.0 (Vision Transformer support from PyTorch)

transformers >= 4.16.0 

## Prepare for pre-training
To pretrain MVLT wtih Swin Transformer as the visual backbone, prepare 3 datasets, RGC, ROCO and MedICaT. Download  train/test split of RGC from [openI](https://openi.nlm.nih.gov/imgs/collections/RGC.zip) and put them in ```./dataset/RGC/``` so that they are organized like ```./datset/RGC/RGC_dataset.json```.
### RGC

> Due to the copyright issue, we cannot directly provide the dataset. If you are interested in this dataset, please send an email to li-control.xu@connect.polyu.hk and we will give the code for materializing the dataset.

Run:
```
python preprocess_rgc.py
```
This will pre-process RGC dataset and save the data in .pkl format.

### ROCO 
Download ROCO following [ROCO](https://github.com/razorx89/roco-dataset) and put the files in ```./dataset/ROCO/```.

### MedICaT
Download MedICaT following [MedICat](https://github.com/allenai/medicat) and put the files in ```./dataset/medicat/```.

### pre-prepraing Swin Transformer
Download ```Swin-S``` [chekcpoint](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth) (swin_small_patch4_window7_224.pth) from [Swin Transformer](https://github.com/microsoft/Swin-Transformer) and put it in ```./checkpoints/```. 

Pre-train the model:
```
python run_pretrain_rgc_roco_medicat.py --conv swintransformer --batch 32 --max_length 80 --save_model_name swin-rgc-roco-medicat --epochs 150 --save_freq 50
```
The pre-trained model will be saved in ```./checkpoints/swin-rgc-roco-medicat/```

The pre-trained model (Swin-S + [RGC+ROCO+MedICaT]) can be found in [Google Drive](https://drive.google.com/file/d/1DKQ2IkULu_gfPBEPfD4vx72Vbx-49UU5/view?usp=share_link)

You can also use Resnet101 as visual backbone by using the argument ```--conv resnet101```

## Fine-tuning
### Medical Visual Question Answering (Med-VQA)
#### SLAKE
Download SLAKE from [Google Drive](https://drive.google.com/file/d/1TzZelZoS7IOUbEbNl_tPmv5uD7m5JnaQ/view?usp=sharing).

put the files in ```./dataset/SLAKE/``` and preprocess SLAKE:
```
python preprocess_VQA.py --dataset SLAKE
```
Fine-tuning on SLAKE
```
python run_vqa.py --batch 64 --conv swintransformer --pretrained --pretrained_path ./checkpoints/swin-rgc-roco-medicat --dataset SLAKE --epochs 100 --total_round 10 --lr 2e-5
```
It will run 10 times with different torch seeds. We can also reduce the repeat time by using ```--total_round 1```

#### VQA-RAD
Download VQA-RAD from [Google Drive](https://drive.google.com/file/d/1Dyp4ZlIYLyPK6hqJoTDKhAlbefjBj2BJ/view?usp=sharing)


put the files in ```./dataset/VQA-RAD/``` and preprocess SLAKE:
```
python preprocess_VQA.py --dataset VQA-RAD
```
Fine-tuning on VQA-RAD
```
python run_vqa.py --batch 64 --conv swintransformer --pretrained --pretrained_path ./checkpoints/swin-rgc-roco-medicat --dataset VQA-RAD --epochs 100 --total_round 10 --lr 2e-5
```
>Note that the learning rate is not the optimal for different platform. In Win11 with PyTorch 2.0, the learning rate can be set to ```3e-5```

### Report Generation on MIMIC-CXR and IU X-Ray
Download MIMIC-CXR and IU X-Ray from [G2GEN](https://github.com/cuhksz-nlp/R2Gen)

Put MIMIC-CXR in ```./dataset/mimic_cxr/```, Put IU X-Ray in ```./dataset/iu_xray/```

```
python run_report_generation.py --batch 32 --conv swintransformer --pretrained --pretrained_path ./checkpoints/swin-rgc-roco-medicat --dataset mimic --test_frq 5
```

### Report Generation on RGC
```
python run_seq2seq.py --batch 32 --conv swintransformer --pretrained --pretrained_path ./checkpoints/swin-rgc-roco-medicat --test_frq 5 --beam_search
```


### Image-Text Retrieval on RGC
```
python run_retrieval.py --batch 32 --conv swintransformer --pretrained --pretrained_path ./checkpoints/swin-rgc-roco-medicat  --do_train --do_test --do_rank --epochs 100 --lr 1e-6
```
You can also directly train the models on the downstream tasks from scratch by removing the argument ```--pretrained```


