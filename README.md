# :houses: Airbert: In-domain Pretraining for Vision-and-Language Navigation :houses:

[![MIT](https://img.shields.io/github/license/airbert-vln/bnb-dataset)](./LICENSE.md)
[![arXiv](https://img.shields.io/badge/arXiv-<INDEX>-green.svg)](https://arxiv.org/abs/<INDEX>)
[![R2R 1st](https://img.shields.io/badge/R2R-🥇-green.svg)](https://eval.ai/web/challenges/challenge-page/97/leaderboard/270)
[![ICCV 2021](https://img.shields.io/badge/ICCV-2021-green.svg)](http://iccv2021.thecvf.com/home)
[![site](https://img.shields.io/badge/site-🌐-green.svg)](https://airbert-vln.github.io)

This repository stores the codebase for Airbert and some pre-trained model.
It is based on the codebase of [VLN-BERT](https://github.com/arjunmajum/vln-bert).


## :hammer_and_wrench: 1. Getting started

You need to have a recent version of Python (higher than 3.6) and install dependencies:

```bash
pip install -r requirements.txt
```



## :minidisc: 2. Preparing dataset

You need first to download the BnB dataset, prepare an LMDB file containing visual features and the BnB dataset files. Everything is described in our [BnB dataset repository](https://github.com/airbert-vln/bnb-dataset).

## :muscle: 3. Training Airbert

Download a checkpoint of VilBERT pre-trained on [Conceptual Captions](https://dl.fbaipublicfiles.com/vilbert-multi-task/pretrained_model.bin).

Fine-tune the checkpoint on the BnB dataset using one of the following path-instruction method.

To make the training faster, a [SLURM script](./slurm/train-bnb-64.sh) is provided with 64 GPUs. You can provide extra arguments depending on the path-instruction method. 

For example:

```bash
export name=pretraining-with-captionless-insertion
echo $name
sbatch --job-name $name \
 --export=name=$name,pretrained=vilbert.bin,args=" --masked_vision --masked_language --min_captioned 2 --separators",prefix=2capt+ \
 train-bnb-8.slurm
```

### :chains: 3.1. Concatenation

Make sure you have the following dataset file:

```
data/bnb/bnb_train.json
data/bnb/bnb_test.json
data/bnb/testset.json

```bash
python train_bnb.py \
  --from_pretrained vilbert.bin \
  --save_name concatenation \
  --separators \
  --min_captioned 7 \
  --masked_vision \
  --masked_language
```



### :busts_in_silhouette: 3.2. Image merging

Make sure you have the following dataset file:

```
data/bnb/merge+bnb_train.json
data/bnb/merge+bnb_test.json
data/bnb/merge+testset.json

```bash
python train_bnb.py \
  --from_pretrained vilbert.bin \
  --save_name image_merging \
  --prefix merge+ \
  --min_captioned 7 \
  --separators \
  --masked_vision \
  --masked_language
```


### 👨‍👩‍👧 3.3. Captionless insertion

Make sure you have the following dataset file:

```
data/bnb/2capt+bnb_train.json
data/bnb/2capt+bnb_test.json
data/bnb/2capt+testset.json

```bash
python train_bnb.py \
  --from_pretrained vilbert.bin \
  --save_name captionless_insertion \
  --prefix 2capt+ \
  --min_captioned 2 \
  --separators \
  --masked_vision \
  --masked_language
```

### 👣 3.4. Instruction rephrasing

Make sure you have the following dataset file:

```
data/bnb/np+bnb_train.json
data/bnb/np+bnb_test.json
data/bnb/np+testset.json
data/np_train.json

```bash
python train_bnb.py \
  --from_pretrained vilbert.bin \
  --save_name instruction_rephrasing \
  --prefix np+ \
  --min_captioned 7 \
  --separators \
  --masked_vision \
  --masked_language \
  --skeleton data/np_train.json
```

## :detective: 4. Fine-tuning on R2R in Discriminative Setting

First of all, you need to download the R2R data:

```
make r2r
```

### 4.1. Fine-tune with masking losses

```
python train.py \
  --from_pretrained bnb-pretrained.bin \
  --save_name r2rM \
  --masked_language --masked_vision --no_ranking
```

### 4.2. Fine-tune with the ranking and the shuffling loss

```
python train.py \
  --from_pretrained r2rM.bin \
  --save_name r2rRS \
  --shuffle_visual_features
```

### 4.3. Fine-tune with the ranking and the shuffling loss and the speaker data augmented

Download the [augmented paths](http://www.cs.unc.edu/~airsplay/aug_paths.json) from [EnvDrop](https://github.com/airsplay/R2R-EnvDrop):

```bash
make speaker
```

Then use the `train.py` script:

```bash
python train.py \
  --from_pretrained r2rM.bin \
  --save_name r2rRS \
  --shuffle_visual_features \
  --prefix aug+ \
  --beam_prefix aug_
```

You can download a pretrained model [from our model zoo](https://github.com/airbert-vln/model-zoos).

## :pregnant_woman: 5. Fine-tuning on REVERIE and R2R in Generative Setting

Please see the repository [dedicated for finetuning Airbert in generative setting](https://github.com/airbert-vln/airbert-recurrentvln).

## :four_leaf_clover: 6. Few-shot learning

You can build the exact same few-shot learning datasets from this command:

```bash
make fewshot
```


## Citing our paper

See the [BibTex file](https://airbert-vln.github.io/bibtex.txt).


