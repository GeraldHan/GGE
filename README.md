# Greedy Gradient Ensemble for De-biased VQA
Code release for "Greedy Gradient Ensemble for Robust Visual Question Answering" (ICCV 2021, Oral). GGE can extend to other tasks with dataset biases.

```
@inproceedings{han2015greedy,
	title={Greedy Gradient Ensemble for Robust Visual Question Answering},
	author={Han, Xinzhe and Wang, Shuhui and Su, Chi and Huang, Qingming and Tian, Qi},
	booktitle={Proceedings of the IEEE international conference on computer vision},
	year={2021}
}
```

## Prerequisites

We use Anaconda to manage our dependencies . You will need to execute the following steps to install all dependencies:

- Edit the value for `prefix` variable in `requirements.yml` file, by assigning it the path to conda environment

- Then, install all dependencies using:
``conda env create -f requirements.yml``

- Change to the new environment:
``bias``


## Data Setup
- Download UpDn features from [google drive](https://drive.google.com/drive/folders/1IXTsTudZtYLqmKzsXxIZbXfCnys_Izxr) into `/data/detection_features` folder
- Download questions/answers for VQAv2 and VQA-CPv2 by executing `bash tools/download.sh`
- Download visual cues/hints provided in [A negative case analysis of visual grounding methods for VQA](https://drive.google.com/drive/folders/1fkydOF-_LRpXK1ecgst5XujhyQdE6It7?usp=sharing) into `data/hints`. Note that we use caption based hints for grounding-based method reproduction, CGR and CGW.
- Preprocess process the data with `bash tools/process.sh`

## Training GGE
Run
```
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cpv2 --mode MODE --debias gradient --topq 1 --topv -1 --qvp 5 --output [] 
```
to train a model.  In `main.py`, `import base_model` for UpDn baseline; `import base_model_ban as base_model` for BAN baseline; `import base_model_block as base_model` for S-MRL baseline.

Set `MODE` as `gge_iter` and `gge_tog` for our best performance model; `gge_d_bias` and `gge_q_bias` for single bias ablation; `base` for baseline model.

## Training ablations in Sec. 3 and Sec. 5
For models in Sec. 3, execute `from train_ab import train` and `import base_model_ab as base_model` in `main.py`. Run
```
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cpv2 --mode MODE --debias METHODS --topq 1 --topv -1 --qvp 5 --output [] 
```
METHODS `learned_mixin` for LMH, MODE `inv_sup` for inv_sup strategy, `v_inverse` for inverse hint. Note that the results for HINT$_inv$ is obtained by running the code from [A negative case analysis of visual grounding methods for VQA](https://drive.google.com/drive/folders/1fkydOF-_LRpXK1ecgst5XujhyQdE6It7?usp=sharing).

To test v_only model, `import base_model_v_only as base_model` in `main.py`.

To test RUBi and LMH+RUBi, run
```
CUDA_VISIBLE_DEVICES=0 python rubi_main.py --dataset cpv2 --mode MODE --output [] 
```
MODE `updn` is for RUBi, `lmh_rubi` is for LMH+RUBi.

## Testing
For test stage, we output the overall Acc, CGR, CGW and CGD at threshold 0.2. 
change base_model to corresponding model in `sensitivity.py` and run
```
CUDA_VISIBLE_DEVICES=0 python sensitivity.py --dataset cpv2 --debias METHOD --load_checkpoint_path logs/your_path --output your_path
```
## Visualization
We provide visualization in `visualization.ipynb`. If you want to see other visualization by yourself, download MS-COCO 2014 to `data/images`.

## Acknowledgements

This repo uses features from [A negative case analysis of visual grounding methods for VQA](https://github.com/erobic/negative_analysis_of_grounding). Some codes are modified from [CSS](https://github.com/chrisc36/bottom-up-attention-vqa) and [UpDn](https://github.com/chrisc36/bottom-up-attention-vqa).

