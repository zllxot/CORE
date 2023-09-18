# CORE: Cooperative Reconstruction for Multi-Agent Perception

![](figures/framework.png)

> [**CORE: Cooperative Reconstruction for Multi-Agent Perception**](https://arxiv.org/abs/2307.11514),            
> Binglu Wang, Lei Zhang, Zhaozhong Wang, Yongqiang Zhao, and [Tianfei Zhou](https://www.tfzhou.com/) <br>
> *ICCV 2023 ([arXiv 2307.11514](https://arxiv.org/abs/2307.11514))*

## Abstract

This paper presents CORE, a conceptually simple, effective and communication-efficient model for multi-agent cooperative perception.  It addresses the task from a novel perspective of cooperative reconstruction, based on two key insights: 1) cooperating agents together provide a more holistic observation of the environment, and 2) the holistic observation can serve as valuable supervision to explicitly guide the model  learning how to reconstruct the ideal observation based on collaboration.  CORE instantiates the idea with three major components: a compressor for each agent to create more compact feature representation for  efficient broadcasting, a lightweight attentive collaboration component for cross-agent message aggregation, and a reconstruction module to  reconstruct the observation based on aggregated feature representations. This learning-to-reconstruct idea is task-agnostic, and offers clear and reasonable supervision to inspire more effective collaboration, eventually  promoting perception tasks. We validate CORE on OPV2V, a large-scale multi-agent percetion dataset, in  two tasks, i.e., 3D object detection and semantic segmentation. Results demonstrate that CORE achieves state-of-the-art performance, and is more communication-efficient.

## Installation
The code is build upon [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) codebase and using following library versions:

* python 3.7
* pytorch 1.12.1
* cudatoolkit 11.3.1

Clone the repository:
```bash
git clone https://github.com/zllxot/CORE.git
```

Create a conda virtual environment:
```bash
conda create -n core python=3.7
conda activate core
```

Install pytorch, cudatoolkit, and torchvision:
```bash
conda install pytorch=1.12.1=py3.7_cuda11.3_cudnn8.3.2_0 torchvision=0.13.1=py37_cu113
```

Install spconv 2.x:
```bash
pip install spconv-cu113
```

Install dependencies:
```bash
cd core
pip install -r requirements.txt
python setup.py develop
```

Install the CUDA version of the NMS calculation:
```bash
python opencood/utils/setup.py build_ext --inplace
```

## Dataset
Our experiments are conducted on the OPV2V dataset.  You can learn more about this dataset by visiting the [website](https://mobility-lab.seas.ucla.edu/opv2v/).

## Quick Start
### Training:
We follow the same configuration as OpenCOOD, utilizing a YAML file to set all the training parameters. To train your model, run the following commands:
```python
python opencood/tools/train.py --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER} --half]
```
Arguments Explanation:
- `hypes_yaml`: the path of the training configuration file, e.g. `opencood/hypes_yaml/second_early_fusion.yaml`, meaning you want to train
an early fusion model which utilizes SECOND as the backbone. See [Tutorial 1: Config System](https://opencood.readthedocs.io/en/latest/md_files/config_tutorial.html) to learn more about the rules of the yaml files.
- `model_dir` (optional) : the path of the checkpoints. This is used to fine-tune the trained models. When the `model_dir` is
given, the trainer will discard the `hypes_yaml` and load the `config.yaml` in the checkpoint folder.
- `half` (optional): If set, the model will be trained with half precision. It cannot be set with multi-gpu training togetger.

To train on **multiple gpus**, run the following command:
```python
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env opencood/tools/train.py --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER}]
```

Here's a example of how to run the training script on a single GPU:
```python
python opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/voxelnet_core.yaml
```

### Evaluation:
Before you run the following command, first make sure the `validation_dir` in config.yaml under your checkpoint folder
refers to the testing dataset path, e.g. `opv2v_data_dumping/test`.

```python
python opencood/tools/inference.py --model_dir ${CHECKPOINT_FOLDER} --fusion_method ${FUSION_STRATEGY} [--show_vis] [--show_sequence]
```
Arguments Explanation:
- `model_dir`: the path to your saved model.
- `fusion_method`: indicate the fusion strategy, currently support 'early', 'late', and 'intermediate'.
- `show_vis`: whether to visualize the detection overlay with point cloud.
- `show_sequence` : the detection results will visualized in a video stream. It can NOT be set with `show_vis` at the same time.

## Acknowledgment
Gratitude to the creators and contributors of the following open-source cooperative perception works, codebases, and datasets, which played a crucial role in shaping this project:
- [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD)
- [OPV2V](https://mobility-lab.seas.ucla.edu/opv2v/)
- [CoPerception](https://github.com/coperception/coperception/tree/dd8fbb660300ee763e0bac870f63fc4987440a35)
- [V2X-Sim](https://ai4ce.github.io/V2X-Sim)
- [DiscoNet](https://github.com/ai4ce/DiscoNet)
- [CoBEVT](https://github.com/DerrickXuNu/CoBEVT)
