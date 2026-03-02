# Simulating Distribution Dynamics: Liquid Temporal Feature Evolution for Single-Domain Generalized Object Detection

[ [Paper][Simulating Distribution Dynamics: Liquid Temporal Feature Evolution for Single-Domain Generalized Object Detection](https://arxiv.org/pdf/2511.09909)) ]

### Installation
Our code is based on [Detectron2](https://github.com/facebookresearch/detectron2) and requires python >= 3.8

Install the required packages
```
pip install -r requirements.txt
```

### Datasets
Set the environment variable DETECTRON2_DATASETS to the parent folder of the datasets

```
    path-to-parent-dir/
        /diverseWeather
            /daytime_clear
            /daytime_foggy
            ...
        /comic
        /watercolor
        /VOC2007
        /VOC2012 
```
### Training
We train our models using four NVIDIA RTX 4090 GPUs.

```
python train.py --config-file configs/diverse_weather_c.yaml 
```

### Citation
```bibtex
@inproceedings{LTFE,
  title={Simulating Distribution Dynamics: Liquid Temporal Feature Evolution for Single-Domain Generalized Object Detection},
  author={Zihao Zhang and Yang Li and Aming Wu and Yahong Han},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}

```
