# TransECE

## Introduction

This repository was used in our paper:  
  
[**Transition-based Directed Graph Construction for Emotion-Cause Pair Extraction.**](https://aclanthology.org/2020.acl-main.342.pdf) Chuang Fan, Chaofa Yuan, Jiachen Du, Lin Gui, Min Yang, Ruifeng Xu. ACL 2020.
  
Please cite our paper if you use this code.  

## Requirements

* Python >= 3.7
* [MindSpore](https://mindspore.cn/) >= 1.7.0
* BERT - Our bert model is adapted from this implementation: https://github.com/lvyufeng/cybertron

## Usage

```bash
python run.py
```

## Citation

The BibTex of the citation is as follow:

```bibtex
@inproceedings{fan-etal-2020-transition,
    title = "Transition-based Directed Graph Construction for Emotion-Cause Pair Extraction",
    author = "Fan, Chuang  and
      Yuan, Chaofa  and
      Du, Jiachen  and
      Gui, Lin  and
      Yang, Min  and
      Xu, Ruifeng",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.acl-main.342",
    doi = "10.18653/v1/2020.acl-main.342",
    pages = "3707--3717",
    abstract = "Emotion-cause pair extraction aims to extract all potential pairs of emotions and corresponding causes from unannotated emotion text. Most existing methods are pipelined framework, which identifies emotions and extracts causes separately, leading to a drawback of error propagation. Towards this issue, we propose a transition-based model to transform the task into a procedure of parsing-like directed graph construction. The proposed model incrementally generates the directed graph with labeled edges based on a sequence of actions, from which we can recognize emotions with the corresponding causes simultaneously, thereby optimizing separate subtasks jointly and maximizing mutual benefits of tasks interdependently. Experimental results show that our approach achieves the best performance, outperforming the state-of-the-art methods by 6.71{\%} (p{\textless}0.01) in F1 measure.",
}
```