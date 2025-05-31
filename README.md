# Towards Universality

This repository contains the official implementation for the ICLR 2025 paper: [Towards Universality: Studying Mechanistic Similarity Across Language Model Architectures](https://arxiv.org/abs/2410.06672)

## Overview

This project investigates the mechanistic similarity across different language model architectures through:
- Feature-level analysis (using [Language-Model-SAEs](https://github.com/OpenMOSS/Language-Model-SAEs) for training SAEs)
- Neuron-level analysis
- Circuit-level analysis (focusing on induction circuits)

We study three representative architectures:
- Transformer (Pythia)
- Mamba (State Space Model)
- RWKV (Linear Attention)

## Acknowledgements

This work builds upon:
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens): A library for mechanistic interpretability of GPT-style language models
- [MambaLens](https://github.com/Phylliida/MambaLens): Mamba support for transformer lens

## Installation
Clone the repository
```bash
git clone https://github.com/SmallMelon-L/Universality.git

cd universality
```

Install dependencies

```bash
pip install -r requirements.txt
```

## Citation

If you find this work useful, please cite it as follows:

```bibtex
@inproceedings{Wang2025Universality,
  author       = {Junxuan Wang and
                  Xuyang Ge and
                  Wentao Shu and
                  Qiong Tang and
                  Yunhua Zhou and
                  Zhengfu He and
                  Xipeng Qiu},
  title        = {Towards Universality: Studying Mechanistic Similarity Across Language
                  Model Architectures},
  booktitle    = {The Thirteenth International Conference on Learning Representations,
                  {ICLR} 2025, Singapore, April 24-28, 2025},
  publisher    = {OpenReview.net},
  year         = {2025},
  url          = {https://openreview.net/forum?id=2J18i8T0oI},
  timestamp    = {Thu, 15 May 2025 17:19:06 +0200},
  biburl       = {https://dblp.org/rec/conf/iclr/WangGSTZHQ25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
