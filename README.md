<h1 align="center">🔍 💡 GIVE: Structured Reasoning of Large Language Models
with Knowledge-Graph-Inspired Veracity Extrapolation</h1>

Paper link: [arXiv](https://arxiv.org/abs/2410.08475)


The success of **DeepSeek-R1** has spotlighted **GRPO (Group Relative Policy Optimization)** as a key reinforcement learning method for large reasoning models.
However, GRPO suffers several key limitations including entropy collapse, difficulty bias, etc. 

*How can we design more effective optimization methods for reinforcing large reasoning models
in a principled manner without inheriting the limitations of GRPO?*


We analyzed GRPO and its variants (Dr. GRPO, DAPO, etc) under a binary reward setting and uncovered two core insights:

* ⚠️ GRPO suffers from **question-level difficulty bias** for its discriminative objective
* 🔍 GRPO has a surprising connection to **discriminative learning** techniques, particularly AUC maximization

---

### 💡 Introducing **DisCO** — *Discriminative Constrained Optimization*

**DisCO** is a new RL framework grounded in **discriminative learning**. It trains models by **increasing scores for positive answers while decreasing those for negatives**, enabling:

* ⚡ Faster convergence
* 🔒 More stable optimization
* 🔁 Long-lasting training dynamics

---

### 🔍 Why DisCO?

* ❌ **No more difficulty bias** – replaces group-relative objective with discriminative objectives
* 🔄 **No clipping operations** – uses non-clipping scoring functions (e.g., log-likelihood, likelihood ratio) for smoother learning
* 📉 **Stable training** – via simple constrained optimization to keep KL divergence in check
* ⚖️ **Handles sparse rewards** – robust to imbalanced data with advanced discriminative approaches

---
- [Getting Started](#getting-started)
    - [Installation](#installation)
    - [Datasets](#datasets)
    - [Training](#training)
    - [Evaluation](#evaluation)
- [Citing DisCO](#citing-disco)

## Getting Started
### Installation
```bash
# Recommend Python 3.10.
conda create -n disco python=3.10
conda activate disco
cd DisCO
pip install -e ./verl
pip install -e ./deepscaler
pip install wandb
```

### Datasets

Datesets utilized in our training are included in the `datasets` folder. Feel free to adapt  file `scripts/data/deepscaler_dataset.py` to generate your own datasets.



### Training

We provide training scripts for both single-node and multi-node setups in `scripts/train/`.

#### Single-Node Training (8 GPUs)
We start with one node for training 1.5b Qwen models with 8k context, with 8 A100-80GB GPUs. For example, let's run DisCO algorithm with `log likelihood` as the score function:
```bash

bash ./scripts/train/run_disco_logL_1.5b_8k.sh   #### DisCO with `log likelihood`
# bash ./scripts/train/run_disco_Lratio_1.5b_8k.sh   #### DisCO with `likelihood ratio`
# bash ./scripts/train/run_discob_logL_1.5b_8k.sh    #### DisCO-b with `log likelihood`
# bash ./scripts/train/run_discob_Lratio_1.5b_8k.sh  #### DisCO-b with `likelihood ratio`
```

#### Multi-Node Training

To train with longer context or larger models, multi-node training is necessary. To achieve this, follow these steps:

1. On the head node:
```bash
# Set XFormers backend to avoid CUDA errors
export VLLM_ATTENTION_BACKEND=XFORMERS
# Start Ray head node
ray start --head
```

2. On each worker node:
```bash
# Set XFormers backend to avoid CUDA errors
export VLLM_ATTENTION_BACKEND=XFORMERS
# Connect to head node (replace with your head node's address)
ray start --address=[RAY_ADDRESS]
```

3. Finally, on the head node, run the training script, such as:
```bash
bash ./scripts/train/run_disco_logL_7b_8k.sh
```


## Evaluation

Our evaluation scripts automatically runs vLLM to generate 16 samples for each problem. To run our evaluation scripts, run:
```bash
./scripts/eval/eval_model.sh --model [CHECKPOINT_PATH] --datasets [DATASET1] [DATASET2] --output-dir [OUTPUT_DIR]
```

We report Pass@1 accuracy averaged over 16 samples for each problem. To replicate our reported numbers, for example, run:
<!-- Notably, our `DeepScaleR-1.5B-Preview` surpasses many open-source 7B models!  -->

```bash
./scripts/eval/eval_model.sh --model ganglii/DisCO-1.5B-logL --datasets aime aime25 math amc minerva olympiad_bench --output-dir ./val_results/DisCO-1.5B-logL
```

## Acknowledgements
- Our training pipeline is built on the Github repository [deepscaler](https://github.com/agentica-project/rllm). We thank the authors for open-sourcing their code.





## Citing DisCO

If you find DisCO useful in your research, please consider citing the following paper:
```bibtex
@article{li2025disco,
  title={DisCO: Reinforcing Large Reasoning Models with Discriminative Constrained Optimization},
  author={Li, Gang and Lin, Ming and Galanti, Tomer and Tu, Zhengzhong and Yang, Tianbao},
  journal={arXiv preprint arXiv:2505.12366},
  year={2025}
}
```
