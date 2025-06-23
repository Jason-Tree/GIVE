<h1 align="center">🔍 💡 GIVE: Structured Reasoning of Large Language Models
with Knowledge-Graph-Inspired Veracity Extrapolation</h1>

Paper link: [arXiv](https://arxiv.org/abs/2410.08475)


Large Language Models stumble on complex-domain questions because of lacking domain-specific internal knowledge. Textual or Knowledge Graph based RAG approaches assume the comprehensiveness of the accssible non-parametric knowledge base, which is costly or not feasible to maintain in scientific domains.

*Can we combine the parametric knowledge and limited non-parametric information to boost human-like associative reasoning?*

---

### Introducing **GIVE** — *Graph Inspired Veracity Extrapolation*

**GIVE** is a retreival and reasoning framework utilizing the structured information in **knowledge graphs**. We argue that in the era of large reasoning models, we need agentic frameworks that go beyond gold context retrieval and self-reflection style reasoning, the problem of **retrieval** and **reasoning** should be unified to advance automatic problem-solving in the hard domain. 

---

### 🔍 Why GIVE?

* ⚖️ **Handles both comprehensive and small KG** – extrapolate and populate the limited KG information
* 🔄 **Interpretable associative reasoning** – associate the structured knowledge with the important queried concepts and relations
* 📉 **Designed for hard domain QA that is beyond the training knowledge** – via "GIVE"ing hints to the agent for problem solving, rather than gold context retrieval
---
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Citing GIVE](#citing-give)

### Inference

We provide all KG and QA datasets in `data.zip` at `https://drive.google.com/drive/folders/1YaekQcYsagnmyn1dh-s605cgqb3L4f4L?usp=drive_link`, unzip this file before running.

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


## Citing GIVE

If you find the data or code in this repo useful in your research, please consider citing our paper:
```bibtex
@article{he2025givestructuredreasoninglarge,
      title={GIVE: Structured Reasoning of Large Language Models with Knowledge Graph Inspired Veracity Extrapolation}, 
      author={Jiashu He and Mingyu Derek Ma and Jinxuan Fan and Dan Roth and Wei Wang and Alejandro Ribeiro},
      year={2025},
      eprint={2410.08475},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2410.08475}, 
}
```
