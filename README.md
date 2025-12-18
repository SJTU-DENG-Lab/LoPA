## LoPA: Scaling dLLM Inference via Lookahead Parallel Decoding

<p align="center">
  <a href="https://arxiv.org/abs/2508.09192"><b>📄 Paper (Coming Soon)</b></a> •
  <a href="https://github.com/zhijie-group/LoPA"><b>💻 GitHub</b></a> •
  <a href="https://huggingface.co/SJTU-Deng-Lab"><b>🤗 Hugging Face</b></a>
</p>


<p align="center">
  <br>
  <small><b>Throughput Performance of LoPA:</b> LoPA accelerates the single-sample throughput for D2F-Dream to up to <b>1073.9 tokens/s</b> and <b>856.5 tokens/s</b> on MBPP and GSM8K respectively, significantly outperforming baselines.</small>
</p>

<hr>

**Lookahead Parallel Decoding (LoPA)** is a training-free, plug-and-play algorithm designed to break the parallelism bottleneck in Diffusion Large Language Models (dLLMs). By identifying that parallelism is highly sensitive to the Token Filling Order (TFO), LoPA actively searches for optimal TFOs to maximize future confidence.

Key features of LoPA include:
- **Massive Speedup:** Increases the Tokens Per Forward pass (TPF) of **D2F-Dream** to **10.1** on GSM8K and **D2F-DiffuCoder** to **8.3** on HumanEval+.
- **High Throughput:** Achieves a single-sample throughput of **1073.9 tokens/s** under multi-GPU deployment using a specialized Branch Parallel (BP) inference system.
- **Training-Free:** Works out-of-the-box with existing confidence-driven dLLMs (like D2F and Dream) without requiring weight updates.

## 🔥 News
* **Dec 18, 2025:** We released the code and paper for LoPA!
* **Dec 2025:** LoPA achieves >1000 tokens/s on Ascend 910C hardware.

## Contents
- [🤔 How It Works](#-how-it-works)
- [📊 Performance Highlights](#-performance-highlights)
- [🚀 Usage Guide](#-usage-guide)
- [🙏 Acknowledgements](#-acknowledgements)
- [©️ Citation](#️-citation)

## 🤔 How It Works

Standard dLLM decoding greedily fills tokens with the highest current confidence, which often leads to suboptimal paths that restrict future parallelism. LoPA solves this by "looking ahead":

1.  **Anchor Branch:** Maintains the standard confidence-driven path.
2.  **Lookahead Branches:** Spawns $k$ parallel branches exploring alternative high-confidence Token Filling Orders (TFOs).
3.  **Parallel Verification:** Verifies all branches in a single forward pass and selects the one with the highest **Branch Confidence** (potential for future parallelism).


## 📊 Performance Highlights

LoPA demonstrates significant improvements in Tokens Per Forward pass (TPF) and overall throughput across mathematical reasoning and code generation tasks.

<center>

**Performance on D2F-Dream (7B)**
<table style="width:100%; border-collapse: collapse; text-align: center;">
  <thead style="background-color:#f2f2f2;">
    <tr>
      <th style="padding: 8px; border: 1px solid #ddd;">Benchmark</th>
      <th style="padding: 8px; border: 1px solid #ddd;">Metric</th>
      <th style="padding: 8px; border: 1px solid #ddd;">Vanilla D2F</th>
      <th style="padding: 8px; border: 1px solid #ddd;">D2F + LoPA (Ours)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2" style="padding: 8px; border: 1px solid #ddd; vertical-align: middle;"><strong>GSM8K (4-shot)</strong></td>
      <td style="padding: 8px; border: 1px solid #ddd;">TPF ↑</td>
      <td style="padding: 8px; border: 1px solid #ddd;">3.1</td>
      <td style="padding: 8px; border: 1px solid #ddd;"><strong>10.1 <font color="green">(3.2x)</font></strong></td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">Score ↑</td>
      <td style="padding: 8px; border: 1px solid #ddd;">78.5</td>
      <td style="padding: 8px; border: 1px solid #ddd;">73.8</td>
    </tr>
    <tr>
      <td rowspan="2" style="padding: 8px; border: 1px solid #ddd; vertical-align: middle; background-color: #fafafa;"><strong>MBPP (3-shot)</strong></td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;">TPF ↑</td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;">2.3</td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;"><strong>3.3 <font color="green">(1.4x)</font></strong></td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;">Score ↑</td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;">53.8</td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;">54.8</td>
    </tr>
  </tbody>
</table>

**Performance on D2F-DiffuCoder (7B)**
<table style="width:100%; border-collapse: collapse; text-align: center;">
  <thead style="background-color:#f2f2f2;">
    <tr>
      <th style="padding: 8px; border: 1px solid #ddd;">Benchmark</th>
      <th style="padding: 8px; border: 1px solid #ddd;">Metric</th>
      <th style="padding: 8px; border: 1px solid #ddd;">Vanilla D2F</th>
      <th style="padding: 8px; border: 1px solid #ddd;">D2F + LoPA (Ours)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2" style="padding: 8px; border: 1px solid #ddd; vertical-align: middle;"><strong>HumanEval+ (0-shot)</strong></td>
      <td style="padding: 8px; border: 1px solid #ddd;">TPF ↑</td>
      <td style="padding: 8px; border: 1px solid #ddd;">2.2</td>
      <td style="padding: 8px; border: 1px solid #ddd;"><strong>8.3 <font color="green">(3.7x)</font></strong></td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">Score ↑</td>
      <td style="padding: 8px; border: 1px solid #ddd;">65.9</td>
      <td style="padding: 8px; border: 1px solid #ddd;">64.0</td>
    </tr>
    <tr>
      <td rowspan="2" style="padding: 8px; border: 1px solid #ddd; vertical-align: middle; background-color: #fafafa;"><strong>MBPP+ (0-shot)</strong></td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;">TPF ↑</td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;">2.2</td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;"><strong>6.7 <font color="green">(3.0x)</font></strong></td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;">Score ↑</td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;">61.9</td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;">61.6</td>
    </tr>
  </tbody>
</table>

</center>

## 🚀 Usage Guide

### 1. Installation

First, clone the repository and install the dependencies.

```shell
# Clone the repository
git clone [https://github.com/zhijie-group/LoPA.git](https://github.com/zhijie-group/LoPA.git)
cd LoPA

# Create environment (Recommended)
conda create -n lopa python=3.10
conda activate lopa

# Install dependencies
pip install -r requirements.txt

```

### 2. Running Experiments

The repository is organized into two main directories corresponding to the models used in our paper: `scale_diffucoder_d2f` and `scale_dream_d2f`.

#### D2F-DiffuCoder Experiments

To evaluate **DiffuCoder** with LoPA on coding benchmarks (HumanEval/MBPP), navigate to the `scale_diffucoder_d2f` directory:

```shell
cd scale_diffucoder_d2f

```

**Run LoPA on HumanEval:**

```shell
bash test_diffucoder_lopa_humaneval.sh

```

**Run LoPA on MBPP:**

```shell
bash test_diffucoder_lopa_mbpp.sh

```

*Note: You can also run the standard D2F baselines using the `test_diffucoder_d2f_*.sh` scripts provided in the same folder.*

#### D2F-Dream Experiments

To evaluate **D2F-Dream** (optimized for mathematical reasoning and general tasks), navigate to the `scale_dream_d2f` directory:

```shell
cd scale_dream_d2f

```

**Run Dream Evaluation:**

```shell
bash eval_dream7.sh

```

*Note: This script will execute the `scale_dream_d2f.py` pipeline. Ensure you have downloaded the necessary model checkpoints into the `model_cache` directory or configured the paths in the script accordingly.*

## 🙏 Acknowledgements

This work is based on **D2F**, **Dream**, and **DiffuCoder**. We thank the authors for their open-source contributions.

## ©️ Citation

If you find LoPA useful for your research, please cite our paper:

```bibtex
@article{xu2025lopa,
  title={LoPA: Scaling dLLM Inference via Lookahead Parallel Decoding},
  author={Xu, Chenkai and Jin, Yijie and Li, Jiajun and Tu, Yi and Long, Guoping and Tu, Dandan and Hou, Tianqi and Yan, Junchi and Deng, Zhijie},
  journal={Preprint},
  year={2025}
}

```


