<div align="center">
<h1>LoPA: Scaling dLLM Inference via Lookahead Parallel Decoding</h1>
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2508.09192"><b>📄 Paper (Coming Soon)</b></a> •
  <a href="https://github.com/zhijie-group/LoPA"><b>💻 GitHub</b></a> •
  <a href="https://huggingface.co/SJTU-Deng-Lab"><b>🤗 Hugging Face</b></a>
</p>

<hr>

**Lookahead Parallel Decoding (LoPA)** is a training-free, plug-and-play algorithm designed to break the parallelism bottleneck in Diffusion Large Language Models (dLLMs). By identifying that parallelism is highly sensitive to the Token Filling Order (TFO), LoPA actively searches for optimal TFOs to maximize future confidence.

Key features of LoPA include:
- **Massive Speedup:** Increases the Tokens Per Forward pass (TPF) of **D2F-Dream** to **10.1** on GSM8K and **D2F-DiffuCoder** to **8.3** on HumanEval+.
- **High Throughput:** Achieves a single-sample throughput of **1073.9 tokens/s** under multi-GPU deployment using a specialized Branch Parallel (BP) inference system.
- **Training-Free:** Works out-of-the-box with existing confidence-driven dLLMs (like D2F and Dream) without requiring weight updates.

<p align="center">
  <img src="docs/assets/img/figure1.png" width="100%" alt="Throughput performance">
  <br>
  <small style="color: gray;">Figure 1. Throughput performance of LoPA under guaranteed inference speed. LoPA accelerates the single-sample throughput for D2F-Dream to up to 1073.9 and 856.5 tokens/s on MBPP and GSM8K respectively, significantly outperforming baselines.</small>
</p>

## 🔥 News
* **Dec 18, 2025:** We released the code and paper for LoPA!
* **Dec 2025:** LoPA achieves >1000 tokens/s on Ascend 910C hardware.

## Contents
- [🤔 How It Works](#-how-it-works)
- [📊 Performance Highlights](#-performance-highlights)
- [⚙️ System Throughput](#️-system-throughput)
- [🚀 Usage Guide](#-usage-guide)
- [©️ Citation](#️-citation)

## 🤔 How It Works

Standard dLLM decoding greedily fills tokens with the highest current confidence, which often leads to suboptimal paths that restrict future parallelism. LoPA solves this by "looking ahead":

1.  **Anchor Branch:** Maintains the standard confidence-driven path.
2.  **Lookahead Branches:** Spawns $k$ parallel branches exploring alternative high-confidence Token Filling Orders (TFOs).
3.  **Parallel Verification:** Verifies all branches in a single forward pass and selects the one with the highest **Branch Confidence** (potential for future parallelism).

<p align="center">
  <img src="docs/assets/img/figure3.png" width="100%" alt="Overview of LoPA">
  <br>
  <small style="color: gray;">Figure 2. Overview of Lookahead Parallel Decoding (LoPA). In each iteration, LoPA generates a anchor branch alongside multiple lookahead branches by independently sampling high-confidence positions. A branch confidence verification mechanism then evaluates all branches in parallel to select the optimal path.</small>
</p>

## 📊 Performance Highlights

LoPA demonstrates significant improvements in Tokens Per Forward pass (TPF) and overall throughput across mathematical reasoning and code generation tasks. It establishes a clear, controllable speed-accuracy trade-off.

<p align="center">
  <img src="docs/assets/img/figure4.png" width="100%" alt="Scaling Curves">
  <br>
  <small style="color: gray;">Figure 3. Scaling Curves of LoPA. LoPA scales the tokens per forward pass (TPS) for D2F-Dream and D2F-DiffuCoder to up to 10.1 and 8.3 on GSM8k and HumanEval+ respectively, with comparable performance.</small>
</p>

<p align="center">
  <img src="docs/assets/img/figure2.png" width="100%" alt="Scaling Analysis">
  <br>
  <small style="color: gray;">Figure 4. Scaling analysis of LoPA on D2F-Dream with varying branch counts. The results illustrate that LoPA effectively scales the TPF of D2F to a peak exceeding 10, thereby significantly reducing the total number of decoding steps.</small>
</p>

### Accuracy-Preserving Parallelism

<div align="center">
<strong>Table 1. Accuracy-preserving parallelism scaling of Dream on multiple benchmarks.</strong>
<table style="width:100%; text-align: center; border-collapse: collapse;">
    <thead>
        <tr style="background-color: #f2f2f2;">
            <th rowspan="2" style="border: 1px solid #ddd; padding: 8px;">Model</th>
            <th rowspan="2" style="border: 1px solid #ddd; padding: 8px;">Decoding algo</th>
            <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">MBPP 3-shot</th>
            <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">Math 4-shot</th>
            <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">HumanEval 0-shot</th>
            <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">GSM8K 4-shot</th>
        </tr>
        <tr style="background-color: #f2f2f2;">
            <th style="border: 1px solid #ddd; padding: 8px;">TPF</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Score</th>
            <th style="border: 1px solid #ddd; padding: 8px;">TPF</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Score</th>
            <th style="border: 1px solid #ddd; padding: 8px;">TPF</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Score</th>
            <th style="border: 1px solid #ddd; padding: 8px;">TPF</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Score</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">Dream</td>
            <td style="border: 1px solid #ddd; padding: 8px;">Vanilla</td>
            <td style="border: 1px solid #ddd; padding: 8px;">1.0</td>
            <td style="border: 1px solid #ddd; padding: 8px;"><b>56.2</b></td>
            <td style="border: 1px solid #ddd; padding: 8px;">1.0</td>
            <td style="border: 1px solid #ddd; padding: 8px;">33.7</td>
            <td style="border: 1px solid #ddd; padding: 8px;">1.0</td>
            <td style="border: 1px solid #ddd; padding: 8px;">55.5</td>
            <td style="border: 1px solid #ddd; padding: 8px;">1.0</td>
            <td style="border: 1px solid #ddd; padding: 8px;">72.6</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">Dream</td>
            <td style="border: 1px solid #ddd; padding: 8px;">Fast-dLLM</td>
            <td style="border: 1px solid #ddd; padding: 8px;">1.9</td>
            <td style="border: 1px solid #ddd; padding: 8px;">55.6</td>
            <td style="border: 1px solid #ddd; padding: 8px;">1.9</td>
            <td style="border: 1px solid #ddd; padding: 8px;"><b>37.6</b></td>
            <td style="border: 1px solid #ddd; padding: 8px;">1.8</td>
            <td style="border: 1px solid #ddd; padding: 8px;">55.5</td>
            <td style="border: 1px solid #ddd; padding: 8px;">2.1</td>
            <td style="border: 1px solid #ddd; padding: 8px;">72.6</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">Dream</td>
            <td style="border: 1px solid #ddd; padding: 8px;">LoPA</td>
            <td style="border: 1px solid #ddd; padding: 8px;">3.3</td>
            <td style="border: 1px solid #ddd; padding: 8px;">54.8</td>
            <td style="border: 1px solid #ddd; padding: 8px;">3.4</td>
            <td style="border: 1px solid #ddd; padding: 8px;">37.0</td>
            <td style="border: 1px solid #ddd; padding: 8px;">2.9</td>
            <td style="border: 1px solid #ddd; padding: 8px;">53.0</td>
            <td style="border: 1px solid #ddd; padding: 8px;">3.1</td>
            <td style="border: 1px solid #ddd; padding: 8px;">73.3</td>
        </tr>
        <tr style="background-color: #fafafa;">
            <td style="border: 1px solid #ddd; padding: 8px;">D2F-Dream</td>
            <td style="border: 1px solid #ddd; padding: 8px;">Vanilla</td>
            <td style="border: 1px solid #ddd; padding: 8px;">2.3</td>
            <td style="border: 1px solid #ddd; padding: 8px;">53.8</td>
            <td style="border: 1px solid #ddd; padding: 8px;">2.6</td>
            <td style="border: 1px solid #ddd; padding: 8px;">36.8</td>
            <td style="border: 1px solid #ddd; padding: 8px;">2.5</td>
            <td style="border: 1px solid #ddd; padding: 8px;"><b>56.1</b></td>
            <td style="border: 1px solid #ddd; padding: 8px;">3.1</td>
            <td style="border: 1px solid #ddd; padding: 8px;"><b>78.5</b></td>
        </tr>
        <tr style="background-color: #e6f7ff;">
            <td style="border: 1px solid #ddd; padding: 8px;">D2F-Dream</td>
            <td style="border: 1px solid #ddd; padding: 8px;">LoPA (Ours)</td>
            <td style="border: 1px solid #ddd; padding: 8px;"><b>5.4</b></td>
            <td style="border: 1px solid #ddd; padding: 8px;">56.0</td>
            <td style="border: 1px solid #ddd; padding: 8px;"><b>8.0</b></td>
            <td style="border: 1px solid #ddd; padding: 8px;">35.2</td>
            <td style="border: 1px solid #ddd; padding: 8px;"><b>6.3</b></td>
            <td style="border: 1px solid #ddd; padding: 8px;"><b>56.1</b></td>
            <td style="border: 1px solid #ddd; padding: 8px;"><b>10.1</b></td>
            <td style="border: 1px solid #ddd; padding: 8px;">73.8</td>
        </tr>
    </tbody>
</table>
</div>
<br>

<div align="center">
<strong>Table 2. Accuracy-preserving parallelism scaling of DiffuCoder.</strong>
<table style="width:100%; text-align: center; border-collapse: collapse;">
    <thead>
        <tr style="background-color: #f2f2f2;">
            <th rowspan="2" style="border: 1px solid #ddd; padding: 8px;">Model</th>
            <th rowspan="2" style="border: 1px solid #ddd; padding: 8px;">Decoding algo</th>
            <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">MBPP+ 0-shot</th>
            <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">HumanEval+ 0-shot</th>
        </tr>
        <tr style="background-color: #f2f2f2;">
            <th style="border: 1px solid #ddd; padding: 8px;">TPF</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Score</th>
            <th style="border: 1px solid #ddd; padding: 8px;">TPF</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Score</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">DiffuCoder</td>
            <td style="border: 1px solid #ddd; padding: 8px;">Vanilla</td>
            <td style="border: 1px solid #ddd; padding: 8px;">1.0</td>
            <td style="border: 1px solid #ddd; padding: 8px;"><b>61.9</b></td>
            <td style="border: 1px solid #ddd; padding: 8px;">1.0</td>
            <td style="border: 1px solid #ddd; padding: 8px;">65.2</td>
        </tr>
        <tr style="background-color: #fafafa;">
            <td style="border: 1px solid #ddd; padding: 8px;">D2F-DiffuCoder</td>
            <td style="border: 1px solid #ddd; padding: 8px;">Vanilla</td>
            <td style="border: 1px solid #ddd; padding: 8px;">2.2</td>
            <td style="border: 1px solid #ddd; padding: 8px;"><b>61.9</b></td>
            <td style="border: 1px solid #ddd; padding: 8px;">2.2</td>
            <td style="border: 1px solid #ddd; padding: 8px;"><b>65.9</b></td>
        </tr>
        <tr style="background-color: #e6f7ff;">
            <td style="border: 1px solid #ddd; padding: 8px;">D2F-DiffuCoder</td>
            <td style="border: 1px solid #ddd; padding: 8px;">LoPA (Ours)</td>
            <td style="border: 1px solid #ddd; padding: 8px;"><b>6.7</b></td>
            <td style="border: 1px solid #ddd; padding: 8px;">61.6</td>
            <td style="border: 1px solid #ddd; padding: 8px;"><b>8.3</b></td>
            <td style="border: 1px solid #ddd; padding: 8px;">64.0</td>
        </tr>
    </tbody>
</table>
</div>

## ⚙️ System Throughput

To fully exploit LoPA’s parallelism, we designed **LoPA-Dist**, a distributed inference system utilizing Branch Parallelism (BP).

<p align="center">
  <img src="docs/assets/img/figure5.png" width="100%" alt="System Design">
  <br>
  <small style="color: gray;">Figure 5. Overview of LoPA Branch Parallel Distributed Inference System Design. A key distinction lies in the KV cache management protocol tailored for different backends.</small>
</p>

The system distributes candidate branches across multiple GPUs for concurrent processing. We provide two specialized implementations:
* **LoPA-Dist-NV (CUDA):** Optimized for low latency using static KV cache and a two-phase update protocol.
* **LoPA-Dist-Ascend (Ascend 910C):** Optimized for high throughput using hybrid parallelism and graph compilation.

<div align="center">
<strong>Table 3. System performance of D2F-Dream under guaranteed inference speed.</strong>
<table style="width:100%; text-align: center; border-collapse: collapse;">
    <thead>
        <tr style="background-color: #f2f2f2;">
            <th rowspan="2" style="border: 1px solid #ddd; padding: 8px;">Model</th>
            <th rowspan="2" style="border: 1px solid #ddd; padding: 8px;">Platform</th>
            <th colspan="4" style="border: 1px solid #ddd; padding: 8px;">MBPP</th>
            <th colspan="4" style="border: 1px solid #ddd; padding: 8px;">GSM8K</th>
        </tr>
        <tr style="background-color: #f2f2f2;">
            <th style="border: 1px solid #ddd; padding: 8px;">Avg TPS</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Max TPS</th>
            <th style="border: 1px solid #ddd; padding: 8px;">TPF</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Latency</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Avg TPS</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Max TPS</th>
            <th style="border: 1px solid #ddd; padding: 8px;">TPF</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Latency</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="2" style="border: 1px solid #ddd; padding: 8px;">D2F-Dream-Base</td>
            <td style="border: 1px solid #ddd; padding: 8px;">LoPA-Dist-NV</td>
            <td style="border: 1px solid #ddd; padding: 8px;">630.28</td>
            <td style="border: 1px solid #ddd; padding: 8px;">1472.37</td>
            <td style="border: 1px solid #ddd; padding: 8px;"><b>15.69</b></td>
            <td style="border: 1px solid #ddd; padding: 8px;">0.84</td>
            <td style="border: 1px solid #ddd; padding: 8px;">566.97</td>
            <td style="border: 1px solid #ddd; padding: 8px;">1305.86</td>
            <td style="border: 1px solid #ddd; padding: 8px;"><b>13.31</b></td>
            <td style="border: 1px solid #ddd; padding: 8px;">0.93</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">LoPA-Dist-Ascend</td>
            <td style="border: 1px solid #ddd; padding: 8px;"><b>1073.86</b></td>
            <td style="border: 1px solid #ddd; padding: 8px;"><b>2400.12</b></td>
            <td style="border: 1px solid #ddd; padding: 8px;">11.92</td>
            <td style="border: 1px solid #ddd; padding: 8px;"><b>0.78</b></td>
            <td style="border: 1px solid #ddd; padding: 8px;"><b>856.46</b></td>
            <td style="border: 1px solid #ddd; padding: 8px;"><b>2751.61</b></td>
            <td style="border: 1px solid #ddd; padding: 8px;">9.34</td>
            <td style="border: 1px solid #ddd; padding: 8px;"><b>0.75</b></td>
        </tr>
        <tr>
            <td rowspan="2" style="border: 1px solid #ddd; padding: 8px;">D2F-Dream-Instruct</td>
            <td style="border: 1px solid #ddd; padding: 8px;">LoPA-Dist-NV</td>
            <td style="border: 1px solid #ddd; padding: 8px;">543.32</td>
            <td style="border: 1px solid #ddd; padding: 8px;">1531.64</td>
            <td style="border: 1px solid #ddd; padding: 8px;"><b>9.45</b></td>
            <td style="border: 1px solid #ddd; padding: 8px;">0.16</td>
            <td style="border: 1px solid #ddd; padding: 8px;">536.71</td>
            <td style="border: 1px solid #ddd; padding: 8px;">1141.71</td>
            <td style="border: 1px solid #ddd; padding: 8px;"><b>11.41</b></td>
            <td style="border: 1px solid #ddd; padding: 8px;">0.29</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">LoPA-Dist-Ascend</td>
            <td style="border: 1px solid #ddd; padding: 8px;"><b>896.21</b></td>
            <td style="border: 1px solid #ddd; padding: 8px;"><b>2586.73</b></td>
            <td style="border: 1px solid #ddd; padding: 8px;">8.64</td>
            <td style="border: 1px solid #ddd; padding: 8px;"><b>0.11</b></td>
            <td style="border: 1px solid #ddd; padding: 8px;"><b>897.10</b></td>
            <td style="border: 1px solid #ddd; padding: 8px;"><b>1868.16</b></td>
            <td style="border: 1px solid #ddd; padding: 8px;">9.30</td>
            <td style="border: 1px solid #ddd; padding: 8px;"><b>0.21</b></td>
        </tr>
    </tbody>
</table>
</div>

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
bash test_dream_d2f_LoPA.sh

```


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

```

```
