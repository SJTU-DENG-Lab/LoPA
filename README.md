<div align="center">
<h1>LoPA: Scaling dLLM Inference via Lookahead Parallel Decoding</h1>
</div>

<p align="center">
  <a href="paper/LoPA.pdf">📄 Paper</a> •
  <a href="https://zhijie-group.github.io/blogs/lopa/">📝 Blog</a> •
  <a href="https://github.com/zhijie-group/Diffulex">🚀 Engine</a> •
  <a href="https://huggingface.co/SJTU-Deng-Lab">🤗 Hugging Face</a>
</p>

<hr>

**Lookahead Parallel Decoding (LoPA)** is a training-free, plug-and-play algorithm designed to break the parallelism bottleneck in Diffusion Large Language Models (dLLMs). By identifying that parallelism is highly sensitive to the Token Filling Order (TFO), LoPA actively searches for optimal TFOs to maximize future confidence.

Key features of LoPA include:

* **Massive Speedup:** Increases the Tokens Per Forward pass (TPF) of **D2F-Dream** to **10.1** on GSM8K and **D2F-DiffuCoder** to **8.3** on HumanEval+.
* **High Throughput:** Achieves a single-sample throughput of **1073.9 tokens/s** under multi-GPU deployment using a specialized Branch Parallel (BP) inference system.
* **Training-Free:** Works out-of-the-box with existing confidence-driven dLLMs (like D2F and Dream) without requiring weight updates.

<p align="center">
<img src="docs/assets/img/figure1.png" width="100%" alt="Throughput performance">





<small style="color: gray;">Figure 1. Throughput performance of LoPA under guaranteed inference speed. LoPA accelerates the single-sample throughput for D2F-Dream to up to 1073.9 and 856.5 tokens/s on MBPP and GSM8K respectively, significantly outperforming baselines.</small>
</p>

## 🔥 News

* **Dec 22, 2025:** We released the code and paper for LoPA-Dist-NV!
* **Dec 18, 2025:** We released the code and paper for LoPA!
* **Dec 2025:** LoPA achieves >1000 tokens/s on Ascend 910C hardware.

## 🔮 Future Works

* **Diffulex:** We are working on a new inference framework for dLLMs, which is flexible and easy to extend. Diffulex supports multiple decoding strategies including D2F, BlockDiffusion, and Fast-dLLM-v2, which is soon to be released. **You can find the code [here](https://github.com/zhijie-group/Diffulex).**

* **LoPA-SDAR:** We will explore adapting LoPA to SDAR and other confidence-driven diffusion language models to further demonstrate its generalizability and effectiveness across diverse model architectures.

## Contents

* [🤔 How It Works](https://www.google.com/search?q=%23-how-it-works)
* [📊 Performance Highlights](https://www.google.com/search?q=%23-performance-highlights)
* [⚙️ System Throughput](https://www.google.com/search?q=%23%EF%B8%8F-system-throughput)
* [🚀 Usage Guide](https://www.google.com/search?q=%23-usage-guide)
* [©️ Citation](https://www.google.com/search?q=%23%EF%B8%8F-citation)

## 🤔 How It Works

Standard dLLM decoding greedily fills tokens with the highest current confidence, which often leads to suboptimal paths that restrict future parallelism. LoPA solves this by "looking ahead":

1. **Anchor Branch:** Maintains the standard confidence-driven path.
2. **Lookahead Branches:** Spawns  parallel branches exploring alternative high-confidence Token Filling Orders (TFOs).
3. **Parallel Verification:** Verifies all branches in a single forward pass and selects the one with the highest **Branch Confidence** (potential for future parallelism).

<p align="center">
<img src="docs/assets/img/figure3.png" width="100%" alt="Overview of LoPA">





<small style="color: gray;">Figure 2. Overview of Lookahead Parallel Decoding (LoPA). In each iteration, LoPA generates a anchor branch alongside multiple lookahead branches by independently sampling high-confidence positions. A branch confidence verification mechanism then evaluates all branches in parallel to select the optimal path.</small>
</p>

## 📊 Performance Highlights

LoPA demonstrates significant improvements in Tokens Per Forward pass (TPF) and overall throughput across mathematical reasoning and code generation tasks. It establishes a clear, controllable speed-accuracy trade-off.

<p align="center">
<img src="docs/assets/img/figure4.png" width="100%" alt="Scaling Curves">





<small style="color: gray;">Figure 3. Scaling Curves of LoPA. LoPA scales the TPF for D2F-Dream and D2F-DiffuCoder to up to 10.1 and 8.3 on GSM8k and HumanEval+ respectively, with comparable performance.</small>
</p>

<p align="center">
<img src="docs/assets/img/figure2.png" width="100%" alt="Scaling Analysis">





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
<td style="border: 1px solid #ddd; padding: 8px;">708.48</td>
<td style="border: 1px solid #ddd; padding: 8px;">1470.95</td>
<td style="border: 1px solid #ddd; padding: 8px;"><b>15.55</b></td>
<td style="border: 1px solid #ddd; padding: 8px;">0.74</td>
<td style="border: 1px solid #ddd; padding: 8px;">619.33</td>
<td style="border: 1px solid #ddd; padding: 8px;">1299.25</td>
<td style="border: 1px solid #ddd; padding: 8px;"><b>13.16</b></td>
<td style="border: 1px solid #ddd; padding: 8px;">0.85</td>
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
<td style="border: 1px solid #ddd; padding: 8px;">636.55</td>
<td style="border: 1px solid #ddd; padding: 8px;">1811.71</td>
<td style="border: 1px solid #ddd; padding: 8px;"><b>9.52</b></td>
<td style="border: 1px solid #ddd; padding: 8px;">0.14</td>
<td style="border: 1px solid #ddd; padding: 8px;">609.90</td>
<td style="border: 1px solid #ddd; padding: 8px;">1407.56</td>
<td style="border: 1px solid #ddd; padding: 8px;"><b>11.42</b></td>
<td style="border: 1px solid #ddd; padding: 8px;">0.26</td>
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

<div align="center">
<strong>Table 4. Performance ablation study of D2F-Dream models on different platforms, corresponding to settings S1-S18.</strong>
<table style="width:100%; text-align: center; border-collapse: collapse;">
<thead>
<tr style="background-color: #f2f2f2;">
<th rowspan="2" style="border: 1px solid #ddd; padding: 8px;">Model</th>
<th rowspan="2" style="border: 1px solid #ddd; padding: 8px;">Sys. Arch.</th>
<th rowspan="2" style="border: 1px solid #ddd; padding: 8px;">Settings</th>
<th colspan="4" style="border: 1px solid #ddd; padding: 8px;">MBPP 3-shot</th>
<th colspan="4" style="border: 1px solid #ddd; padding: 8px;">GSM8K 4-shot</th>
</tr>
<tr style="background-color: #f2f2f2;">
<th style="border: 1px solid #ddd; padding: 8px;">Avg TPS</th>
<th style="border: 1px solid #ddd; padding: 8px;">Max TPS</th>
<th style="border: 1px solid #ddd; padding: 8px;">Top-10 TPS</th>
<th style="border: 1px solid #ddd; padding: 8px;">Score</th>
<th style="border: 1px solid #ddd; padding: 8px;">Avg TPS</th>
<th style="border: 1px solid #ddd; padding: 8px;">Max TPS</th>
<th style="border: 1px solid #ddd; padding: 8px;">Top-10 TPS</th>
<th style="border: 1px solid #ddd; padding: 8px;">Score</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="18" style="border: 1px solid #ddd; padding: 8px;">D2F-Dream-Base</td>
<td rowspan="12" style="border: 1px solid #ddd; padding: 8px;">LoPA-Dist-NV</td>
<td style="border: 1px solid #ddd; padding: 8px;">S1</td>
<td style="border: 1px solid #ddd; padding: 8px;">415.19</td>
<td style="border: 1px solid #ddd; padding: 8px;">813.04</td>
<td style="border: 1px solid #ddd; padding: 8px;">720.35</td>
<td style="border: 1px solid #ddd; padding: 8px;">53.00</td>
<td style="border: 1px solid #ddd; padding: 8px;">345.52</td>
<td style="border: 1px solid #ddd; padding: 8px;">959.05</td>
<td style="border: 1px solid #ddd; padding: 8px;">704.39</td>
<td style="border: 1px solid #ddd; padding: 8px;">75.97</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">S2</td>
<td style="border: 1px solid #ddd; padding: 8px;">500.33</td>
<td style="border: 1px solid #ddd; padding: 8px;">1185.77</td>
<td style="border: 1px solid #ddd; padding: 8px;">874.87</td>
<td style="border: 1px solid #ddd; padding: 8px;">53.40</td>
<td style="border: 1px solid #ddd; padding: 8px;">402.52</td>
<td style="border: 1px solid #ddd; padding: 8px;">913.12</td>
<td style="border: 1px solid #ddd; padding: 8px;">842.83</td>
<td style="border: 1px solid #ddd; padding: 8px;">73.54</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">S3</td>
<td style="border: 1px solid #ddd; padding: 8px;">550.37</td>
<td style="border: 1px solid #ddd; padding: 8px;">1472.41</td>
<td style="border: 1px solid #ddd; padding: 8px;">929.72</td>
<td style="border: 1px solid #ddd; padding: 8px;">51.20</td>
<td style="border: 1px solid #ddd; padding: 8px;">436.22</td>
<td style="border: 1px solid #ddd; padding: 8px;">994.82</td>
<td style="border: 1px solid #ddd; padding: 8px;">885.27</td>
<td style="border: 1px solid #ddd; padding: 8px;">71.19</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">S4</td>
<td style="border: 1px solid #ddd; padding: 8px;">589.22</td>
<td style="border: 1px solid #ddd; padding: 8px;">1576.93</td>
<td style="border: 1px solid #ddd; padding: 8px;">1006.57</td>
<td style="border: 1px solid #ddd; padding: 8px;">47.20</td>
<td style="border: 1px solid #ddd; padding: 8px;">475.58</td>
<td style="border: 1px solid #ddd; padding: 8px;">1203.61</td>
<td style="border: 1px solid #ddd; padding: 8px;">1028.15</td>
<td style="border: 1px solid #ddd; padding: 8px;">68.16</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">S5</td>
<td style="border: 1px solid #ddd; padding: 8px;">633.16</td>
<td style="border: 1px solid #ddd; padding: 8px;">1408.40</td>
<td style="border: 1px solid #ddd; padding: 8px;">963.67</td>
<td style="border: 1px solid #ddd; padding: 8px;">46.80</td>
<td style="border: 1px solid #ddd; padding: 8px;">516.85</td>
<td style="border: 1px solid #ddd; padding: 8px;">1212.65</td>
<td style="border: 1px solid #ddd; padding: 8px;">1055.08</td>
<td style="border: 1px solid #ddd; padding: 8px;">66.79</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">S6</td>
<td style="border: 1px solid #ddd; padding: 8px;">678.26</td>
<td style="border: 1px solid #ddd; padding: 8px;">1615.30</td>
<td style="border: 1px solid #ddd; padding: 8px;">1150.65</td>
<td style="border: 1px solid #ddd; padding: 8px;">41.80</td>
<td style="border: 1px solid #ddd; padding: 8px;">546.72</td>
<td style="border: 1px solid #ddd; padding: 8px;">1225.21</td>
<td style="border: 1px solid #ddd; padding: 8px;">1121.57</td>
<td style="border: 1px solid #ddd; padding: 8px;">64.14</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">S7</td>
<td style="border: 1px solid #ddd; padding: 8px;">466.27</td>
<td style="border: 1px solid #ddd; padding: 8px;">784.33</td>
<td style="border: 1px solid #ddd; padding: 8px;">764.52</td>
<td style="border: 1px solid #ddd; padding: 8px;">51.80</td>
<td style="border: 1px solid #ddd; padding: 8px;">416.91</td>
<td style="border: 1px solid #ddd; padding: 8px;">909.82</td>
<td style="border: 1px solid #ddd; padding: 8px;">841.95</td>
<td style="border: 1px solid #ddd; padding: 8px;">71.27</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">S8</td>
<td style="border: 1px solid #ddd; padding: 8px;">545.90</td>
<td style="border: 1px solid #ddd; padding: 8px;">1497.22</td>
<td style="border: 1px solid #ddd; padding: 8px;">927.67</td>
<td style="border: 1px solid #ddd; padding: 8px;">51.40</td>
<td style="border: 1px solid #ddd; padding: 8px;">486.94</td>
<td style="border: 1px solid #ddd; padding: 8px;">1176.14</td>
<td style="border: 1px solid #ddd; padding: 8px;">959.37</td>
<td style="border: 1px solid #ddd; padding: 8px;">68.39</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">S9</td>
<td style="border: 1px solid #ddd; padding: 8px;">588.00</td>
<td style="border: 1px solid #ddd; padding: 8px;">1584.28</td>
<td style="border: 1px solid #ddd; padding: 8px;">983.09</td>
<td style="border: 1px solid #ddd; padding: 8px;">48.60</td>
<td style="border: 1px solid #ddd; padding: 8px;">520.70</td>
<td style="border: 1px solid #ddd; padding: 8px;">1250.67</td>
<td style="border: 1px solid #ddd; padding: 8px;">1056.01</td>
<td style="border: 1px solid #ddd; padding: 8px;">68.01</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">S10</td>
<td style="border: 1px solid #ddd; padding: 8px;">637.38</td>
<td style="border: 1px solid #ddd; padding: 8px;">1552.56</td>
<td style="border: 1px solid #ddd; padding: 8px;">1028.97</td>
<td style="border: 1px solid #ddd; padding: 8px;">47.00</td>
<td style="border: 1px solid #ddd; padding: 8px;">558.01</td>
<td style="border: 1px solid #ddd; padding: 8px;">1115.26</td>
<td style="border: 1px solid #ddd; padding: 8px;">1071.66</td>
<td style="border: 1px solid #ddd; padding: 8px;">65.05</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">S11</td>
<td style="border: 1px solid #ddd; padding: 8px;">655.45</td>
<td style="border: 1px solid #ddd; padding: 8px;">1535.10</td>
<td style="border: 1px solid #ddd; padding: 8px;">1059.72</td>
<td style="border: 1px solid #ddd; padding: 8px;">43.80</td>
<td style="border: 1px solid #ddd; padding: 8px;">592.94</td>
<td style="border: 1px solid #ddd; padding: 8px;">1315.93</td>
<td style="border: 1px solid #ddd; padding: 8px;">1155.11</td>
<td style="border: 1px solid #ddd; padding: 8px;">64.44</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">S12</td>
<td style="border: 1px solid #ddd; padding: 8px;">708.48</td>
<td style="border: 1px solid #ddd; padding: 8px;">1470.95</td>
<td style="border: 1px solid #ddd; padding: 8px;">1132.78</td>
<td style="border: 1px solid #ddd; padding: 8px;">39.80</td>
<td style="border: 1px solid #ddd; padding: 8px;">619.33</td>
<td style="border: 1px solid #ddd; padding: 8px;">1299.25</td>
<td style="border: 1px solid #ddd; padding: 8px;">1201.18</td>
<td style="border: 1px solid #ddd; padding: 8px;">60.88</td>
</tr>
<tr>
<td rowspan="6" style="border: 1px solid #ddd; padding: 8px;">LoPA-Dist-Ascend</td>
<td style="border: 1px solid #ddd; padding: 8px;">S13</td>
<td style="border: 1px solid #ddd; padding: 8px;">615.74</td>
<td style="border: 1px solid #ddd; padding: 8px;">2173.7</td>
<td style="border: 1px solid #ddd; padding: 8px;">1253.07</td>
<td style="border: 1px solid #ddd; padding: 8px;">50.20</td>
<td style="border: 1px solid #ddd; padding: 8px;">492.94</td>
<td style="border: 1px solid #ddd; padding: 8px;">1337.60</td>
<td style="border: 1px solid #ddd; padding: 8px;">1158.18</td>
<td style="border: 1px solid #ddd; padding: 8px;">75.06</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">S14</td>
<td style="border: 1px solid #ddd; padding: 8px;">753.78</td>
<td style="border: 1px solid #ddd; padding: 8px;">2115.55</td>
<td style="border: 1px solid #ddd; padding: 8px;">1397.85</td>
<td style="border: 1px solid #ddd; padding: 8px;">50.20</td>
<td style="border: 1px solid #ddd; padding: 8px;">589.77</td>
<td style="border: 1px solid #ddd; padding: 8px;">1532.99</td>
<td style="border: 1px solid #ddd; padding: 8px;">1342.79</td>
<td style="border: 1px solid #ddd; padding: 8px;">72.86</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">S15</td>
<td style="border: 1px solid #ddd; padding: 8px;">842.97</td>
<td style="border: 1px solid #ddd; padding: 8px;">2470.79</td>
<td style="border: 1px solid #ddd; padding: 8px;">1538.16</td>
<td style="border: 1px solid #ddd; padding: 8px;">50.00</td>
<td style="border: 1px solid #ddd; padding: 8px;">644.34</td>
<td style="border: 1px solid #ddd; padding: 8px;">1723.19</td>
<td style="border: 1px solid #ddd; padding: 8px;">1476.24</td>
<td style="border: 1px solid #ddd; padding: 8px;">70.58</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">S16</td>
<td style="border: 1px solid #ddd; padding: 8px;">923.35</td>
<td style="border: 1px solid #ddd; padding: 8px;">2647.12</td>
<td style="border: 1px solid #ddd; padding: 8px;">1513.54</td>
<td style="border: 1px solid #ddd; padding: 8px;">45.60</td>
<td style="border: 1px solid #ddd; padding: 8px;">700.14</td>
<td style="border: 1px solid #ddd; padding: 8px;">1756.58</td>
<td style="border: 1px solid #ddd; padding: 8px;">1601.93</td>
<td style="border: 1px solid #ddd; padding: 8px;">68.69</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">S17</td>
<td style="border: 1px solid #ddd; padding: 8px;">994.88</td>
<td style="border: 1px solid #ddd; padding: 8px;">2740.54</td>
<td style="border: 1px solid #ddd; padding: 8px;">1739.85</td>
<td style="border: 1px solid #ddd; padding: 8px;">43.00</td>
<td style="border: 1px solid #ddd; padding: 8px;">754.75</td>
<td style="border: 1px solid #ddd; padding: 8px;">2583.76</td>
<td style="border: 1px solid #ddd; padding: 8px;">1848.82</td>
<td style="border: 1px solid #ddd; padding: 8px;">64.29</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">S18</td>
<td style="border: 1px solid #ddd; padding: 8px;">1073.86</td>
<td style="border: 1px solid #ddd; padding: 8px;">2400.12</td>
<td style="border: 1px solid #ddd; padding: 8px;">1939.22</td>
<td style="border: 1px solid #ddd; padding: 8px;">41.80</td>
<td style="border: 1px solid #ddd; padding: 8px;">856.46</td>
<td style="border: 1px solid #ddd; padding: 8px;">2751.61</td>
<td style="border: 1px solid #ddd; padding: 8px;">2098.72</td>
<td style="border: 1px solid #ddd; padding: 8px;">62.55</td>
</tr>
<tr>
<td rowspan="18" style="border: 1px solid #ddd; padding: 8px;">D2F-Dream-Instruct</td>
<td rowspan="12" style="border: 1px solid #ddd; padding: 8px;">LoPA-Dist-NV</td>
<td style="border: 1px solid #ddd; padding: 8px;">S1</td>
<td style="border: 1px solid #ddd; padding: 8px;">305.74</td>
<td style="border: 1px solid #ddd; padding: 8px;">959.00</td>
<td style="border: 1px solid #ddd; padding: 8px;">695.88</td>
<td style="border: 1px solid #ddd; padding: 8px;">52.80</td>
<td style="border: 1px solid #ddd; padding: 8px;">330.62</td>
<td style="border: 1px solid #ddd; padding: 8px;">758.34</td>
<td style="border: 1px solid #ddd; padding: 8px;">674.53</td>
<td style="border: 1px solid #ddd; padding: 8px;">78.17</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">S2</td>
<td style="border: 1px solid #ddd; padding: 8px;">373.23</td>
<td style="border: 1px solid #ddd; padding: 8px;">1302.99</td>
<td style="border: 1px solid #ddd; padding: 8px;">877.12</td>
<td style="border: 1px solid #ddd; padding: 8px;">51.40</td>
<td style="border: 1px solid #ddd; padding: 8px;">402.63</td>
<td style="border: 1px solid #ddd; padding: 8px;">961.29</td>
<td style="border: 1px solid #ddd; padding: 8px;">804.31</td>
<td style="border: 1px solid #ddd; padding: 8px;">74.22</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">S3</td>
<td style="border: 1px solid #ddd; padding: 8px;">451.62</td>
<td style="border: 1px solid #ddd; padding: 8px;">1419.09</td>
<td style="border: 1px solid #ddd; padding: 8px;">1143.30</td>
<td style="border: 1px solid #ddd; padding: 8px;">53.00</td>
<td style="border: 1px solid #ddd; padding: 8px;">444.73</td>
<td style="border: 1px solid #ddd; padding: 8px;">943.22</td>
<td style="border: 1px solid #ddd; padding: 8px;">870.85</td>
<td style="border: 1px solid #ddd; padding: 8px;">73.39</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">S4</td>
<td style="border: 1px solid #ddd; padding: 8px;">503.71</td>
<td style="border: 1px solid #ddd; padding: 8px;">1779.60</td>
<td style="border: 1px solid #ddd; padding: 8px;">1226.72</td>
<td style="border: 1px solid #ddd; padding: 8px;">46.60</td>
<td style="border: 1px solid #ddd; padding: 8px;">495.93</td>
<td style="border: 1px solid #ddd; padding: 8px;">1131.64</td>
<td style="border: 1px solid #ddd; padding: 8px;">941.23</td>
<td style="border: 1px solid #ddd; padding: 8px;">72.48</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">S5</td>
<td style="border: 1px solid #ddd; padding: 8px;">568.65</td>
<td style="border: 1px solid #ddd; padding: 8px;">1660.89</td>
<td style="border: 1px solid #ddd; padding: 8px;">1317.38</td>
<td style="border: 1px solid #ddd; padding: 8px;">42.00</td>
<td style="border: 1px solid #ddd; padding: 8px;">540.76</td>
<td style="border: 1px solid #ddd; padding: 8px;">1185.14</td>
<td style="border: 1px solid #ddd; padding: 8px;">1033.60</td>
<td style="border: 1px solid #ddd; padding: 8px;">68.99</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">S6</td>
<td style="border: 1px solid #ddd; padding: 8px;">615.95</td>
<td style="border: 1px solid #ddd; padding: 8px;">1951.86</td>
<td style="border: 1px solid #ddd; padding: 8px;">1542.82</td>
<td style="border: 1px solid #ddd; padding: 8px;">37.60</td>
<td style="border: 1px solid #ddd; padding: 8px;">568.75</td>
<td style="border: 1px solid #ddd; padding: 8px;">1352.22</td>
<td style="border: 1px solid #ddd; padding: 8px;">1139.06</td>
<td style="border: 1px solid #ddd; padding: 8px;">65.88</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">S7</td>
<td style="border: 1px solid #ddd; padding: 8px;">325.15</td>
<td style="border: 1px solid #ddd; padding: 8px;">697.49</td>
<td style="border: 1px solid #ddd; padding: 8px;">620.42</td>
<td style="border: 1px solid #ddd; padding: 8px;">50.80</td>
<td style="border: 1px solid #ddd; padding: 8px;">379.42</td>
<td style="border: 1px solid #ddd; padding: 8px;">839.65</td>
<td style="border: 1px solid #ddd; padding: 8px;">710.10</td>
<td style="border: 1px solid #ddd; padding: 8px;">75.28</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">S8</td>
<td style="border: 1px solid #ddd; padding: 8px;">408.37</td>
<td style="border: 1px solid #ddd; padding: 8px;">1182.69</td>
<td style="border: 1px solid #ddd; padding: 8px;">866.90</td>
<td style="border: 1px solid #ddd; padding: 8px;">51.00</td>
<td style="border: 1px solid #ddd; padding: 8px;">449.56</td>
<td style="border: 1px solid #ddd; padding: 8px;">934.55</td>
<td style="border: 1px solid #ddd; padding: 8px;">838.35</td>
<td style="border: 1px solid #ddd; padding: 8px;">75.13</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">S9</td>
<td style="border: 1px solid #ddd; padding: 8px;">465.55</td>
<td style="border: 1px solid #ddd; padding: 8px;">1097.40</td>
<td style="border: 1px solid #ddd; padding: 8px;">1016.91</td>
<td style="border: 1px solid #ddd; padding: 8px;">50.60</td>
<td style="border: 1px solid #ddd; padding: 8px;">497.47</td>
<td style="border: 1px solid #ddd; padding: 8px;">1172.31</td>
<td style="border: 1px solid #ddd; padding: 8px;">946.98</td>
<td style="border: 1px solid #ddd; padding: 8px;">74.75</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">S10</td>
<td style="border: 1px solid #ddd; padding: 8px;">544.72</td>
<td style="border: 1px solid #ddd; padding: 8px;">1542.99</td>
<td style="border: 1px solid #ddd; padding: 8px;">1145.55</td>
<td style="border: 1px solid #ddd; padding: 8px;">46.80</td>
<td style="border: 1px solid #ddd; padding: 8px;">539.28</td>
<td style="border: 1px solid #ddd; padding: 8px;">1147.95</td>
<td style="border: 1px solid #ddd; padding: 8px;">1021.96</td>
<td style="border: 1px solid #ddd; padding: 8px;">71.34</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">S11</td>
<td style="border: 1px solid #ddd; padding: 8px;">591.57</td>
<td style="border: 1px solid #ddd; padding: 8px;">1578.00</td>
<td style="border: 1px solid #ddd; padding: 8px;">1204.05</td>
<td style="border: 1px solid #ddd; padding: 8px;">42.20</td>
<td style="border: 1px solid #ddd; padding: 8px;">580.04</td>
<td style="border: 1px solid #ddd; padding: 8px;">1292.18</td>
<td style="border: 1px solid #ddd; padding: 8px;">1132.19</td>
<td style="border: 1px solid #ddd; padding: 8px;">66.94</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">S12</td>
<td style="border: 1px solid #ddd; padding: 8px;">636.55</td>
<td style="border: 1px solid #ddd; padding: 8px;">1811.71</td>
<td style="border: 1px solid #ddd; padding: 8px;">1500.59</td>
<td style="border: 1px solid #ddd; padding: 8px;">36.00</td>
<td style="border: 1px solid #ddd; padding: 8px;">609.90</td>
<td style="border: 1px solid #ddd; padding: 8px;">1407.56</td>
<td style="border: 1px solid #ddd; padding: 8px;">1159.28</td>
<td style="border: 1px solid #ddd; padding: 8px;">65.50</td>
</tr>
<tr>
<td rowspan="6" style="border: 1px solid #ddd; padding: 8px;">LoPA-Dist-Ascend</td>
<td style="border: 1px solid #ddd; padding: 8px;">S13</td>
<td style="border: 1px solid #ddd; padding: 8px;">412.90</td>
<td style="border: 1px solid #ddd; padding: 8px;">911.73</td>
<td style="border: 1px solid #ddd; padding: 8px;">911.73</td>
<td style="border: 1px solid #ddd; padding: 8px;">50.80</td>
<td style="border: 1px solid #ddd; padding: 8px;">515.01</td>
<td style="border: 1px solid #ddd; padding: 8px;">1235.84</td>
<td style="border: 1px solid #ddd; padding: 8px;">1090.45</td>
<td style="border: 1px solid #ddd; padding: 8px;">76.12</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">S14</td>
<td style="border: 1px solid #ddd; padding: 8px;">525.66</td>
<td style="border: 1px solid #ddd; padding: 8px;">1546.34</td>
<td style="border: 1px solid #ddd; padding: 8px;">1143.37</td>
<td style="border: 1px solid #ddd; padding: 8px;">48.40</td>
<td style="border: 1px solid #ddd; padding: 8px;">619.58</td>
<td style="border: 1px solid #ddd; padding: 8px;">1424.32</td>
<td style="border: 1px solid #ddd; padding: 8px;">1310.35</td>
<td style="border: 1px solid #ddd; padding: 8px;">75.36</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">S15</td>
<td style="border: 1px solid #ddd; padding: 8px;">625.53</td>
<td style="border: 1px solid #ddd; padding: 8px;">1729.78</td>
<td style="border: 1px solid #ddd; padding: 8px;">1435.06</td>
<td style="border: 1px solid #ddd; padding: 8px;">46.20</td>
<td style="border: 1px solid #ddd; padding: 8px;">689.89</td>
<td style="border: 1px solid #ddd; padding: 8px;">1644.74</td>
<td style="border: 1px solid #ddd; padding: 8px;">1356.36</td>
<td style="border: 1px solid #ddd; padding: 8px;">72.63</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">S16</td>
<td style="border: 1px solid #ddd; padding: 8px;">716.19</td>
<td style="border: 1px solid #ddd; padding: 8px;">1780.41</td>
<td style="border: 1px solid #ddd; padding: 8px;">1558.00</td>
<td style="border: 1px solid #ddd; padding: 8px;">43.80</td>
<td style="border: 1px solid #ddd; padding: 8px;">770.78</td>
<td style="border: 1px solid #ddd; padding: 8px;">1589.69</td>
<td style="border: 1px solid #ddd; padding: 8px;">1480.56</td>
<td style="border: 1px solid #ddd; padding: 8px;">71.49</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">S17</td>
<td style="border: 1px solid #ddd; padding: 8px;">796.65</td>
<td style="border: 1px solid #ddd; padding: 8px;">1798.14</td>
<td style="border: 1px solid #ddd; padding: 8px;">1687.69</td>
<td style="border: 1px solid #ddd; padding: 8px;">39.80</td>
<td style="border: 1px solid #ddd; padding: 8px;">837.21</td>
<td style="border: 1px solid #ddd; padding: 8px;">1782.80</td>
<td style="border: 1px solid #ddd; padding: 8px;">1517.90</td>
<td style="border: 1px solid #ddd; padding: 8px;">67.78</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;">S18</td>
<td style="border: 1px solid #ddd; padding: 8px;">896.21</td>
<td style="border: 1px solid #ddd; padding: 8px;">2586.73</td>
<td style="border: 1px solid #ddd; padding: 8px;">2086.04</td>
<td style="border: 1px solid #ddd; padding: 8px;">36.40</td>
<td style="border: 1px solid #ddd; padding: 8px;">897.10</td>
<td style="border: 1px solid #ddd; padding: 8px;">1868.16</td>
<td style="border: 1px solid #ddd; padding: 8px;">1642.72</td>
<td style="border: 1px solid #ddd; padding: 8px;">66.87</td>
</tr>
</tbody>
</table>
</div>

The results illustrate the trade-off between inference throughput and generation quality across varying branch configurations and system backends.

## 🚀 Usage Guide

### 1. Installation

First, clone the repository and install the dependencies.

#### UV Setup (Recommended)

```shell
# Clone the repository
git clone https://github.com/zhijie-group/LoPA.git
cd LoPA

# Init the project
uv sync
source .venv/bin/activate
```

#### Conda Setup

```shell
# Create environment
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

#### LoPA-Dist-NV Experiments

To evaluate the performance of LoPA-Dist-NV, navigate to the `lopa_dist_nv` directory:

```shell
cd lopa_dist_nv
```

**Run LoPA-Dist-NV Experiments:**

```shell
bash launch_all.sh
```

## ©️ Citation

If you find LoPA useful for your research, please cite our paper:

```bibtex
@misc{xu2025lopascalingdllminference,
      title={LoPA: Scaling dLLM Inference via Lookahead Parallel Decoding}, 
      author={Chenkai Xu and Yijie Jin and Jiajun Li and Yi Tu and Guoping Long and Dandan Tu and Tianqi Hou and Junchi Yan and Zhijie Deng},
      year={2025},
      eprint={2512.16229},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.16229}, 
}

```

