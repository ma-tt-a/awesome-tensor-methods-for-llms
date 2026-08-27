# Tensor Methods for Language Models: From Token Representation to Training, Adaptation, Compression, Inference, and Interpretability

<!-- [![arXiv](https://img.shields.io/badge/arXiv-tbd-b31b1b.svg)](tbd) -->

A curated list of tensor methods for large language models — tensor decompositions and tensor networks applied inside the Transformer (embedding layer, attention, feed-forward networks) and across the model lifecycle, from tokenization through embeddings, pre-training, adaptation, compression, and inference to interpretability.

This repository accompanies our survey and collects the papers, software, and background literature behind it.

## Contents

---

- [Survey](#survey)
- [Research Papers](#research-papers)
- [Software](#software)
- [Related Literature](#related-literature)
- [Citation](#citation)

## Survey

---

Tensor decompositions are usually presented as isolated compression mechanisms. Our survey treats tensorization as a common structural principle acting on token representations, weights, adaptation updates, caches, and activations, and organizes the literature through two complementary views:

- a **component view** — *what* inside a Transformer is tensorized: the embedding layer, the attention mechanism, and the feed-forward network, each with its own structural constraints and compatible decompositions
- a **lifecycle view** — *when and why* tensorization is introduced: tokenization, embeddings, pre-training, adaptation, compression, inference, and interpretability

Together, the two views separate the structural compatibility of a decomposition with a model object from the training or deployment objective for which it is used.

The survey also provides:

- unified notation and theoretical foundations for tensor operations, decompositions, and networks in Transformer-like language models, with tensor network diagrams throughout
- comparison tables that explicitly record differences in model scale, baselines, evaluation protocols, and reported metrics
- connections to neighboring efficiency techniques and to probabilistic tensor networks
- open challenges, in particular the *compression-realization gap*, formulated through the metric $\rho_{\rm gap}$, which separates algorithmic overhead from hardware realization

## Research Papers

---

Papers are grouped by lifecycle stage, following the lifecycle view of the survey.

### Embeddings

| Title                                                                                                                             | Decomposition | Description                                                              | Venue | Year |
| --------------------------------------------------------------------------------------------------------------------------------- | ------------- | ------------------------------------------------------------------------ | ----- | ---- |
| [Tensorizing Engram: Sharing Latents Across N-Gram Embeddings is Beneficial in LLMs](https://arxiv.org/abs/2606.08347)            | CP            | N-gram table in CP form: size linear in n instead of exponential.        | Arxiv | 2026 |
| [TensorGPT: Efficient Compression of Large Language Models based on Tensor-Train Decomposition](https://arxiv.org/abs/2307.00526) | TT            | Row-wise TT-SVD of a pretrained embedding table, no retraining required. | Arxiv | 2024 |
| [MorphTE: Injecting Morphology in Tensorized Embeddings](https://arxiv.org/abs/2210.15379)                                        | -             | Word vectors built as Kronecker products of morpheme embeddings.         | NIPS  | 2022 |
| [Tensorized Embedding Layers](https://aclanthology.org/2020.findings-emnlp.436.pdf)                                               | TTM           | Embedding matrix replaced by a TTM representation, trained from scratch. | EMNLP | 2020 |

### Pre-training

| Title                                                                                                                                                                               | Decomposition                  | Description                                                                        | Venue  | Year |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ | ---------------------------------------------------------------------------------- | ------ | ---- |
| [CoMERA: Computing- and Memory-Efficient Training via Rank-Adaptive Tensor Optimization](https://arxiv.org/abs/2405.14377)                                                          | Modified (Adaptive TT-rank) TT | Diagonal factors interleaved with TT cores make TT-rank adaptive.                  | NIPS   | 2024 |
| [Efficient GPT Model Pre-training using Tensor Train Matrix Representation](https://aclanthology.org/2023.paclic-1.60/)                                                             | TTM                            | Every projection matrix of the dense layers replaced by a TTM parameterization.    | PACLIC | 2023 |
| [Hypoformer: Hybrid Decomposition Transformer for Edge-friendly Neural Machine Translation](https://aclanthology.org/2022.emnlp-main.475/)                                          | Hybrid TTM                     | Parallel dense and low-rank TTM branches for one projection.                       | EMNLP  | 2022 |
| [Shapeshifter: a Parameter-efficient Transformer using Factorized Reshaped Matrices](https://proceedings.neurips.cc/paper/2021/hash/09def3ebbc44ff3426b28fcd88c83554-Abstract.html) | Kronecker                      | All weight matrices as a sum of Kronecker products.                                | NIPS   | 2021 |
| [A Tensorized Transformer for Language Modeling](https://arxiv.org/abs/1906.09777)                                                                                                  | BT                             | Attention in BT representation form: Q, K, V as shared factors, one core per head. | NIPS   | 2019 |

### Adaptation

| Title                                                                                                                                                            | Decomposition   | Description                                                                                                     | Venue | Year |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- | --------------------------------------------------------------------------------------------------------------- | ----- | ---- |
| [TeRA: Vector-based Random Tensor Network for High-Rank Adaptation of Large Language Models](https://aclanthology.org/2026.acl-long.106/)                        | Tucker          | Frozen core and factors shared across updates; only diagonal factors are trained.                               | ACL   | 2026 |
| [Quantum-PEFT: Ultra parameter-efficient fine-tuning](https://arxiv.org/abs/2503.05431)                                                                          | SVD-like        | Unitary factors expressed via generalized Pauli rotations; parameter count logarithmic in the matrix dimension. | ICLR  | 2025 |
| [MetaTT: A Global Tensor-Train Adapter for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2506.09105)                                                    | TT              | One global TT-parameterized update tensor over layers and projection types.                                     | TMLR  | 2025 |
| [DoTA: Weight-Decomposed Tensor Adaptation for Large Language Models](https://link.springer.com/chapter/10.1007/978-981-96-8186-0_2)                             | MPO             | MPO update initialized from the MPO decomposition of the pretrained weight.                                     | PAKDD | 2025 |
| [QuanTA: Efficient High-Rank Fine-Tuning of LLMs with Quantum-Informed Tensor Adaptation](https://arxiv.org/abs/2406.00132)                                      | Quantum circuit | Update as a network of two-axis gates, not restricted to low rank.                                              | NIPS  | 2024 |
| [Tensor Train Low-rank Approximation (TT-LoRA): Democratizing AI with Accelerated LLMs](https://ieeexplore.ieee.org/abstract/document/10903446)                  | TT              | Update matrix parameterized directly in TT form.                                                                | ICMLA | 2024 |
| [LoRETTA: Low-Rank Economic Tensor-Train Adaptation for Ultra-Low-Parameter Fine-Tuning of Large Language Models](https://aclanthology.org/2024.naacl-long.174/) | TT              | Two variants: a TT-parameterized adapter, or TT-parameterized LoRA factors.                                     | NAACL | 2024 |
| [LoRTA: Low Rank Tensor Adaptation of Large Language Models](https://arxiv.org/abs/2410.04060)                                                                   | CP              | All updates aggregated into one tensor represented in CP form.                                                  | Arxiv | 2024 |
| [LoTR: Low Tensor Rank Weight Adaptation](https://arxiv.org/abs/2402.01376)                                                                                      | Tucker          | Updates stacked into a third-order tensor; Tucker-2 factors shared across projections.                          | Arxiv | 2024 |
| [KronA: Parameter Efficient Tuning with Kronecker Adapter](https://arxiv.org/abs/2212.10650)                                                                     | Kronecker       | Kronecker rank-1 factorization instead of LoRA's low-rank factorization; the update can be full rank.           | Arxiv | 2022 |

### Compression

| Title                                                                                                                                                | Decomposition               | Description                                                                                                                                         | Venue | Year |
| ---------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ----- | ---- |
| [Rethinking the Role of Tensor Decompositions in Post-Training LLM Compression](https://arxiv.org/abs/2606.03465)                                    | -                           | Evidence that  a Frobenius-optimal factorization need not be good in the operator sense; lightweight repair recovers only part of the lost quality. | Arxiv | 2026 |
| [LeSTD: LLM Compression via Learning-based Sparse Tensor Decomposition](https://openreview.net/forum?id=0oHaazjMUX)                                  | Tucker with sparse core     | TensorLLM with a sparsified core and reconstruction-free inference.                                                                                 | ICLR  | 2026 |
| [Saten: Sparse Augmented Tensor Networks for Post-Training Compression of Large Language Models](https://aclanthology.org/2025.findings-emnlp.1287/) | TT with sparsification      | Error-based TT-SVD plus a multiplicative sparse correction of the residual.                                                                         | EMNLP | 2025 |
| [TensorLLM: Tensorising Multi-Head Attention for Enhanced Reasoning and Compression in LLMs](https://ieeexplore.ieee.org/document/11228585)          | Tucker                      | Tucker on stacked projections with the head mode left uncompressed.                                                                                 | IJCNN | 2025 |
| [TRAWL: Tensor Reduced and Approximated Weights for Large Language Models](https://link.springer.com/chapter/10.1007/978-981-96-8298-0_32)           | Tucker                      | First to decompose a tensor aggregated from projections; also targets noise reduction.                                                              | PAKDD | 2025 |
| [LatentLLM: Attention-Aware Joint Tensor Compression](https://arxiv.org/abs/2505.18413)                                                              | Tucker                      | Activation-aware joint factorization of projection groups on a calibration set.                                                                     | Arxiv | 2025 |
| [TQCompressor: Improving Tensor Decomposition Methods in Neural Networks Via Permutations](https://ieeexplore.ieee.org/document/10707874)            | Kronecker with permutations | Row and column permutations added to the Kronecker parameterization.                                                                                | MIPR  | 2024 |
| [CompactifAI: Extreme Compression of Large Language Models using Quantum-Inspired Tensor Networks](https://arxiv.org/abs/2401.14109)                 | MPO                         | MPO decomposition of pretrained weights with short healing; combined with quantization.                                                             | Arxiv | 2024 |
| [Kronecker Decomposition for GPT Compression](https://aclanthology.org/2022.acl-short.24/)                                                           | Kronecker                   | Kronecker rank-1 compression of GPT-2 with a distillation healing stage.                                                                            | ACL   | 2022 |

### Inference

| Title                                                                                                                                      | Decomposition         | Description                                                                      | Venue                   | Year |
| ------------------------------------------------------------------------------------------------------------------------------------------ | --------------------- | -------------------------------------------------------------------------------- | ----------------------- | ---- |
| [Tucker Attention: A generalization of approximate attention mechanisms](https://openreview.net/forum?id=ErcPPRZaiq)                       | Tucker                | Tucker-parameterized attention tensors exploiting cross-head redundancy.         | ICML                    | 2026 |
| [EinSort: Sorting is All We Need for Tensorizing LLM](https://openreview.net/forum?id=yoIh7UwdAC)                                          | Arbitrary             | Invertible permutation before compression — sorted tensors approximate better.   | ICML workshop (CoLoRAI) | 2026 |
| [Tensor Product Attention Is All You Need](https://arxiv.org/abs/2501.06425)                                                               | CP                    | Per-token queries, keys and values as context-dependent sums of tensor products. | NIPS (spotlight)        | 2025 |
| [Unlocking Data-free Low-bit Quantization with Matrix Decomposition for KV Cache Compression](https://aclanthology.org/2024.acl-long.133/) | MPO with quantization | Mixed-precision quantization exploiting the distributions of the two MPO cores.  | ACL                     | 2024 |

### Interpretability

| Title                                                                                                                                                                                                                                        | Description                                                                                                   | Venue            | Year |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ---------------- | ---- |
| [PolySAE: Modeling Feature Interactions in Sparse Autoencoders via Polynomial Decoding](https://openreview.net/forum?id=XAhDgsYn3a&referrer=%5Bthe%20profile%20of%20Panagiotis%20Koromilas%5D%28%2Fprofile%3Fid%3D~Panagiotis_Koromilas1%29) | Quadratic and cubic decoder terms let SAEs separate compound concepts from their parts.                       | ICML             | 2026 |
| [TensorLens: End-to-End Transformer Analysis via High-Order Attention Tensors](https://aclanthology.org/2026.acl-long.156/)                                                                                                                  | The whole Transformer stack as one fourth-order generalized attention tensor.                                 | ACL              | 2026 |
| [Bilinear MLPs enable weight-based mechanistic interpretability](https://openreview.net/forum?id=gI0kPklUKS)                                                                                                                                 | Removing the GLU nonlinearity makes the layer an exact third-order tensor.                                    | ICLR (spotlight) | 2025 |
| [An introduction to graphical tensor notation for mechanistic interpretability](https://arxiv.org/abs/2402.01790)                                                                                                                            | Tutorial rewriting the transformer-circuits framework in graphical tensor notation, up to the induction head. | Arxiv            | 2024 |

## Software

---

### Python tensor libraries

| Library                                                    | Decompositions                 | Description                                                                                    | NN layers | Backend                                         |
| ---------------------------------------------------------- | ------------------------------ | ---------------------------------------------------------------------------------------------- | --------- | ----------------------------------------------- |
| [TensorLy](https://github.com/tensorly/tensorly)           | CP, Tucker, TT                 | General-purpose tensor toolbox: tensor algebra, tensor decompositions, tensor regression       | –         | NumPy, PyTorch, JAX, TensorFlow, CuPy or Paddle |
| [TensorLy-Torch](https://github.com/tensorly/torch)        | CP, Tucker, TT, BlockTT (TTM)  | Ready-made tensorized and factorized layers built on TensorLy                                  | +         | PyTorch                                         |
| [tntorch](https://github.com/rballester/tntorch)           | CP, Tucker, TT, Hybrid formats | Unified interface over TN formats with cross-approximation, rank truncation and fancy indexing | -         | PyTorch                                         |
| [T3F](https://github.com/Bihaqo/t3f/tree/develop)          | TT family                      | TT computations with batching and Riemannian optimization; TT layers for compressing networks  | +         | TensorFlow                                      |
| [TensorKrowch](https://github.com/joserapa98/tensorkrowch) | Arbitrary TNs                  | Construct, train and embed arbitrary tensor networks as layers inside deep models              | +         | PyTorch                                         |
| [tn4ml](https://github.com/bsc-quantic/tn4ml)              | MPS, MPO, spaced-MPO           | TN training pipelines for ML: data embedding, objectives, DMRG-like optimization               | +         | Build on quimb and JAX                          |
| [quimb](https://github.com/jcmgray/quimb)                  | Arbitrary TNs                  | Many-body library with a strong TN module: arbitrary geometry, MPS/PEPS/MERA, DMRG/TEBD        | –         | Arbitrary via autoray                           |
| [cotengra](https://github.com/jcmgray/cotengra)            | –                              | Contraction path optimization for large tensor networks and einsum expressions                 | –         | Arbitrary via autoray                           |

### Hardware co-design

| Design                                                                                                                                      | Description                                           | Venue | Year |
| ------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- | ----- | ---- |
| [ETTE](https://dl.acm.org/doi/10.1145/3579371.3589103)                                                                                      | TT-format DNN inference, algorithm/hardware co-design | ISCA  | 2023 |
| [FETTA](https://ieeexplore.ieee.org/document/11333282)                                                                                      | Tensorized NN training, flexible contraction order    | IEEE  | 2025 |
| [A Tensor-Train Decomposition based Compression of LLMs on Group Vector Systolic Accelerator](https://arxiv.org/abs/2501.19135)             | TT-compressed LLM inference                           | Arxiv | 2025 |
| [Ultra Memory-Efficient On-FPGA Training of Transformers via Tensor-Compressed Optimization](https://ieeexplore.ieee.org/document/11121368) | Memory-efficient tensorized Transformer training      | IEEE  | 2025 |

### Others

| Library                                                                | Language / Backend |
| ---------------------------------------------------------------------- | ------------------ |
| [ITensor](https://github.com/ITensor/ITensors.jl)                      | Julia, C++         |
| [Tensor Toolbox for MATLAB](https://gitlab.com/tensors/tensor_toolbox) | MATLAB             |
| [TenDeC++](https://github.com/osmint/TenDeC)                           | C++                |
| [Scikit-TT](https://github.com/PGelss/scikit_tt)                       | Python, NumPy      |
| [TTAX](https://github.com/fasghq/ttax)                                 | Python, JAX        |
| [TorchMPS](https://github.com/jemisjoky/TorchMPS)                      | Python, PyTorch    |
| [TedNet](https://github.com/tnbar/tednet)                              | Python, PyTorch    |

### Case Studies

- [Compressing Transformer Language Models via Matrix Product Operator Decomposition: A Case Study on PicoGPT](https://arxiv.org/abs/2603.28534)
- [A Practical Tensor-Network Compression Pipeline for Production-Scale Large Language Models](https://arxiv.org/abs/2602.01613v1)

## Related Literature

---

### Classical tensor decomposition and tensor networks

- [Tensor Decompositions and Applications](https://epubs.siam.org/doi/abs/10.1137/07070111X?journalCode=siread)
- [Tensor Networks for Dimensionality Reduction and Large-Scale Optimization Part 1 Low-Rank Tensor Decompositions](https://www.emerald.com/ftmal/article-abstract/9/4-5/249/1332391/Tensor-Networks-for-Dimensionality-Reduction-and?redirectedFrom=fulltext)
- [Tensor Networks for Dimensionality Reduction and Large-Scale Optimizations Part 2 Applications and Future Perspectives](https://www.emerald.com/ftmal/article-abstract/9/6/431/1332832/Tensor-Networks-for-Dimensionality-Reduction-and?redirectedFrom=fulltext)
- [A practical introduction to tensor networks: Matrix product states and projected entangled pair states](https://www.sciencedirect.com/science/article/abs/pii/S0003491614001596)

### Tensor methods for ML and NNs

- [Tensor Networks Meet Neural Networks: A Survey and Future Perspectives](https://arxiv.org/abs/2302.09019)
- [Tensor Decomposition for Signal Processing and Machine Learning](https://ieeexplore.ieee.org/document/7891546)
- [A Survey on Tensor Techniques and Applications in Machine Learning](https://ieeexplore.ieee.org/document/8884203)
- [Tensor Methods in Computer Vision and Deep Learning](https://ieeexplore.ieee.org/document/9420085)
- [Low Rank Optimization for Efficient Deep Learning: Making A Balance between Compact Architecture and Fast Training](https://ieeexplore.ieee.org/document/10355073)
- [A survey of latent factorization of tensor-based model compression: Algorithms, toolboxes and future directions](https://www.sciencedirect.com/science/article/pii/S0925231226008520)

### LLM efficiency techniques

- [Efficient Large Language Models: A Survey](https://arxiv.org/abs/2312.03863)
- [Parameter-Efficient Fine-Tuning in Large Models: A Survey of Methodologies](https://arxiv.org/abs/2410.19878)
- [A Survey on Model Compression for Large Language Models](https://arxiv.org/abs/2308.07633)
- [A Survey on Transformer Compression](https://arxiv.org/abs/2402.05964)
- [Efficient Compressing and Tuning Methods for Large Language Models: A Systematic Literature Review](https://dl.acm.org/doi/10.1145/3728636)
- [Model Compression and Efficient Inference for Large Language Models: A Survey](https://arxiv.org/abs/2402.09748)
- [A Survey on Efficient Inference for Large Language Models](https://arxiv.org/abs/2404.14294)
- [LLM Inference Unveiled: Survey and Roofline Model Insights](https://arxiv.org/abs/2402.16363)
- [A Survey on Large Language Model Acceleration based on KV Cache Management](https://arxiv.org/abs/2412.19442)
- [Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems](https://dl.acm.org/doi/10.1145/3754448)

### LLM mechanistic interpretability

- [Bridging the Black Box: A Survey on Mechanistic Interpretability in AI](https://dl.acm.org/doi/10.1145/3787104)
- [A Practical Review of Mechanistic Interpretability for Transformer-Based Language Models](https://arxiv.org/abs/2407.02646)
- [Mechanistic Interpretability for AI Safety -- A Review](https://arxiv.org/abs/2404.14082)

### Related GitHub pages

- [Awesome Tensorial Neural Networks](https://github.com/tnbar/awesome-tensorial-neural-networks)
- [Awesome Tensor Decomposition](https://github.com/vantienpham/Awesome-Tensor-Decomposition)
- [Awesome LLM Pre-training](https://github.com/RUCAIBox/awesome-llm-pretraining)
- [Awesome LLM Compression](https://github.com/HuangOwen/Awesome-LLM-Compression)
- [Awesome LLM Inference](https://github.com/xlite-dev/Awesome-LLM-Inference)
- [Awesome LMMs Mechanistic Interpretability](https://github.com/itsqyh/Awesome-LMMs-Mechanistic-Interpretability)

## Citation

---

```BibTeX
@misc{tarasov2026tensorsforllms,
author = {Tarasov, Matvei and Ahmadi-Asl, Salman and de Almeida, Andr\'e L. F. and Cichocki, Andrzej},
title = {Tensor Methods for Language Models: From Token Representation to Training, Adaptation, Compression, Inference, and Interpretability},
eprint={tbd},
archivePrefix={arXiv},
year = {2026},
url = {tbd}
}
```
