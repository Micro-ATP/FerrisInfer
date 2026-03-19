# Attention Residuals
## TECHNICAL REPORT OF ATTENTION RESIDUALS
### Kimi Team
### https://github.com/MoonshotAI/Attention-Residuals

## ABSTRACT
Residual connections [12] with PreNorm [60] are standard in modern LLMs, yet they accumulate all layer outputs with fixed unit weights. This uniform aggregation causes uncontrolled hidden-state growth with depth, progressively diluting each layer’s contribution [27]. We propose Attention Residuals (AttnRes), which replaces this fixed accumulation with softmax attention over preceding layer outputs, allowing each layer to selectively aggregate earlier representations with learned, input-dependent weights. To address the memory and communication overhead of attending over all preceding layer outputs for large-scale model training, we introduce Block AttnRes, which partitions layers into blocks and attends over block-level representations, reducing the memory footprint while preserving most of the gains of full AttnRes. Combined with cache-based pipeline communication and a two-phase computation strategy, Block AttnRes becomes a practical drop-in replacement for standard residual connections with minimal overhead.

Scaling law experiments confirm that the improvement is consistent across model sizes, and ablations validate the benefit of content-dependent depth-wise selection. We further integrate AttnRes into the Kimi Linear architecture [69] (48B total / 3B activated parameters) and pre-train on 1.4T tokens, where AttnRes mitigates PreNorm dilution, yielding more uniform output magnitudes and gradient distribution across depth, and improves downstream performance across all evaluated tasks.

*Figure 1: Overview of Attention Residuals. (a) Standard Residuals: standard residual connections with uniform additive accumulation. (b) Full AttnRes: each layer selectively aggregates all previous layer outputs via learned attention weights. (c) Block AttnRes: layers are grouped into blocks, reducing memory from $O(L d)$ to $O(N d)$.*

## 1 Introduction
Standard residual connections [12] are the de facto building block of modern LLMs [35, 51, 9]. The update $h_{l}=h_{l-1}+f_{l-1}(h_{l-1})$ is widely understood as a gradient highway that lets gradients bypass transformations via identity mappings, enabling stable training at depth. Yet residuals also play a second role that has received less attention. Unrolling the recurrence shows that every layer receives the same uniformly-weighted sum of all prior layer outputs; residuals define how information aggregates across depth. Unlike sequence mixing and expert routing, which now employ learnable input-dependent weighting [53, 20, 9], this depth-wise aggregation remains governed by fixed unit weights, with no mechanism to selectively emphasize or suppress individual layer contributions.

In practice, PreNorm [60] has become the dominant paradigm, yet its unweighted accumulation causes hidden-state magnitudes to grow as $O(L)$ with depth, progressively diluting each layer’s relative contribution [27]. Early-layer information is buried and cannot be selectively retrieved; empirically, a significant fraction of layers can be pruned with minimal loss [11]. Recent efforts such as scaled residual paths [54] and multi-stream recurrences [72] remain bound to the additive recurrence, while methods that do introduce cross-layer access [36, 56] are difficult to scale. The situation parallels the challenges that recurrent neural networks (RNNs) faced over the sequence dimension before attention mechanism provided an alternative.

We observe a formal duality between depth-wise accumulation and the sequential recurrence in RNNs. Building on this duality, we propose Attention Residuals (AttnRes), which replaces the fixed accumulation $h_{l}=\sum_{i} v_{i}$ with $h_{l}=\sum_{i} \alpha_{i \to l} \cdot v_{i}$ , where $\alpha_{i \to l}$ are softmax attention weights computed from a single learned pseudo-query $w_{l} \in \mathbb{R}^{d}$ per layer. This lightweight mechanism enables selective, content-aware retrieval across depth with only one d -dimensional vector per layer. Indeed, standard residual connections and prior recurrence-based variants can all be shown to perform depth-wise linear attention; AttnRes generalizes them to depth-wise softmax attention, completing for depth the same linear-to-softmax transition that proved transformative over sequences (§6.2, §6.1).

In standard training, Full AttnRes adds negligible overhead, since the layer outputs it requires are already retained for backpropagation. At scale, however, activation recomputation and pipeline parallelism are routinely employed, and these activations must now be explicitly preserved and communicated across pipeline stages. We introduce Block AttnRes to maintain efficiency in this regime: layers are partitioned into N blocks, each reduced to a single representation via standard residuals, with cross-block attention applied only over the N block-level summaries. This brings both memory and communication down to $O(N d)$ , and together with infrastructure optimizations (§4), Block AttnRes serves as a drop-in replacement for standard residual connections with marginal training cost and negligible inference latency overhead.

Scaling law experiments confirm that AttnRes consistently outperforms the baseline across compute budgets, with Block AttnRes matching the loss of a baseline trained with 1.25× more compute. We further integrate AttnRes into the Kimi Linear architecture [69] (48B total / 3B activated parameters) and pre-train on 1.4T tokens. Analysis of the resulting training dynamics reveals that AttnRes mitigates PreNorm dilution, with output magnitudes remaining bounded across depth and gradient norms distributing more uniformly across layers. On downstream benchmarks, our final model improves over the baseline across all evaluated tasks.

### Contributions
1. **Attention Residuals**: We propose AttnRes, which replaces fixed residual accumulation with learned softmax attention over depth, and its scalable variant Block AttnRes that reduces memory and communication from $O(L d)$ to $O(N d)$ . Through a unified structured-matrix analysis, we show that standard residuals and prior recurrence-based variants correspond to depth-wise linear attention, while AttnRes performs depth-wise softmax attention.
2. **Infrastructure for scale**: We develop system optimizations that make Block AttnRes practical and efficient at scale, including cross-stage caching that eliminates redundant transfers under pipeline parallelism and a two-phase inference strategy that amortizes cross-block attention via online softmax [31]. The resulting training overhead is marginal, and the inference latency overhead is less than 2% on typical inference workloads.
3. **Comprehensive evaluation and analysis**: We validate AttnRes through scaling law experiments, component ablations, and downstream benchmarks on a 48B-parameter model pre-trained on 1.4T tokens, demonstrating consistent improvements over standard residual connections. Training dynamics analysis further reveals that AttnRes mitigates PreNorm dilution, yielding bounded hidden-state magnitudes and more uniform gradient distribution across depth.

## 2 Motivation
### Notation
Consider a batch of input sequences with shape $B ×T ×d$ , where B is the batch size, T is the sequence length, and d is the hidden dimension. For clarity, we write formulas for a single token: $h_{l} \in \mathbb{R}^{d}$ denotes the hidden state entering layer l, where $l \in\{1, ..., L\}$ is the layer index and L is the total number of layers. The token embedding is $h_{1}$ . The function $f_{l}$ represents the transformation applied by layer l. In Transformer models, we treat each self-attention or MLP as an individual layer.

### 2.1 Training Deep Networks via Residuals
#### Residual Learning
Residual learning [12] proves to be a critical technique in training deep networks as it allows gradients to bypass transformations. Specifically, each layer updates the hidden state as:
$$h_{l}=h_{l-1}+f_{l-1}\left(h_{l-1}\right)$$

Expanding this recurrence, the hidden state at layer l is the sum of the embedding and all preceding layer outputs: $h_{l}=h_{1}+\sum_{i=1}^{l-1} f_{i}(h_{i})$ . The key insight behind residual connections is identity mapping: each layer preserves a direct path for both information and gradients to flow unchanged. During back-propagation, the gradient with respect to an intermediate hidden state is:
$$\frac{\partial \mathcal{L}}{\partial h_{l}}=\frac{\partial \mathcal{L}}{\partial h_{L}} \cdot \prod_{j=l}^{L-1}\left(I+\frac{\partial f_{j}}{\partial h_{j}}\right)$$

Expanding this product yields I plus higher-order terms involving the layer Jacobians $\partial f_{j} / \partial h_{j}$ . The identity term is always preserved, providing a direct gradient path from the loss to any layer regardless of depth.

#### Generalizing Residuals
While effective, the fixed unit coefficients in the residual update treat every layer’s contribution uniformly, offering no mechanism to adapt the mixing across depth. Highway networks [45] relax this by introducing learned element-wise gates:
$$h_{l}=\left(1-g_{l}\right) \odot h_{l-1}+g_{l} \odot f_{l-1}\left(h_{l-1}\right)$$
where $g_{l} \in[0,1]^{d}$ interpolates between the transformation and the identity path. More generally, both are instances of a weighted recurrence $h_{l}=\alpha_{l} \cdot h_{l-1}+\beta_{l} \cdot f_{l-1}(h_{l-1})$ , with residual setting $\alpha_{l}=\beta_{l}=1$ and Highway setting $\alpha_{l}=1-g_{l}$，$\beta_{l}=g_{l}$。

#### Limitations
Whether fixed or gated, both approaches share a fundamental constraint: each layer can only access its immediate input $h_{l-1}$ , a single compressed state that conflates all earlier layer outputs, rather than the individual outputs themselves. This entails several limitations:
1. **no selective access**: different layer types (e.g., attention vs. MLP) receive the same aggregated state, despite potentially benefiting from different weightings;
2. **irreversible loss**: information lost through aggregation cannot be selectively recovered in deeper layers;
3. **output growth**: later layers learn increasingly larger outputs to gain influence over the accumulated residual, which can destabilize training.

These limitations motivate a mechanism that lets each layer selectively aggregate information from all preceding layers.

## 3 Attention Residuals: A Unified View of Time and Depth
The limitations discussed above are reminiscent of similar bottlenecks in sequence modeling, suggesting that we seek similar solutions for the depth dimension.

### The Duality of Time and Depth
Like RNNs over time, residual connections compress all prior information into a single state $h_{l}$ over depth. For sequence modeling, the Transformer improved upon RNNs by replacing recurrence with attention [3, 52], allowing each position to selectively access all previous positions with data-dependent weights. We propose the same methodology for depth:
$$h_{l}=\alpha_{0 \to l} \cdot h_{1}+\sum_{i=1}^{l-1} \alpha_{i \to l} \cdot f_{i}\left(h_{i}\right)$$
where $\alpha_{i \to l}$ are layer-specific attention weights satisfying $\sum_{i=0}^{l-1} \alpha_{i \to l}=1$ . Unlike sequence length (which can reach millions of tokens), network depth is typically modest ( $L<1000$ ), making $O(L^{2})$ attention over depth computationally feasible. We call this approach Attention Residuals, abbreviated as AttnRes.

### 3.1 Full Attention Residuals
The attention weights can be written as $\alpha_{i \to l}=\phi(q_{l}, k_{i})$ for a kernel function $\phi: \mathbb{R}^{d} ×\mathbb{R}^{d} \to \mathbb{R}_{≥0}$ , where $q_{l}$ and $k_{i}$ are query and key vectors [23, 70]. Different choices of ϕ recover different residual variants (§6.2); we adopt $\phi(q, k)=\exp (q^{\top} RMSNorm(k))$ [66] with normalization, yielding softmax attention over depth:
$$\alpha_{i \to l}=\frac {\phi\left(q_{l}, k_{i}\right)}{\sum_{j=0}^{l-1} \phi\left(q_{l}, k_{j}\right)} \tag{2}$$

For each layer $l$ , we define:
$$q_{l}=w_{l}, k_{i}=v_{i}=\left\{ \begin{array} {ll}h_{1}&i=0\\ f_{i}\left(h_{i}\right) & 1\leq i\leq l-1\end{array} \right. \tag{3}$$
where the query $q_{l}=w_{l}$ is a layer-specific learnable vector in $\mathbb{R}^{d}$ . The RMSNorm inside ϕ prevents layers with large-magnitude outputs from dominating the attention weights. The input to layer l is then:
$$h_{l}=\sum_{i=0}^{l-1} \alpha_{i \to l} \cdot v_{i} \tag{4}$$

We call this form full attention residuals. For each token, Full AttnRes requires $O(L^{2} d)$ arithmetic and $O(L d)$ memory to store layer outputs. Since depth is far smaller than sequence length, the arithmetic cost is modest.

#### Overhead
The $O(L d)$ memory overlaps entirely with the activations already retained for backpropagation, so Full AttnRes introduces no additional memory overhead in vanilla training. At scale, however, activation recomputation and pipeline parallelism are widely adopted: layer outputs that would otherwise be freed and recomputed must now be kept alive for all subsequent layers, and under pipeline parallelism each must further be transmitted across stage boundaries. Both the memory and communication overhead then grow as $O(L d)$ .

#### Blockwise optimization
A deliberate design choice in Full AttnRes is that the pseudo-query $w_{l}$ is a learned parameter decoupled from the layer’s forward computation. This independence means that attention weights for any group of layers can be computed in parallel without waiting for their sequential outputs, and in particular permits grouping the L layers into N blocks of S layers each and batching the attention computation within each block, reducing per-layer memory I/O from $O(L d)$ to $O((S+N) d)$ (we defer the detailed two-phase strategy to §4). Under current distributed training regimes, however, the dominant cost is not local memory bandwidth but cross-stage communication under pipeline parallelism: every layer output must still be transmitted between stages, and this $O(L d)$ communication overhead cannot be alleviated by local batching. This motivates the Block AttnRes variant introduced below, which reduces the number of cross-stage representations from L to N . We anticipate that future interconnect improvements will make the full $O(L d)$ communication practical, fully realizing the potential of Full AttnRes.

### 3.2 Block Attention Residuals
We propose Block Attention Residuals, which partitions the L layers into N blocks: within each block, the layer outputs are reduced to a single representation via summation, and across blocks, we apply full attention over only N block-level representations and the token embedding. This reduces both memory and communication overhead from $O(L d)$ to $O(N d)$ .

#### Intra-Block Accumulation
Specifically, we divide the L layers into N blocks of $S=L / N$ layers each, assuming L is divisible by N ; otherwise, the last block contains the remaining L mod N layers. Let $B_{n}$ denote the set of layer indices in block $n(n=1, ..., N)$ ). To form a block, we sum all of its layer outputs:
$$b_{n}=\sum_{j \in \mathcal{B}_{n}} f_{j}\left(h_{j}\right)$$

We further denote $b_{n}^{i}$ as the partial sum over the first i layers in $B_{n}$ , so that $b_{n}=b_{n}^{S}$ . When L is not divisible by N , the final partial sum is taken as the last block’s representation. As in Full AttnRes, the RMSNorm inside ϕ prevents magnitude differences between complete blocks and partial sums from biasing the attention weights.

#### Inter-Block Attention
In Full AttnRes, the input to layer l is computed by attending over all outputs up to $f_{l-1}(h_{l-1})$ . The block-wise variant replaces these individual outputs with block representations, defining $b_{0}=h_{1}$ so that the token embedding is always included as a source. For the i-th layer in block n , the value matrix is:
$$V= \begin{cases}\left[b_{0}, b_{1}, ..., b_{n-1}\right]^{\top} & if i=1 (first layer of block n ) \\ \left[b_{0}, b_{1}, ..., b_{n-1}, b_{n}^{i-1}\right]^{\top} & if i \geq 2 (subsequent layers) \end{cases}$$

Keys and attention weights follow Eq. 3 and Eq. 2. The input of the very first layer of the network is the token embeddings, i.e. $b_{0}=h_{1}$ . In each block, the first layer receives the previous block representations and the token embeddings, and the subsequent layers additionally attend to the partial sum $b_{n}^{i-1}$ . The final output layer aggregates all N block representations. Fig. 2 provides PyTorch-style pseudocode for Block AttnRes.

```python
def block_attn_res(blocks: list[Tensor], partial_block: Tensor, proj: Linear, norm: RMSNorm) -> Tensor:
    """
    Inter-block attention: attend over block reps + partial sum.
    blocks: list of [B, T, D]: completed block representations for each previous block
    partial_block: [B, T, D]: intra-block partial sum (b_n^i)
    """
    V = torch.stack(blocks + [partial_block]) # [N+1, B, T, D]
    K = norm(V)
    logits = torch.einsum('d, n b t d -> n b t', proj.weight.squeeze(), K)
    h = torch.einsum('n b t, n b t d -> b t d', logits.softmax(0), V)
    return h

def forward(self, blocks: list[Tensor], hidden_states: Tensor) -> tuple[list[Tensor], Tensor]:
    partial_block = hidden_states
    # apply block attnres before attn
    # blocks already include token embedding
    h = block_attn_res(blocks, partial_block, self.attn_res_proj, self.attn_res_norm)
    
    # if reaches block boundary, start new block
    # block_size counts ATTN + MLP; each transformer layer has 2
    if self.layer_number % (self.block_size // 2) == 0:
        blocks.append(partial_block)
        partial_block = None
    
    # self-attention layer
    attn_out = self.attn(self.attn_norm(h))
    partial_block = partial_block + attn_out if partial_block is not None else attn_out
    
    # apply block attnres before MLP
    h = block_attn_res(blocks, partial_block, self.mlp_res_proj, self.mlp_res_norm)
    
    # MLP layer
    mlp_out = self.mlp(self.mlp_norm(h))
    partial_block = partial_block + mlp_out
    
    return blocks, partial_block
```
*Figure 2: PyTorch-style pseudo code for Block Attention Residuals. block_attn_res computes softmax attention over block representations using a learned pseudo-query $w_{i}$. forward is a single-layer pass that maintains partial_block ( $b_{n}^{i}$ , intra-block residual) and blocks $([b_{0}, ..., b_{n-1}]$ , inter-block history).*

#### Efficiency
Since each layer now attends over N block representations rather than L individual outputs, memory reduces from $O(L)$ to $O(N)$ and computation from $O(L^{2})$ to $O(N^{2})$ . The block count N interpolates between two extremes: $N=L$ recovers Full AttnRes, while $N=1$ reduces to standard residual connections with the embedding isolated as $b_{0}$ . Empirically, we find that $N ≈8$ recovers most of the benefit across model scales, requiring only eight stored hidden states per token (see § 5).

Beyond memory and computation, the block structure also benefits inference latency: block boundaries define the dispatch granularity for the blockwise optimization described in §3, and the fixed block count N bounds the KV cache size. The parallel inter-block results are merged with the sequential intra-block partial sums via online softmax [31], preserving exact equivalence (§4).

## 4 Infrastructure Design
Block AttnRes introduces additional system challenges compared to standard residual connections. For large-scale model training, block representations must be propagated across pipeline stages, causing heavy communication in a naïve implementation. During inference, repeated access to accumulated block representations increases latency, while long-context prefilling amplifies the memory cost of caching block representations. We address these challenges with cross-stage caching in training, and with a two-phase computation strategy together with a memory-efficient prefilling scheme in inference.

### 4.1 Training
For small-scale training, AttnRes adds a tiny computation overhead and no extra memory usage, as the activations need to be saved for backpropagation regardless. Under large-scale distributed training, pipeline parallelism poses the primary infrastructure challenge for AttnRes. Full AttnRes requires all L layer outputs to be transmitted across stages; Block AttnRes reduces this to N block representations, and the optimizations below further minimize the remaining overhead.

#### Pipeline communication
With standard residual connections, pipeline parallelism [18] transfers a fixed-size hidden state between adjacent stages, independent of pipeline depth. Block AttnRes requires all accumulated block representations at each stage for inter-block attention, and naïvely transmitting the full history at every transition incurs redundant communication.

Consider an interleaved pipeline schedule [33] with P physical stages and V virtual stages per physical stage. For simplicity, assume each physical stage produces on average $N_{p}$ block representations of dimension d per token. With $C=P V$ total chunks (each physical stage in each virtual stage), the j -th chunk accumulates $j N_{p}$ blocks. Naïvely transmitting all accumulated blocks at every transition incurs per-token communication cost:
$$Comm_{naive }=\sum_{j=1}^{C-1} j N_{p} \cdot d=\frac{C(C-1)}{2} N_{p} d . \tag{7}$$

#### Cross-stage caching
Since each physical stage processes multiple virtual stages in succession, we can eliminate this redundancy by caching blocks locally: blocks received during earlier virtual stages remain in local memory and need not be re-transmitted. The first virtual stage ( $v=1$ ) has no cache and accumulates normally; for $v ≥2$ , each transition conveys only the $P N_{p}$ incremental blocks accumulated since the receiver’s corresponding chunk in the previous virtual stage. Total communication reduces to:
$$Comm_{cached}=\underbrace {\frac{P(P-1)}{2} N_{p}d}_{first virtual stage }+\underbrace {(V-1) P^{2} N_{p} d}_{subsequent virtual stages }.$$

Caching reduces peak per-transition cost from $O(C)$ to $O(P)$ ,a $V \times$ improvement that enables full overlap with computation during steady-state 1F1B. The backward pass benefits from the same scheme. Fig. 3 illustrates this optimization with $P=4$ and $V=2$ : for the second virtual stage, caching eliminates 6 redundant block transmissions.

*Note: In practice, block boundaries need not align with physical stage boundaries. For example, in Fig. 3, each block spans two physical stages, so only every other transition involves a newly completed block.*

*Figure 3: Cache-based pipeline communication example with 4 physical ranks and 2 virtual stages per rank, where hatched boxes denote end of AttnRes blocks. Numbers indicate micro-batch indices. Each rank caches previously received blocks; stage transitions only transmit incremental blocks ( $(+[b_{1}, b_{2}])$ ) instead of the full history.*

#### Memory overhead
With cross-stage caching, each block is stored exactly once across all V virtual stages, which becomes negligible relative to standard per-layer activation cache. Crucially, the per-layer activation footprint remains identical to standard architectures, as activation checkpointing eliminates all inter-block attention intermediates, and the checkpointed input $p_{l}$ matches the memory size of the hidden state $h_{l}$ it replaces.

In terms of wall-clock time, Block AttnRes adds negligible training overhead when pipeline parallelism is not enabled; under pipeline parallelism, the measured end-to-end overhead is less than 4%.

### 4.2 Inference
The two-phase computation strategy described below applies to both Full and Block AttnRes: in either case, layers are grouped into blocks of size S , with Phase 1 batching the inter-block queries and Phase 2 handling sequential intra-block lookback. For Full AttnRes, this reduces per-layer I/O from $O(L d)$ to $O((S+N) d)$ (detailed derivation shown in Appendix B); Block AttnRes further reduces the stored representations from L to N , since each block is compressed into a single vector. In what follows, we focus on Block AttnRes and detail the two-phase computation strategy together with a sequence-sharded prefilling scheme for long-context inputs.

#### Two-phase computation strategy
The layer-wise attention computation of Block AttnRes resembles autoregressive decoding, where block representations serve as a shared KV cache reused across layers. A naïve implementation computes the attention residual at every layer, each requiring a full pass over all preceding blocks, resulting in $O(L \cdot N)$ memory accesses. Since the pseudo-query vectors are decoupled from the forward computation (§3), all $S=L / N$ queries within a block can be batched into a single matrix multiplication, amortizing memory access from S reads to 1.

Algorithm 1 instantiates a two-phase computation strategy exploiting this property.
- **Phase 1** computes inter-block attention for all S layers simultaneously via a single batched query against the cached block representations, returning both outputs and softmax statistics (max and log-sum-exp). This amortizes the memory access cost, reducing reads from S times to just once per block.
- **Phase 2** computes intra-block attention sequentially for each layer using the evolving partial sum, then merges with Phase 1 outputs through online softmax [31]. Because the online-softmax merge is elementwise, this phase naturally admits kernel fusion with surrounding operations, further reducing I/O overhead.

**Algorithm 1: Two-phase computation for block n**
Input: Pseudo queries $\{w_l\}_{l∈Bn}$, block representations $\{b0, . . . , bn−1\}$
```
/* Phase 1: Parallel inter-block attention */
// [S, d] // [n, d]
1 Q ←[wl]l∈Bn
2 K, V ←[b0; . . . ; bn−1]
3 {o(1) l , m(1) l , ℓ(1) l }l∈Bn ←ATTNWITHSTATS(Q, K, V) // Return LSE
4
/* Phase 2: Sequential intra-block attention + Online softmax merge */
5 i ←0
6 for l ∈Bn do
7    if i = 0 then
8        hl ←o(1) l /ℓ(1) l // Inter-block only
9    else
10       o(2) l , m(2) l , ℓ(2) l ←ATTNWITHSTATS(wl, bi n, bi n) // Intra-block
11       ml ←max(m(1) l , m(2) l )
12       hl ←$\frac{e^{m_l - m^{(1)}_l}o^{(1)}_l + e^{m_l - m^{(2)}_l}o^{(2)}_l}{e^{m_l - m^{(1)}_l}\ell^{(1)}_l + e^{m_l - m^{(2)}_l}\ell^{(2)}_l}$ // Online softmax merge
13    i ←i + 1
14    bi n ←bi−1 n + fl(hl) // Update partial sum; b0 n := 0
15 return {hl}l∈Bn
```

With the two-phase design, Phase 2 preserves an I/O footprint similar to that of standard residual connections, whereas the main additional cost arises from Phase 1 inter-block attention. Because these inter-block reads are amortized across all layers in a block through batching, the total per-layer memory access cost remains only $(\frac{N}{S}+3) d$ reads and $2 d$ writes (Table 1). This is substantially lower than the residual-stream I/O of prior residual generalizations such as (m)HC under typical settings. In practice, Phase 1 can also partially overlap with the computation of the first layer in the block, further reducing its wall-clock impact. As a result, the end-to-end inference latency overhead is less than 2% on typical inference workloads.

*Table 1: Memory access cost per token per layer incurred by the residual mechanism under each scheme. The internal I/O of the layer function $f_{t}$ is excluded. For AttnRes, both Full and Block variants use the two-phase inference schedule described in Appendix B; amortized costs are averaged over N layers within a block. Typical values: $L=128$ , $N=8$ ,$S=L / N=16$ , $m=4$*

| Scheme |  | Reads | Writes |
| ---- | ---- | ---- | ---- |
| Standard Residuals |  | 3d | 3d |
| mHC |  | d+m | md |
| Full AttnRes | Phase 1 (amortized) | 24d | - |
|  | Phase 2 | - | - |
| Block AttnRes |  | (N+5)d | 5.5d |

#### Memory-efficient prefilling
Storing block representations during prefilling requires $N \cdot T \cdot d$ elements, which incurs 15 GB of memory for a 128K-token sequence with 8 blocks. We mitigate this by sharding these representations along the sequence dimension across P tensor-parallel devices, allowing Phase 1 to execute independently on local sequence shards. The Phase 2 online-softmax merge then integrates into the standard TP all-reduce communication path: the output is reduce-scattered, merged locally, and reconstructed via all-gather, naturally admitting kernel fusion with operations like RMSNorm. This reduces the per-device memory footprint to $N \cdot(T / P)·d$—lowering the 128K-context example from 15 GB to roughly 1.9 GB per device. Combined with chunked prefill (e.g., 16K chunk size), the overhead further reduces to under 0.3 GB per device.

## 5 Experiments
### Architecture Details
Our architecture is identical to Kimi Linear [69], a Mixture-of-Experts (MoE) Transformer following the Moonlight [28] / DeepSeek-V3 [9] design, which interleaves Kimi Delta Attention (KDA) and Multi-Head Latent Attention (MLA) layers in a 3:1 ratio, each followed by an MoE feed-forward layer. The only modification is the addition of AttnRes to the residual connections; all other components (model depth, hidden dimensions, expert routing, and MLP structure) remain unchanged. AttnRes introduces only one RMSNorm and one pseudo-query vector $w_{l} \in \mathbb{R}^{d}$ per layer, amounting to a negligible fraction of the total parameter count. Crucially, all pseudo-query vectors must be initialized to zero. This ensures that the initial attention weights $\alpha_{i \to l}$ are uniform across source layers, which reduces AttnRes to an equal-weight average at the start of training and prevents training volatility, as we validated empirically.

### 5.1 Scaling Laws
We sweep five model sizes (Table 2) and train three variants per size: a PreNorm baseline, Full AttnRes, and Block AttnRes with ≈8 blocks. They are trained with an 8192-token context window and a cosine learning rate schedule. Within each scaling law size group, all variants share identical hyperparameters selected under the baseline to ensure fair comparison; this setup intentionally favors the baseline and thus makes the comparison conservative. Following standard practice, we fit power-law curves of the form $L=A ×C^{-\alpha}$ [22, 15], where L is validation loss and C is compute measured in PFLOP/s-days.

#### Scaling Behavior
Fig. 4 presents the fitted scaling curves. The Baseline follows $L=1.891 ×C^{-0.057}$ , while Block AttnRes fits $L=1.870 ×C^{-0.058}$ , and Full AttnRes fits $L=1.865 ×C^{-0.057}$ . All three variants exhibit a similar slope, but AttnRes consistently achieves lower loss across the entire compute range. Based on the fitted curves, at 5.6 PFLOP/s-days, Block AttnRes reaches 1.692 versus the Baseline’s 1.714, equivalent to a 1.25× compute advantage. The gap between Full and Block AttnRes narrows with scale, shrinking to just 0.001 at the largest size. We also list mHC(-lite) [64] in Table 2 for reference. Full AttnRes outperforms mHC, while Block AttnRes matches it at lower memory I/O per layer: 5.5d versus 34d for mHC with $m=4$ streams (Table 1).

*Table 2: Baseline vs Block AttnRes ( $N=8$ ) vs Full AttnRes vs mHC(-lite) [64]: Model configurations, Hyperparameters, and Validation Loss.
† Denotes the number of activated parameters in our MoE models, excluding embeddings. ‡ All models were trained with a context length of 8192. * $L_{b}=L / 2$ denotes the number of Transformer blocks.*

| #Act. Params† | Tokens | Lb | H | dmodel | da | lr | batch size‡ | Baseline | Block AttnRes | Full AttnRes | mHC(-lite) |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 194M | 38.7B | 12 | 12 | 896 | 400 | 2.99x10-3 | 192 | 1.931 | 1.909 | 1.899 | 1.906 |
| 241M | 45.4B | 13 | 13 | 960 | 432 | 2.80x10-3 | 256 | 1.895 | 1.875 | 1.874 | 1.869 |
| 296M | 62.1B | 14 | 14 | 1024 | 464 | 2.50x10-3 | 320 | 1.829 | 1.809 | 1.804 | 1.807 |
| 436M | 87.9B | 16 | 16 | 1168 | 528 | 2.20x10-3 | 384 | 1.766 | 1.746 | 1.737 | 1.747 |
| 528M | 119.0B | 17 | 17 | 1264 | 560 | 2.02x10-3 | 432 | 1.719 | 1.693 | 1.692 | 1.694 |

*Figure 4: Scaling law curves for Attention Residuals. Both Full and Block AttnRes consistently outperform the baseline across all scales. Block AttnRes closely tracks Full AttnRes, recovering most of the gain at the largest scale.*

### 5.2 Main Results
#### Training recipe
The largest models we study are based on the full Kimi Linear 48B configuration: 27 Transformer blocks (54 layers) with 8 out of 256 routed experts plus 1 shared expert, yielding 48B total and 3B activated parameters. This model applies Block AttnRes with 6 layers per block, producing 9 blocks plus the token embedding for a total of 10 depth-wise sources.

We follow the same data and training recipe as the Kimi Linear 1.4T-token runs [69]: all models are pre-trained with a 4096-token context window, the Muon optimizer [28], and a WSD (Warmup–Stable–Decay) learning rate schedule [16], with a global batch size of 8M tokens. Training of the final model proceeds in two stages: (i) a WSD pre-training phase on 1T tokens, followed by (ii) a mid-training phase on ≈400B high-quality tokens, following the annealing recipe of Moonlight [28].

After mid-training, we continue training with progressively longer sequence length of 32K tokens. Since our architecture uses hybrid KDA/MLA attention [69], where MLA operates without positional encodings (NoPE) [61], context extension requires no modifications such as YaRN [37] or attention temperature rescaling.

#### Training dynamics
We compare the training dynamics of our final Baseline and Block AttnRes models over 1T tokens in Fig. 5.

*Figure 5: Training dynamics of Baseline and Block AttnRes. (a) Validation loss during training. (b) Each transformer block’s output magnitude at the end of training. (c) Each transformer block’s gradient magnitude.*

1. **Validation loss**: AttnRes achieves consistently lower validation loss throughout training, with the gap widening during the decay phase and resulting in a notably lower final loss.
2. **Output magnitude**: The Baseline suffers from the PreNorm dilution problem [60, 27]: as hidden-state magnitudes grow monotonically with depth, deeper layers are compelled to learn increasingly large outputs from fixed-scale normalized inputs to remain influential. Block AttnRes confines this growth within each block, as selective aggregation at block boundaries resets the accumulation, yielding a bounded periodic pattern.
3. **Gradient magnitude**: With all residual weights fixed to 1, the Baseline provides no means of regulating gradient flow across depth, leading to disproportionately large gradients in the earliest layers. The learnable softmax weights in Block AttnRes (Fig. 8) introduce competition among sources for probability mass, resulting in a substantially more uniform gradient distribution.

#### Downstream performance
Following the evaluation protocol of Kimi Linear [69], we assess both models across three areas (Table 3):
1. **Language understanding and reasoning**: MMLU [13], MMLU-Pro Hard [55], GPQA-Diamond [41], BBH [48], ARC-Challenge [6], HellaSwag [65], and TriviaQA [21].
2. **Reasoning (Code and Math)**: GSM8K [7], MGSM [44], Math [25], CMath [14], HumanEval [5], and MBPP [1].
3. **Chinese language understanding**: CMMLU [26] and C-Eval [19].

*Table 3: Performance comparison of AttnRes with the baseline, both after the same pre-training recipe. Best per-row results are bolded.*

| Task Type | Task | Baseline | AttnRes |
| ---- | ---- | ---- | ---- |
| General | MMLU | 73.5 | **74.6** |
|  | MMLU-Pro | 52.2 | **52.2** |
|  | GPQA-Diamond | 36.9 | **44.4** |
|  | BBH | 76.3 | **78.0** |
|  | ARC-Challenge | 64.6 | **65.7** |
|  | HellaSwag | 83.2 | **83.4** |
|  | TriviaQA | 69.9 | **71.8** |
| Math & Code | GSM8K | 81.7 | **82.4** |
|  | MGSM | 64.9 | **66.1** |
|  | Math | 53.5 | **57.1** |
|  | CMath | 84.7 | **85.1** |
|  | HumanEval | 59.1 | **62.2** |
|  | MBPP | 72.0 | **73.9** |
| Chinese | CMMLU | 82.0 | **82.9** |
|  | C-Eval | 79.6 | **82.5** |

As shown in Table 3, Block AttnRes matches or outperforms the baseline on all benchmarks. The improvements are particularly pronounced on multi-step reasoning tasks such as GPQA-Diamond (+7.5) and Minerva Math (+3.6), as well as code generation such as HumanEval (+3.1), while knowledge-oriented benchmarks such as MMLU (+1.1) and TriviaQA (+1.9) also show solid gains. This pattern is consistent with the hypothesis that improved depth-wise information flow benefits compositional tasks, where later layers can selectively retrieve and build upon earlier representations.

### 5.3 Ablation Study
We conduct ablation studies on the 16-head model from Table 2 to validate key design choices in AttnRes (Table 4). All models share identical hyperparameters and compute budget.

*Table 4: Ablation on key components of AttnRes (16-layer model).*

| Variant | Loss |
| ---- | ---- |
| Baseline (PreNorm) | 1.766 |
| DenseFormer [36] | 1.767 |
| mHC [59] | 1.747 |
| AttnRes Full | 1.737 |
| AttnRes Full w/ input-dependent query | 1.731 |
| AttnRes Full w/ input-independent mixing | 1.749 |
| AttnRes SWA (W = 1 + 8) | 1.764 |
| AttnRes Block (S = 4) | 1.746 |
| AttnRes Block w/ sigmoid | 1.741 |
| AttnRes Block w/o RMSNorm | 1.750 |
| AttnRes Block w/ multihead (H = 16) | 1.752 |
| AttnRes Full w/o RMSNorm | 1.743 |

#### Comparison with prior methods
We compare AttnRes against the PreNorm baseline (loss 1.766) and two representative methods that generalize residual connections. DenseFormer [36] grants each layer access to all previous outputs but combines them with fixed, input-independent scalar coefficients; it shows no gain over the baseline (1.767), highlighting the importance of input-dependent weighting. mHC [59] introduces input dependence through m parallel streams with learned mixing matrices, improving to 1.747. AttnRes takes this further with explicit content-dependent selection via softmax attention: Full AttnRes achieves 1.737 and Block AttnRes 1.746, outperforming both methods with only a single query vector per layer.

#### Cross-layer access
We compare three granularities of cross-layer access. Full AttnRes follows directly from the time–depth duality (§ 3), applying attention over all previous layers, and achieves the lowest loss (1.737). A simple way to reduce its memory cost is sliding-window aggregation (SWA), which retains only the most recent $W=8$ layer outputs plus the token embedding; it improves over baseline (1.764) but falls well short of both Full and Block AttnRes, suggesting that selectively accessing distant layers matters more than attending to many nearby ones.

Block AttnRes offers a better trade-off: with block size $S=4$ it reaches 1.746 while keeping memory overhead constant per layer. Fig. 6 sweeps S across the full spectrum from $S=1$ (i.e. Full AttnRes) to increasingly coarse groupings. Loss degrades gracefully as S grows, with $S=2,4,8$ all landing near 1.746 while larger blocks ( $S=16,32$ ) move toward baseline. In practice, we fix the number of blocks to $≈8$ for infrastructure efficiency (§ 4). As future hardware alleviates memory capacity constraints, adopting finer-grained block sizes or Full AttnRes represents a natural pathway to further improve performance.

*Figure 6: Effect of block size on validation loss (16-layer model).*

#### Component design
We further ablate individual components of the attention mechanism:
1. **Input-dependent query**: A natural extension is to make the query input-dependent by projecting it from the current hidden state. This further lowers loss to 1.731, but introduces a $d ×d$ projection per layer and requires sequential memory access during decoding, so we default to the learned query.
2. **Input-independent mixing**: We removed the query and key and replaced them with learnable, input-independent scalars to weigh previous layers, which hurts performance (1.749 vs. 1.737).
3. **softmax vs. sigmoid**: Replacing softmax with sigmoid degrades performance (1.741). We attribute this to softmax’s competitive normalization, which forces sharper selection among sources.
4. **Multihead attention**: We test per-head depth aggregation ( $H=16$ ) on Block AttnRes, allowing different channel groups to attend to different source layers. This hurts performance (1.752 vs. 1.746), indicating that the optimal depth-wise mixture is largely uniform across channels: when a layer’s output is relevant, it is relevant as a whole.
5. **RMSNorm on keys**: Removing RMSNorm degrades both Full AttnRes (1.743) and Block AttnRes (1.750). For Full AttnRes, it prevents individual layers with naturally larger outputs from dominating the softmax. This becomes even more critical for Block AttnRes, as block-level representations accumulate over more layers and can develop large magnitude differences; RMSNorm prevents these from biasing the attention weights.

### 5.4 Analysis
#### 5.4.1 Optimal Architecture
To understand how AttnRes reshapes optimal architectural scaling, we perform a controlled capacity reallocation study under a fixed compute and parameter budget. Our central question is whether AttnRes alters the preferred depth–width–attention trade-off, and in particular, given its potential strength on the depth dimension, whether it favors deeper models compared to conventional Transformer design heuristics. To isolate structural factors directly coupled to depth, we fix the per-expert MLP expansion ratio based on internal empirical observations ( $d_{ff} / d_{model } ≈0.45$ ). We further fix total training compute ( $FLOPs ≈6.5 ×10^{19}$ ) and active parameters ( $≈2.3 ×10^{8}$ ), ensuring that any performance variation arises purely from architectural reallocation rather than overall capacity differences. Under this constrained budget, we enumerate 25 configurations on a 5 × 5 grid over $d_{model } / L_{b} \in\{15,30,45,60,75\}$ and $H / L_{b} \in\{0.3,0.4,0.5,0.6,0.7\}$ , where $L_{b}=L / 2$ is the number of Transformer blocks and H the number of attention heads. The results are shown in Fig. 7.

*Figure 7: Architecture sweep under fixed compute ( $≈6.5 ×10^{19} FLOPs$ , $≈2.3 ×10^{8}$ active parameters). Each cell reports validation loss for a $(d_{model } / L_{b}, H / L_{b})$ configuration, where $L_{b}=L / 2$ is the number of Transformer blocks; the star marks the optimum.*

Both heatmaps exhibit a shared pattern: loss decreases with growing $d_{model } / L_{b}$ and shrinking $H / L_{b}$ , and both methods reach their optima at $H / L_{b} ≈0.3$ . Despite this shared trend, AttnRes achieves a lower loss than the baseline in each of the 25 configurations, by 0.019–0.063. The most apparent difference lies in the location of the optimum: the baseline achieves its lowest loss at $d_{model } / L_{b} ≈60(1.847)$ ), whereas AttnRes shifts it to $d_{model } / L_{b} ≈45$ (1.802). Under a fixed parameter budget, a lower $d_{model } / L_{b}$ corresponds to a deeper, narrower network, suggesting that AttnRes can exploit additional depth more effectively. We note that this preference for depth does not directly translate to a deployment recommendation, as deeper models generally incur higher inference latency due to their sequential computation [39]. Rather, this sweep serves as a diagnostic that reveals where AttnRes benefits most, and this depth preference can be factored into the architecture selection alongside inference cost.

#### 5.4.2 Analyzing Learned AttnRes Patterns
We visualize the learned weights $\alpha_{i \to l}$ in Fig. 8 for the 16-head model (from Table 2) with both full and block ( $N=8$ ) AttnRes. Each heatmap shows how the lth attention or MLP layer (rows) allocates its attention over previous sources (columns), with pre-attention and pre-MLP layers shown separately. We highlight three key observations:

*Figure 8: Depth-wise attention weight distributions for a 16-head model with full (top) and block (bottom) Attention Residuals, averaged over tokens. The model has 16 attention and 16 MLP layers. Each row shows how the lth attention (left) or MLP (right) layer distributes weight over previous sources. Diagonal dominance indicates locality remains the primary information pathway, while persistent weights on source 0 (embedding) and occasional off-diagonal concentrations reveal learned skip connections. Block attention ( $N=8$ ) recovers the essential structure with sharper, more decisive weight distributions.*

1. **Preserved locality**: Each layer attends most strongly to its immediate predecessor, yet selective off-diagonal concentrations emerge (e.g., layer 4 attending to early sources, layers 15–16 reaching back under the block setting), indicating learned skip connections beyond the standard residual path.
2. **Layer specialization**: The embedding $h_{1}$ retains non-trivial weight throughout, especially in pre-attention layers. Pre-MLP inputs show sharper diagonal reliance on recent representations, while pre-attention inputs maintain broader receptive fields, consistent with attention routing information across layers and MLPs operating locally.
3. **Block AttnRes preserves structure**: Diagonal dominance, embedding persistence, and layer specialization all transfer from the full to the block variant, suggesting that block-wise compression acts as implicit regularization while preserving the essential information pathways.

## 6 Discussions
### 6.1 Sequence-Depth Duality
Residual connections propagate information over depth via a fixed recurrence $h_{l}=h_{l-1}+f_{l-1}(h_{l-1})$ , much as RNNs propagate information over time. Test-Time Training (TTT) [46] formalizes the sequence side of this analogy (cf. Fast Weight Programmers [43, 32]), casting each recurrent step as gradient descent on a self-supervised loss:
$$W_{t}=W_{t-1}-\eta \nabla \ell\left(W_{t-1} ; x_{t}\right)$$
where a slow network parameterizes $\ell$ and the state W is updated once per token. When f is linear, this reduces to vanilla linear attention $\dot{S}_{t}=S_{t-1}+k_{t} v_{t}^{\top}$ . The standard residual exhibits the same additive form along depth, with $h_{l}$ serving as the state and each layer $f_{l}$ acting as one “gradient step.”

As noted by [4], this duality extends to richer variants (Table 5). Data-dependent gates on the sequence side [47, 63] correspond to Highway networks [45] on the depth side; the delta rule [42, 62, 69] corresponds to DDL [67]; and MRLA [10] mirrors GLA’s [63] gated linear attention. These methods all refine the recurrent update while remaining within the recurrence paradigm. AttnRes goes a step further and replaces depth-wise recurrence with direct cross-layer attention, just as Transformers replaced temporal recurrence with self-attention. Since the number of layers in current architectures remains well within the practical regime of softmax attention, we adopt vanilla depth-wise attention. Incorporating more expressive yet memory-efficient (e.g. linear-complexity) alternatives is a natural direction for future work.

*Table 5: Comparison of residual update mechanisms. Weight: whether the mixing coefficients are architecture-fixed, learned-static (fixed after training), or input-dependent (dynamic). Source: which earlier representations layer l can access. Normalization is omitted from most formulas for clarity.*

| Type | Method | Update rule | Weight | Source |
| ---- | ---- | ---- | ---- | ---- |
| Single-state recurrence: layer l receives only hi-1 | Residual [12] | $h_l=h_{l-1}+f_{l-1}(h_{l-1})$ | Fixed | $h_{l-1}$ |
|  | ReZero [2] | $h_l=h_{l-1}+\alpha_l f_{l-1}(h_{l-1})$ | Static | $h_{l-1}$ |
|  | LayerScale [50] | $h_l=h_{l-1}+diag(\lambda)·f_{l-