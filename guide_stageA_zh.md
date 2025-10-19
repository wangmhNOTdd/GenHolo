
# RAE-for-Pose —— 阶段A（方案1）训练指南（中文）
_目标：只用一张 4090D 24GB（或更高显存）在 PLINDER 的 pose 任务上完成 **阶段A：解码器训练**，为后续阶段B的潜空间生成打好基础。_

**本指南只覆盖“方案1”的配置：**
- 潜向量维度 \(d = 512\)
- 序列长度上限 \(T \le 256\)（口袋残基 + 配体原子总长度）
- 解码器：EGNN / SE(3)-Transformer **3 层**，hidden **256**
- 采用 **Holo 态**潜表示训练，启用**潜空间高斯噪声增强**
- 预计算并缓存 \(Z_{\text{holo}}\)，训练时**冻结编码器**

---

## 1. 问题与数据形态
- **任务**：给定 Holo 态复合物（蛋白口袋 + 配体），学习从潜向量 \(Z\) **稳健地还原** 3D 结构（主要是配体原子坐标，可选口袋少量柔性）。
- **输入张量（训练）**：\(\tilde Z \in \mathbb{R}^{T \times d}\)，其中 \(T \le 256, d=512\)。
- **输出张量**：\(\hat X^{L} \in \mathbb{R}^{N_L \times 3}\)（配体重原子坐标）。
- **评估**：重建类的坐标误差（L1/Huber）、键长/键角/扭转约束；阶段A完成后才进入 PLINDER 官方结构指标（lDDT-PLI/BiSyRMSD）的端到端评估。

> 阶段A只训练“**潜 → 3D**”的**解码器**；**不进行 Apo→Holo 的迁移学习**（那是阶段B的职责）。

---

## 2. 编码器与潜向量准备（离线）
**目的**：把显存压力从训练期移走，只在训练图中保留解码器。

1) **冻结编码器**（不训练）：
- 蛋白侧：ESM 系列（ESM3/ESMFold/ESM-IF 等）获取**残基级 token**；做**口袋裁剪**（见 §3）。
- 配体侧：3D 友好的分子编码器（Uni-Mol / SchNet / EGNN 系等）获取**原子级 token**。

2) **通道投影与拼接**：
- 把两侧 token 都线性投影到 \(d=512\)，再按序列拼接：  
  \[
  Z = \mathrm{Concat}(H^P, H^L) \in \mathbb{R}^{T \times 512},\quad T=T_P+T_L \le 256.
  \]

3) **离线缓存**：
- 将每个样本的 \(Z_{\text{holo}}\)、mask、配体拓扑（键图）、原子/残基类型等**写入磁盘**（.pt/.npz）。
- 训练时 DataLoader 直接读缓存，**无需前向通过编码器**。

---

## 3. 序列长度控制（口袋裁剪）
为保证 \(T \le 256\)，建议：
- 以配体重心为球心，取**半径 6–10 Å** 的残基作为口袋；
- 若仍超限，对残基按与配体的**最小重原子距离**排序，保留 Top-K（如 160–200 个残基）；
- 配体侧去氢，必要时可合并罕见超长侧链/可视作刚体部分（仅训练期）。

---

## 4. 训练输入（潜空间噪声增强）
为使解码器对阶段B输出的带噪潜向量**鲁棒**，阶段A在潜空间注入高斯噪声：
\[
\tilde{Z} = Z_{\text{holo}} + N,\quad N \sim \mathcal{N}(0,\sigma^2 I),\quad \sigma \sim \mathrm{Uniform}(0,\tau).
\]
- 推荐 \(\tau = 0.7\)。可在 \([0.5, 0.8]\) 网格搜索。
- 每个 step 独立采样 \(\sigma\) 与 \(N\)。

---

## 5. 解码器结构（方案1规格）
使用**轻量 SE(3) 解码头**（EGNN 或 SE(3)-Transformer 实现皆可）：
- 层数：**3 层**
- 隐藏维：**256**
- 输入：\(\tilde Z \in \mathbb{R}^{T\times 512}\) 与 mask；可添加几何 side-feature（残基局部坐标系、原子/键类型、键图邻接等）。
- 输出：\(\hat X^{L} \in \mathbb{R}^{N_L \times 3}\)（可用绝对坐标或残基/原子局部帧中的位移）。
- 归一化与激活：LayerNorm + GELU（或 SiLU）。
- 注意事项：
  - **消息传递范围**：可只在“配体原子”内部与“口袋-配体接触对”之间传递，降低复杂度；
  - **可选**输出口袋少量侧链扭转角，用于后处理的微调，但方案1可以先只输出配体坐标。

---

## 6. 损失函数
综合几何与化学约束：
\[
\mathcal{L}_{\text{dec}} = 
\lambda_1 \,\underbrace{\| \hat{X}^L - X^L \|_{1,\text{Huber}}}_{\text{坐标误差}}
+ \lambda_2 \,\underbrace{\mathcal{L}_{\text{chem}}}_{\text{键长/角/扭转正则}}
+ \lambda_3 \,\underbrace{\mathcal{L}_{\text{geom}}}_{\text{FAPE 或 lDDT-LP}}
+ \lambda_4 \,\underbrace{\mathcal{L}_{\text{repel}}}_{\text{短程排斥}}.
\]
- \(\mathcal{L}_{\text{chem}}\)：基于配体键图的键长/键角/扭转项；环平面/芳香性可加软约束。
- \(\mathcal{L}_{\text{geom}}\)：FAPE 或 lDDT-LP（只在口袋局部评估，以对齐 PLINDER 口径）。
- \(\mathcal{L}_{\text{repel}}\)：\(\sum_{i<j} \max(0, r_c - \|x_i-x_j\|)^2\)，对口袋-配体近邻对生效。

**建议权重（起步）**：\(\lambda_1: \lambda_2: \lambda_3: \lambda_4 = 1.0: 0.2: 0.5: 0.1\)。

---

## 7. 训练与显存设置（4090D 24GB）
- **精度**：AMP（bf16 或 fp16）
- **batch**：micro-batch **2**；**梯度累积 8** → 等效全局批量 16
- **优化器**：AdamW（lr \(1\!\sim\!3\times10^{-4}\)，wd 0.05），EMA 0.9995，梯度裁剪 1.0
- **激活检查点**：对每层 EGNN/SE(3) block 开启
- **训练轮数**：10–20 epoch 通常即可把重建收敛到稳健水平
- **I/O**：DataLoader 直接读取离线缓存的 \(Z_{\text{holo}}\)/mask/键图，避免在训练图里调用编码器

> 在上述配置下，模型峰值显存通常 **< 24GB**；若仍偏紧，可将 hidden 从 256 减到 192，或把 T 上限从 256 降到 224。

---

## 8. 伪代码（PyTorch 风格）
```python
for batch in loader:  # batch items are precomputed Z_holo, masks, ligand_graph, X_holo
    Z = batch["Z_holo"]           # [B, T, 512], float32 (loaded, not computed)
    mask = batch["mask"]          # [B, T], bool
    graph = batch["lig_graph"]    # bond indices/types, ring info, etc.
    X_holo = batch["X_holo"]      # [B, N_L, 3]

    # Noise-augmented latent
    sigma = torch.rand(B, 1, 1, device=Z.device) * tau   # tau=0.7
    Z_tilde = Z + sigma * torch.randn_like(Z)

    with autocast(dtype=torch.bfloat16):
        X_pred = decoder(Z_tilde, mask, graph)  # [B, N_L, 3]
        loss = L1Huber(X_pred, X_holo) \
             + chem_regularizer(X_pred, graph) \
             + geom_loss(X_pred, X_holo) \
             + repel_loss(X_pred, pocket_coords_subset)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
    scaler.step(optimizer); scaler.update()
    optimizer.zero_grad(set_to_none=True); ema.update()
```

---

## 9. 验证与早停
- 监控：配体坐标的 Huber/L1、键长/角/扭转误差、（可选）lDDT-LP。
- 观察：把随机 \(\sigma\) 提高/降低，看重建的鲁棒性曲线是否平滑；对比“无噪声增强”版本，验证鲁棒性增益。
- 早停：验证集上 3–5 次评估指标不再提升时停止。

---

## 10. 常见问题（FAQ）
- **Q：为什么阶段A只用 Holo 潜表示？**  
  A：阶段A的职责是“把**像样**的潜向量稳健还原成 3D”；Apo→Holo 的“运输”学习放在阶段B。把 Apo 潜向量拿来回归 Holo 坐标会混淆职责，训练更难、也不稳。
- **Q：噪声增强会降低重建精度吗？**  
  A：可能略降，但能显著提升**生成阶段**的鲁棒性（阶段B输出的潜向量往往带噪或偏移）。
- **Q：一定要 SE(3) 解码器吗？**  
  A：强烈建议。EGNN/SE(3) 在几何任务上的稳定性优于纯注意力头；若用纯注意力，需额外几何先验才能达到同等水平。

---

## 11. 完成阶段A后的交付物
- 训练好的解码器权重（仅数百万级参数量）。
- 用于阶段B的**固定接口**：`X_pred = decoder(Z_tilde, mask, graph)`。
- 验证报告：重建误差、化学/几何约束达标情况、对不同 \(\sigma\) 的鲁棒性曲线。
