# 进度 — summer

> 唯一个人进度源。状态词汇见 [../../progress.md](../../progress.md)。图例：✅ 已完成 · ▶ 材料就绪可开课 · ・ 材料待起草

## 下一节

- **先填 [baseline.md](baseline.md)**，然后从 **T0·S01** 与 **A1·S01** 开始（双线并行，均 ▶ 就绪）

## 总览

- 完成计数：**0 / 41** · ledger：0 条

## A 线 · 机制（0 / 19）

| 节 | 主题 | 状态 |
|---|---|---|
| A1·S01 | Attention 重激活：手算 + 50 行实现 | ▶ [材料](../../modules/a1-transformer/s01-attention-reactivation.md) |
| A1·S02 | Causal mask + 多头 + mini decoder | ▶ [材料](../../modules/a1-transformer/s02-causal-mask-multihead.md) |
| A1·S03 | 训练闭环：tiny GPT 训到能生成 | ・ |
| A1·S04 | KV cache 与 prefill/decode | ・ |
| A1·S05 | RoPE / RMSNorm / SwiGLU 换装 | ・ |
| A1·S06 | Induction head | ・ |
| A1·S07 | burstiness → ICL 复现 | ・ |
| A2·S01 | action token + NLL | ・ |
| A2·S02 | Gaussian+MSE 的 mode averaging | ・ |
| A2·S03 | Diffusion（DDPM） | ・ |
| A2·S04 | Flow matching | ・ |
| A2·S05 | DiT / ACT / π0.5 三方对照 | ・ |
| A3·S01 | 视觉基础组件：ViT/CLIP/DINOv2 | ・ |
| A3·S02 | Projection / adapter | ・ |
| A3·S03 | Latent query / resampler | ・ |
| A3·S04 | 双塔与对比对齐 | ・ |
| A3·S05 | MoT：π0.5 解剖 | ・ |
| A3·S06 | MoE 实现 | ・ |
| A3·S07 | Generator × Reasoner / WAM | ・ |

## T 线 · 训练（0 / 22）

| 节 | 主题 | 状态 |
|---|---|---|
| T0·S01 | Loss 作为接口：第一次闭环 | ▶ [材料](../../modules/t0-training-loop/s01-loss-as-interface.md) |
| T0·S02 | 优化几何与数据分布：五臂干预 | ▶ [材料](../../modules/t0-training-loop/s02-five-arm-interventions.md) |
| T1·S01 | pre/mid/post 全景 + Cosmos 3 | ・ |
| T1·S02 | Continual pretraining toy | ・ |
| T1·S03 | π0.5 配方与自身 SFT 定位 | ・ |
| T1·S04 | CT vs SFT 判断演练 | ・ |
| T2·S01 | Tabular：Bellman T 落成代码 | ・ |
| T2·S02 | REINFORCE 从零 | ・ |
| T2·S03 | A2C 与 stop-gradient | ・ |
| T2·S04 | GAE 实现 + λ 扫描 | ・ |
| T2·S05 | IS 推导 → PPO ratio/clip | ・ |
| T2·S06 | PPO 对齐 CleanRL | ・ |
| T2·S07 | 选修：DQN 或 SAC | ・ |
| T3·S01 | PPO 连续控制 | ・ |
| T3·S02 | SAC 连续控制 | ・ |
| T3·S03 | Domain randomization + teacher-student | ・ |
| T3·S04 | Sim2Real 判断框架 | ・ |
| T3·S05 | Offline RL / AWR / IQL | ・ |
| T3·S06 | Flow-policy RL estimator | ・ |
| T3·S07 | Capstone：selective-RL proposal | ・ |
| T4·S01 | 现代训练循环工程 + JAX 对照 | ・ |
| T4·S02 | 分布式：DDP / FSDP | ・ |

## 更新记录

| 日期 | 课次 | 状态变化 | 备注 |
|---|---|---|---|
| 2026-08-29 | — | 学员目录建立，待填 baseline | |
