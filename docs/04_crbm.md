# `src/crbm.py` 分析文档

## 1. 文件职责

实现**连续受限玻尔兹曼机（CRBM, Chen & Murray 2003）**，作为深度自编码器
逐层无监督预训练的"积木"。对应论文 **2.1 节，Eq.(2)–(7)**。

## 2. 在流水线中的位置

```
step03_pretrain_crbm.py ──实例化并训练──▶ CRBM（每层一个）
                                        │
                                        └─ transform(v) 得到隐层表示 →
                                           作为下一层 CRBM 的可见层输入
```

CRBM 只负责"学权重"，不直接产出最终异常图。

## 3. 模型定义

### 3.1 结构

可见层 v（连续，取值 [0,1]）与隐层 h 之间有对称连接矩阵 `W`，
两层各有偏置 `bv`、`bh`，以及**噪声控制参数** `a_v`、`a_h`。

| 数组 | 形状 | 含义 |
| --- | --- | --- |
| `W` | `(n_visible, n_hidden)` | 连接权重 |
| `bv` | `(n_visible,)` | 可见层偏置 |
| `bh` | `(n_hidden,)` | 隐层偏置 |
| `a_v` / `a_h` | `(n_visible,)` / `(n_hidden,)` | 噪声控制参数（sigmoid 斜率） |

### 3.2 前向与重构（论文 Eq.2 / Eq.3）

```python
h = phi(W^T ... )        # 实际代码：pre = v @ W + bh; h = phi(pre, a_h)
v = phi(h @ W^T + bv, a_v)
```

其中激活函数为带噪声控制的 sigmoid（论文 Eq.4）：

$$\phi(x;a)=\theta_L+\frac{\theta_H-\theta_L}{1+\exp(-a x)}$$

代码里 `theta_L=0, theta_H=1`（因为数据已归一化到 [0,1]），
并对 `-a*x` 做了 `clip(-60, 60)` 防止 `exp` 溢出。

### 3.3 CD-1 训练（论文 Eq.5–7）

每批数据执行"正相 → 负相"一步对比散度：

```
h0 = phi(v0 W + bh + σ·N(0,1))          # 正相
v1 = phi(h0 Wᵀ + bv + σ·N(0,1))         # 负相（重构）
h1 = phi(v1 W + bh + σ·N(0,1))
ΔW  = v0ᵀh0 − v1ᵀh1
Δbv = v0 − v1
Δbh = h0 − h1
```

**动量实现的关键细节**：学习率放在累积项**内部**

```python
self._dW = momentum * self._dW + lr * (gW - weight_decay * W)
self.W  += self._dW
```

初版曾写成 `self._dW = momentum*self._dW + gW; W += lr*self._dW`，
这会把有效步长放大约 `1/(1-momentum)=10` 倍，导致训练发散
（重构 MSE 先降后升）。改成上式后训练稳定。

## 4. 关于噪声控制参数 `a` 的重要说明

论文 Eq.(6) 用"输入方差之差"更新 `a`：

$$\Delta a_j \propto \frac{\langle s_j^2\rangle-\langle s_j\rangle^2}{a_j^2}$$

**实测结论：该更新在本区数据上不可用。**

| 配置 | 200 epoch 后重构 MSE | 结果 |
| --- | --- | --- |
| `learn_a=False`（a 固定 = 1） | **0.00393** | 收敛，矿点/背景误差比 1.44 |
| `learn_a=True`, lr_a=0.01, momentum=0.9 | 0.03942 | **发散**，a 顶到上界 20，sigmoid 饱和 |
| `learn_a=True`, lr_a=0.01, momentum=0（关闭动量） | 0.00504 | 勉强收敛 |
| `learn_a=True`, lr_a=0.001, lr=0.1, momentum=0.9 | 0.00424 | 收敛但更慢 |

原因是该式是启发式的方差差，在 CD 不动点附近不严格为零，
再叠加动量就会单调地把 `a` 推向边界。

**因此本项目默认 `CRBM_LEARN_A=0`（a 固定为 1）**，并在 `config.py`
中以注释形式说明这是对论文的一处**必要偏离**。此时模型等价于
"可见层连续、隐层 sigmoid 的高斯-伯努利型 RBM"，仍然完整保留论文的核心机制
（小概率样本重构误差高）。若要在论文方向上继续实验，把 `CRBM_LEARN_A` 置 1 即可。

## 5. 关键类与方法

| 成员 | 作用 |
| --- | --- |
| `CRBMConfig` | 超参数数据类（尺寸、σ、学习率、动量、渐近线等） |
| `CRBM.__init__` | 用 `N(0, 0.01)` 初始化 W、bv、bh（论文 Eq.14），`a` 初始化为 1 |
| `hidden_mean(v, add_noise)` | 计算隐层状态（Eq.2） |
| `visible_mean(h, add_noise)` | 计算可见层状态（Eq.3） |
| `reconstruct(v, n_gibbs)` | **无噪声**往返重构，用于评估与算重构误差 |
| `transform(v)` | 无噪声隐层输出，**逐层堆叠时的关键接口** |
| `fit(X, epochs, batch_size, patience)` | 小批量 CD-1 训练主循环 |
| `get_params()` / `from_params()` | 权重存取（与 npz 互转） |

## 6. 注意事项

1. `transform` 与 `hidden_mean(add_noise=False)` 等价 —— 堆叠时**必须用无噪声版本**，
   否则误差会逐层放大。
2. `fit` 内部每个 epoch 都会重新打乱样本顺序（`rng.permutation`），保证随机性。
3. `patience>0` 时会启用早停并回滚到最优快照；默认 `0` 表示严格按 epoch 数训练
   （论文的做法）。
4. 训练曲线记录在 `self.history`，由 step03 保存为 CSV 与图片。