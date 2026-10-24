我建议 **Iteration 61 不再做“下一种学习机制”**，而是把 Iteration 5 暴露出的唯一主失败——**记忆经过时间和后续学习后衰减过快**——一次性拆清楚。

Iteration 5 已经把项目推进到约 **84/100**。V61 的目标应该非常明确：

$$
\boxed{\text{解释并修复 }0/4\text{ interference half-retention failure}}
$$

而不是同时碰 cue formation、MiniFly compression、Rachad higher-order learning、真实 odor physiology 等问题。

## 1. 为什么现在应该盯住 retention

Iteration 5 的四个旧记忆：

| Pair | Acquisition | 相同时间纯休息 | 经其他学习后 |
| ---- | ----------: | ------: | -----: |
| 0    |       -9.97 |   -2.85 |  -0.70 |
| 1    |       -9.70 |   -3.70 |  -2.10 |
| 2    |       -9.02 |   -2.62 |  -2.27 |
| 3    |       -9.44 |   -3.68 |  -3.05 |

这里其实已经透露了一个非常重要的信息。

纯休息后只剩 acquisition magnitude 的大约：

$$
29\%,\,38\%,\,29\%,\,39\%.
$$

也就是说，**即使完全没有新学习，四个 memory 都已经低于预注册的 50% retention gate。**

而进一步的新学习造成的额外损失：

* Pair 0：很大；
* Pair 1：明显；
* Pair 2：很小；
* Pair 3：较小。

所以目前的主问题不是单一“catastrophic interference”。

更像：

$$
\boxed{
\text{基础时间衰减太强}
+
\text{部分 cue 上再叠加额外 interference}
}
$$

这两件事必须分开，否则下一轮很容易又错误地发明 anti-interference mechanism。

---

# 2. V61 第一部分应该完全不改模型

我会叫它：

$$
\boxed{\text{Retention Causal Decomposition}}
$$

从**完全相同的 acquisition checkpoint**分叉。

至少跑以下条件，所有条件保持完全相同的 elapsed time 和 event count：

### A. REST

什么都不学，只经过相同时间。

已经有，但重新进入统一 panel。

### B. SENSORY-ONLY

把那 8 个 intervening pair 的输入全部呈现一遍，但：

* teaching payload = 0；
* plasticity = 0。

这告诉我们纯 sensory/adaptation activity 会不会伤旧记忆。

### C. TEACHING/PATHWAY CONTROL

呈现同样的 stream，让 teaching circuitry 活动，但禁止实际长期写入。

这样可以区分：

$$
\text{teacher/controller state interference}
$$

和：

$$
\text{actual plastic-weight interference}.
$$

### D. FULL LEARNING

Iteration 5 原来的完整 intervening learning。

于是可以得到类似：

$$
\begin{aligned}
L_{\rm time} &= A_{\rm acquisition}-A_{\rm rest}\\
L_{\rm sensory} &= A_{\rm rest}-A_{\rm sensory}\\
L_{\rm controller} &= A_{\rm sensory}-A_{\rm teaching-control}\\
L_{\rm write} &= A_{\rm teaching-control}-A_{\rm full}.
\end{aligned}
$$

不必机械地把它们当完全加性的物理量，但会极大缩小机制空间。

---

# 3. 不要只看 MBON 输出——直接看 memory state 是怎么坏的

这是我认为 V61 最关键的新 diagnostic。

现在我们有真正的固定 mutable state：

$$
4\times5177=20,708
$$

个值。

所以对于每一个旧 memory，都可以保存 acquisition 后的 memory-state vector：

$$
w_0.
$$

经过 rest / intervening learning 后得到：

$$
w_t.
$$

把它分解成：

$$
w_t=\alpha w_0+w_\perp.
$$

其中

$$
\alpha=
\frac{\langle w_t,w_0\rangle}
{\|w_0\|^2}.
$$

这样立即可以区分两种完全不同的 failure。

### 情况一：只是 fading

如果：

$$
\alpha\ll1,\qquad w_\perp\approx0,
$$

说明记忆方向还对，只是 amplitude 缩小。

那问题主要是：

$$
\boxed{\text{persistence / consolidation}}
$$

### 情况二：被新记忆改写

如果：

$$
w_\perp
$$

明显增加，

说明新学习在把原 memory vector **旋转**到别的方向。

那才是真正：

$$
\boxed{\text{interference / overwrite}}
$$

过去几十轮我们很少能够这么直接观察内部 memory geometry。

现在能了，就应该利用，而不是只看最终 accuracy。

---

# 4. 同时检查 fast/slow 两个现有 trace 到底各自干了什么

这里尤其不要再“发明 fast/slow”。

**我们已经有 fast/slow state。**

问题应该是：

> Iteration 5 的六次 acquisition，最终到底有多少信息进入了 slow component？

例如分别记录：

$$
M_{\rm fast},\qquad M_{\rm slow}
$$

在：

* acquisition；
* matched rest；
* interference

三个时间点的 memory projection。

有几种可能。

### A. acquisition 几乎全部进入 fast trace

那 Iteration 5 的 integration interface 可能没有正确利用 Nature-derived slow state。

这种情况下，不应该新加 memory bank。

应该修：

$$
\boxed{\text{existing fast→persistent-state engagement}}
$$

### B. slow trace 本来很强，但也快速下降

那是 current causal kernel 本身的问题。

### C. slow trace保留良好，但 MBON expression 消失

那根本不是 memory loss，而是：

$$
\boxed{\text{readout/expression problem}}
$$

这和 Iteration 5 的 24h saturation failure 已经有一点呼应。

这三种情况下一步完全不同。

所以 V61 必须先测。

---

# 5. 再把 interference 和 KC overlap 对起来

Iteration 5 已经有真实 KC overlap matrix。

而四个 pair 的额外 interference 大小差别很大：

$$
2.15,\quad1.60,\quad0.36,\quad0.64\ {\rm Hz}
$$

左右。

这给我们一个现成问题：

> **受到更多 later-cue KC overlap 的旧 memory，是不是损伤更严重？**

不要只算简单 Jaccard。

最好定义：

$$
I_i
=
\sum_{j\in\text{later cues}}
\sum_{k}
a_{ik}a_{jk}
\,r_k
$$

其中 \(r_k\) 可以包括：

* KC activity；
* teaching-write eligibility；
* KC→MBON readout strength。

然后看：

$$
I_i
$$

和：

$$
\text{extra loss beyond matched rest}
$$

是否相关。

如果相关，就真正建立了：

$$
\boxed{
\text{shared physical memory support}
\rightarrow
\text{interference}
}
$$

的机制证据。

如果不相关，我们就不要再把 overlap 当故事。

---

# 6. V61 后半部分最多允许一个 rescue mechanism

这里我要非常严格。

**不要同时跑 replay、lower ETA、third trace、new bank、homeostasis、metaplasticity……**

这些都是过去版本容易进入 swamp 的方式。

先根据上面的 diagnosis 选一个。

---

## Branch A：如果主要是 passive decay

也就是：

$$
REST\approx FULL
$$

而 memory vector 主要是 scalar shrinkage。

那么 V61 的唯一 intervention 应该针对：

$$
\boxed{\text{existing slow-state engagement / consolidation}}
$$

而不是“加一个 slow memory”。

因为 current model 已经有 two traces。

可以测试一个非常有限的问题：

> 六次一致训练是否应该使现有 slow component 获得更高权重？

必须：

* 不增加 state dimension；
* 不增加 cue-specific slots；
* 保留 revision；
* acquisition 不得明显变差；
* 不重新搜索几十个 decay constants。

这是**修正当前已有两状态模型**，不是把过去失败的 fast/slow architecture 换名字重新引进。

---

## Branch B：如果主要是 write interference

如果：

$$
FULL\ll REST
$$

而且 memory vector 出现大量 orthogonal rotation，并与 KC overlap 明显相关，

那么唯一 intervention 应针对：

$$
\boxed{\text{local write specificity}}
$$

也就是：

> 新事件为什么修改了本来属于旧 association 的 state？

这里再检查现有 teaching routing / activity gate 是否过宽。

不要立即加 replay。

因为如果问题是“错误的连接被写了”，replay只是不断重新修补错误。

---

## Branch C：如果 memory state 没坏，只有 readout坏

那反而最好。

如果内部 slow-state projection还在，但 output memory contrast掉了：

> 不要动 learning mechanism。

直接定位：

$$
\boxed{\text{readout saturation / gain / normalization}}
$$

Iteration 5 已经出现一个 24h pair 两边都顶在 19.96 Hz ceiling。

如果 sequential failure 很大部分也是 expression problem，那么此前所谓“retention failure”会被重新定性。

但 readout calibration必须在独立 development conditions 上做，不能为了救这4个pair事后调 gain。

---

# 7. V61 的 acceptance criterion 我会保持简单

不要创造十几个新 gates。

### Primary

四个原始 memory 中：

$$
\boxed{\ge3/4}
$$

经过 intervening learning 后仍满足：

$$
\frac{|M_{\rm after}|}{|M_{\rm acquired}|}\ge0.5
$$

并保持正确方向。

### Mandatory non-regression

* acquisition ≥ 7/8，最好仍 8/8；
* 24h persistence ≥ 7/8；
* revision 必须仍成功；
* fixed state size 完全不变；
* checkpoint/resume exact；
* 不新增 cue ID / slot / replay store。

### Mechanistic requirement

还必须能解释 improvement 来自：

* 更少 passive decay；
* 更少 state rotation；
* 或更好 readout expression。

不能只是：

> “最终数字变好了。”

---

# 8. 如果 V61 成功，评分会发生真正跃迁

现在我给 Iteration 5：

$$
84/100.
$$

如果 V61做到：

* 3/4 或 4/4 half-retention；
* acquisition保住；
* revision保住；
* fixed state保住；

那么我会直接提高到：

$$
\boxed{90\text{–}92/100}.
$$

因为那意味着基本核心链条：

$$
\text{learn}
\rightarrow
\text{retain}
\rightarrow
\text{learn something else}
\rightarrow
\text{retain old}
\rightarrow
\text{revise old}
$$

第一次真正闭环。

剩下的主要是：

* autonomous cue formation；
* autonomous behavioral readout；
* self-generated teaching；
* scaling/sample efficiency。

也就是我们约定暂时不计后两项以后，已经非常接近 core success。

---

# 9. 如果 V61 失败，也应该停止这一分支继续堆版本

这点很重要。

如果 decomposition 显示：

* passive decay不是主因；
* overlap也解释不了；
* state vector没有简单 failure；
* 一个最小 targeted rescue也失败；

那么不要：

$$
V62,V63,V64...
$$

继续小修。

那时才值得让 **Ultra 做一次 architecture-level reset/audit**：

> 为什么已经 integrated 的 V50-like memory + biological teaching controller 不能同时满足 retention 和 revision？

也就是我们之前讨论的真正适合 Ultra 的场景。

---

## 因此我建议把 Iteration 61 定义成一句话

> **Freeze Iteration 5. Decompose old-memory loss into passive decay, sensory/controller effects, actual write interference, and readout expression using identical checkpoints; then permit exactly one diagnosis-driven rescue without increasing persistent state.**

这会比“再设计一个 V61 mechanism”有效得多。

而且计算量应该并不大——Iteration 5 整个 panel 本来只有秒级。真正需要 Codex **Extra High** 的是设计和审计；实际运行用 High 足够。

我认为这是目前最可能用**一轮而不是十轮**判断我们能不能从 84 分跨到 90 分的实验。
