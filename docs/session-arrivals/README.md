# 新会话到达率 × 池内接续：合成 workload 可行性研究

后续已选定 **SWE固定顺序＋匀速新会话＋完成后等待**，见
[当前压测客户端](../SESSION-ARRIVALS.md)。本页保留前期离线研究，下面的名义时间轴
不是当前HTTP客户端协议。

状态：**离线研究工具与候选流量骨架**。尚未选定或实现 token 合成策略，
未接入 HTTP 发压器，未跑模型性能。原 `swe-prefix-reuse run` 的闭环语义不变。

## 已迁移的统计方法

来自 Fletcher 的 `CubeLander/ruijie-workshop`，方法提交
`ddcec06550b531b2cde1027c187d317d21b8da94`，具体是
`quantification/prefix-traces/session_stats.py`、`prefix_stats.py`。
数据为 [UW TraceLab v0.0.2](https://github.com/uw-syfi/TraceLab/releases/tag/v0.0.2)，
Kan Zhu 等作者，**CC BY 4.0**。这里的统计、匿名相对时间/长度轨迹是派生材料，
不是原始文本；代码仍遵循本仓库 Apache-2.0。来源及实际原始压缩包摘要见
[source.json](source.json)，完整统计见 [statistics.json](statistics.json)。

- 内存中按 `(user, provider, session_id)` 分组，重复 round ID 直接失败。
- 单调用会话、零间隔和长尾全部保留；provider 分层，不混合成一个总体。
- 以完整 session 为抽样单位，保持调用数、时间和长度序列的联合关系。
- 只导出匿名顺序号、相对首输出时间、输入/已命中/未命中/输出长度。
- API 缓存记账满足 `input_total = prefix + newly_append`；后者不是本轮真正
  追加的文本量（其中可能包含未命中的旧历史）。
- 时间是**首输出到首输出**，不是 arrival，也不是服务完成后的 think time。
- 观测到的 session 起点/终点可能被截尾，不能宣称完整真实会话寿命。

本地全量复算665,453条、8,058个session。两provider各21项存续/调用数/间隔统计
与 workshop 已保存结果相符；没有复跑无关 TTL 仿真。

## 合成是否可行？可以，但必须显式建模边界

详见 [synthesis-audit.json](synthesis-audit.json)。长度上的追加条件定义为：

`next_input_total >= previous_input_total + previous_output_tokens`

这只是构造可行性，不证明原文相同，也不能反推出真实专家路由。

| 观测 | Claude | Codex |
|---|---:|---:|
| 相邻调用对 |300,126|357,269|
| 长度允许直接追加 |295,941（98.61%）|341,767（95.66%）|
| 不能直接追加的边界 |4,185（1.39%）|15,502（4.34%）|
| 首条调用已报告缓存命中 |3,120/5,319（58.66%）|2,670/2,739（97.48%）|
| 输入+输出超过262,144的调用 |92,871（30.41%）|1,316（0.37%）|
| 输出长度0的调用 |260|21|
| 输入长度0的调用 |246|21|

**不能按“整个 session 必须可追加”筛选**：会只留下Claude109,284/305,445次调用，
Codex35,359/360,008次调用，分别约35.8%和9.8%。这种筛选严重偏向短会话。
即便仅按整会话256K限制筛选，Claude也只保留127,994次调用，Codex347,053次。
不能把它叫原始分布的无偏缩样。

候选合成规则（还未采纳为正式 workload）：

1. 每个新到达创建独立session身份；首输入按记录长度合成，作为**进入本服务的
   冷会话**，并不声称这是生产原会话的真正起点。首条已有API命中也不伪造本地缓存。
2. 在可追加边界，保留实际已生成token IDs，只补足
   `next_input - previous_input - actual_output` 个输入token；输出budget跟随轨迹。
   输入素材可以使用固定公共代码/文本语料，不使用单一重复token，不声称语义复原。
3. 在不可追加边界，保留session生命周期，但开启新的前缀epoch，显式冷建新历史。
   这是保守的**合成重置策略**，不是观测证明了compaction。GDN状态也不能仅裁掉KV
   就假定能回滚到任意历史水位。若以后研究部分保留，必须单独给出可恢复状态契约。
4. API `prefix_tokens`只作为观测对照，不直接指定真实复用长度/命中/物理I/O。
   在新后端上重新测实际命中和搬运量，这是待测结果，不是输入强制结果。
5. 超context和零输入/零输出还需要明确策略；不静默截断、改成1或删除。应同时保留
   未修改的母样本和转换审计。优先研究Codex原尺度是建议，不是已经切换默认数据集。

为什么不只用时间分布：长度、输出budget、会话深度和时序都已经联合保存，
可以保留更多缓存压力特征。不过TraceLab不含完整文本/token IDs，无法复现语义、
分支谱系、真实模型路由或服务商的原缓存行为。

## 候选新会话流量骨架

离线 `plan` 明确区分**新sessions/s**和所有requests/s。
- 新session按Poisson（或显式constant）独立进入；不因服务器忙而降速。
- 分开随机流产生到达与session选择，同seed跨rate复用同一模板序列。
- 以源首输出相对offset作为**合成名义释放时间轴**，不是测得的真实arrival。
- 接续实际eligible时间为 `max(session_arrival + offset, predecessor_end)`。
  不把输出间隔再次完整加到执行结束之后；同session不并发。
- 冷启动空池，所有session尾部保留，不将时间窗末尾截尾误报为正常结束。
  30分钟示例不是稳态负载，更不意味着72天长尾应一直保留在HBM。
- 超安全规模直接报错，不抽稀、不把open-loop悄悄改成closed-loop。
- `preview`使用**人为固定服务时间**演练排队，独立记录新到达及未完成session；
  只用于协议验证，绝不可当成模型吞吐/TTFT。

已生成本地示例：两个provider各0.02/0.05/0.1new sessions/s，1800秒，seed20261002。
每组到达31/88/173个新session；名义窗口调用数Claude571/1316/2499，
Codex677/1736/3472。只代表这次冷启动样本，不拿 `rate × 全量平均轮数`
替代有限窗口实际流量。待token规则确定，再衔接已有严格HTTP/SSE客户端。

## 重现（纯CPU，不启动/访问推理服务）

```bash
pip install -e '.[test]'
# 原始压缩包不入Git，下载后以source.json中SHA256核对身份
mkdir -p prepared/tracelab
curl -fL https://github.com/uw-syfi/TraceLab/releases/download/v0.0.2/syfi_coding_trace.jsonl.gz \
  -o prepared/tracelab/source.jsonl.gz
python scripts/session_workload.py profile --source prepared/tracelab/source.jsonl.gz \
  --output prepared/tracelab/profile.json.gz
python scripts/session_workload.py audit --profile prepared/tracelab/profile.json.gz \
  --output prepared/tracelab/audit.json
python scripts/session_workload.py plan --profile prepared/tracelab/profile.json.gz \
  --provider codex --rate 0.05 --duration 1800 --seed 20261002 \
  --output prepared/tracelab/plan.json
python scripts/session_workload.py preview --plan prepared/tracelab/plan.json \
  --service-seconds 1 --workers 32 --output prepared/tracelab/preview.json
pytest -q
```

输出拒绝覆盖。模型端真正需要的token合成、重置、超限与零budget策略仍待讨论；
本轮没有更改PD、KV pipeline、服务器环境，也没有向公共仓库发布。
