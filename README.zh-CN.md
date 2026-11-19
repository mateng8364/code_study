# Kimi K3.1 Pro：A3 双机 1P1D 的 256 专家裁剪实验

## 1. 目标与状态

本目录提供离线裁剪脚本和实验流程。目标是将每个 MoE 层的 **896 个 routed experts 保留为 256 个**，保留 top-16 路由和 2 个 shared experts，准备 A3 双机、1P1D 的容量、精度和性能实验。专家编号为 0–895，895 是最大编号，不是专家总数。

**256 是待验证的容量候选，尚不是精度达标或单机装载成功的结论。** 当前仅有 CPU 合成 checkpoint 测试；没有真实权重加载、NPU 显存或 GSM8K/GPQA 结果。精度下降容忍度需要在实验前确定。

参考来源：

- [参考方案：Kimi K3 / 256 experts / W4A8 MXFP](https://github.com/CuiZ2007/vllm-ascend/blob/test/kimi-k3-pruned-experts-accuracy/tools/kimi_k3/README.zh-CN.md)。
- [参考裁剪脚本](https://github.com/CuiZ2007/vllm-ascend/blob/test/kimi-k3-pruned-experts-accuracy/tools/kimi_k3/prune_experts.py)。
- [本仓库 Kimi K3.1 Pro 部署指南](../../docs/source/tutorials/models/Kimi-K3.1-Pro.md)。其中的已验证参考配置是八机 4P4D，不能视为双机裁剪方案的验证结果。
- [本仓库 A3 W4A8 加载器的存储格式说明](../../vllm_ascend/quantization/methods/w4a8/w4a8.py)。

本脚本参考了原方案的按字节复制、router 同步裁剪、SHA-256 校验和完成标记思路，重新实现 A3 格式检查及逐层选择。原方案的 A5/MXFP、256 专家与 GSM8K 历史结果不作为本实验的格式依据或精度基线。

## 2. 权重准备和容量

在有足够磁盘空间的 Linux 服务器或存储节点上准备完整 checkpoint，无须先下载到 Windows 开发电脑。优先检查部署指南列出的 A3 `Kimi-K3.1-Pro-w4a8` 权重；固定下载 revision、来源、量化工具版本与参数。

本脚本不做量化，也不把官方 MXFP4 或 W4A8_MXFP 转为 A3 W4A8。若只有 BF16 原始权重，需要另行完成经过验证的 A3 量化流程。若从现成量化权重裁剪，则复制其已有 INT4 数据和量化参数，不解包、不重新量化。

当前支持的输入契约如下；真实权重不满足时，脚本会拒绝写入，不能仅改标签绕过检查。

| 项目 | 当前支持范围 |
| --- | --- |
| 配置 | `config.json` 内 `text_config`；93 层，layer 0 dense，layer 1–92 MoE，896 routed、top-16、2 shared |
| 专家命名 | `language_model.model.layers.L.block_sparse_moe.experts.E.w1/w2/w3.*`；每个专家单独存储 |
| 权重格式 | ModelSlim per-channel `W4A8_DYNAMIC`；`I8` 容器按输出维打包两个 INT4 |
| 专家参数 | 每个 w1/w2/w3 都有 `weight`、`weight_scale`、`weight_offset`、`scale_bias`；后 3 项为 F32 |
| Router | BF16 或 F32 的 `gate.weight`、`gate.e_score_correction_bias`；沿专家维同步选择 |
| 量化描述 | 顶层以完整 tensor 名为 key 的 `quant_model_description.json` |
| 索引 | `quant_model_weights.safetensors.index.json` 或 `model.safetensors.index.json`，只允许存在一种 |
| 当前不支持 | MXFP、compressed-tensors、融合存储全部专家的 tensor、多组 grouped routing、其他打包布局 |

非 routed-expert 张量原样保留，包括 attention、embedding、shared experts、视觉模块和 latent 投影。**不会自动把这些权重转换为 BF16**；必须核对源权重是否符合“其余模型权重 BF16”的实验约束。Router、量化 scale 等辅助参数允许 F32，不能一概转换为 BF16。

按 92 个 MoE 层、latent dim 3584、intermediate size 3072 计算，256 专家的纯 INT4 矩阵约为：

```text
92 × 256 × 3 × 3584 × 3072 × 0.5 / 2^30 = 362.25 GiB / 模型副本
```

这不含量化参数、BF16 权重、KV cache、运行时 workspace、通信 buffer 和草稿模型。实际文件 tensor 字节数以 `plan` 输出为准；显存以目标并行配置加载后每个 device 的峰值为准。磁盘还需要容纳完整原始权重、裁剪输出、报告和额外余量。

PD 分离通常要求 P、D 各自加载完整的目标模型副本，各副本内部再做 TP/EP 分片。不能将两台机器的总显存相加后据此判断单个 P 或 D 副本是否能装下。

## 3. 专家选择策略

脚本支持两种输入：

1. 不传 `--expert-map`：每层按 `floor(j × 896 / 256)` 均匀保留，`j=0..255`。这是容量对照组，manifest 标记 `uniform_capacity_only`，不声称保精度。
2. 传 `--expert-map`：每层提供独立的 256 个原始专家 ID，列表顺序决定裁剪后的新 ID。可接入基于校准集得到的选择结果；脚本只执行并检查映射，不自动估计重要性。

逐层清单 JSON 的结构如下，实际文件必须完整覆盖 `"1"` 到 `"92"`，每层恰好 256 个互不重复的 0–895 整数；以下只是结构示意，不可直接执行：

```text
{
  "1": [该层保留的 256 个原始专家 ID，按新编号顺序排列],
  ...,
  "92": [该层保留的 256 个原始专家 ID，按新编号顺序排列]
}
```

精度候选建议使用独立校准集，统计逐层路由命中、路由权重及专家贡献，结合消融验证后选取；简单按调用频率排序不能保证精度。不得使用 GSM8K/GPQA 的最终评测样本来选择专家。保留 calibration 数据版本、选择方法和映射文件 SHA-256。若原配置使用多组路由，需要另外设计组内均衡选择及配置适配，当前脚本直接拒绝。允许 `use_grouped_topk=true` 且 `num_expert_group=topk_group=1` 的单组情况，保留原始路由配置；它不涉及跨组选择。

每层执行相同的同步变换：

- 保留选中专家的 3 组矩阵及全部 scale/offset/scale_bias，重编号为 0–255。
- 按完全相同的顺序选择 router weight 行及 correction bias 元素。
- 更新 `text_config.num_experts`，其余 config 保持原值。
- 重写量化描述中的专家 key 和 safetensors 索引，重新计算 `total_size`。

## 4. 执行命令

在仓库根目录执行，使用 Python 3.10+，无需安装 torch、vLLM 或 safetensors。源目录必须是完整、已落盘、停止写入的 checkpoint，包含 tokenizer、processor、config 和量化描述。源树中的软链接需预先物化；不直接使用含符号链接的下载缓存 snapshot。

先查看真实权重格式，只读取配置、索引及 safetensors header，不读取大型 tensor 数据：

```bash
python -m tools.kimi_k31.prune_experts \
  --mode inspect \
  --src /data/models/Kimi-K3.1-Pro-w4a8 \
  --report /data/reports/k31-source-inventory.json
```

`inspect` 给出配置、量化标签统计、dtype 字节分布和第 1 层第 0 个专家及 router 的 shape。它是清点工具，不代表完整兼容性校验。报告父目录需预先创建。

再生成容量对照组的裁剪计划；不生成权重副本：

```bash
python -m tools.kimi_k31.prune_experts \
  --mode plan \
  --src /data/models/Kimi-K3.1-Pro-w4a8 \
  --keep-experts 256 \
  --report /data/reports/k31-e256-uniform-plan.json
```

真实写入到一个**尚不存在**的新目录：

```bash
python -m tools.kimi_k31.prune_experts \
  --mode run \
  --src /data/models/Kimi-K3.1-Pro-w4a8 \
  --dst /data/models/Kimi-K3.1-Pro-w4a8-e256-uniform \
  --keep-experts 256 \
  --report /data/reports/k31-e256-uniform-run-plan.json
```

若使用精度候选，在 `plan` 和 `run` 两条命令中都添加相同的参数：

```text
--expert-map /data/reports/k31-e256-calibrated-map.json
```

同时为候选使用不同的目标目录和 report 文件名。`run` 会重新扫描源文件，不直接消费旧 plan；因此两次必须使用相同的源 revision 和选择文件。report 不覆盖已有文件，且不能位于源或目标目录内。

写入过程按 shard 顺序执行，按 8 MiB 块流式复制，无需把完整权重装入内存；router 按行复制。每个输出 shard 关闭后重新读取并核对 SHA-256，索引只引用保留 tensor 的 shard。原始权重不修改。

第一版有意不支持断点续写：中断会留下 `PRUNING_INCOMPLETE.json`，这类目录不得部署。重试时选择另一个不存在的目标目录，确认失败目录用途后自行清理。脚本不会覆盖已有目录或自动删除模型文件。

完成后再次验证，或复制到部署服务器后验证：

```bash
python -m tools.kimi_k31.prune_experts \
  --mode verify \
  --dst /data/models/Kimi-K3.1-Pro-w4a8-e256-uniform
```

`PRUNING_COMPLETE.json` 保存逐层选择、源索引与辅助文件 hash、源 shard header hash/文件状态、脚本 hash 和输出文件 SHA-256。源大文件的 payload 没有在 plan 阶段完整预哈希，操作期间必须保持源只读；源 revision 仍应在实验记录中单独固定。verify 验证输出文件与 manifest 一致，不等于模型加载或数学精度验证。

## 5. 部署时权重放在哪里

无需上传 GitHub，也不要求上传公共模型平台。将完整裁剪输出目录放到两台 A3 可读的本地磁盘，或挂载同一份只读共享存储；推荐两台的容器内路径统一：

```text
P 服务器：/data/models/Kimi-K3.1-Pro-w4a8-e256-uniform
D 服务器：/data/models/Kimi-K3.1-Pro-w4a8-e256-uniform
容器挂载：宿主机 /data/models → 容器 /models，建议只读
服务加载：/models/Kimi-K3.1-Pro-w4a8-e256-uniform
```

复制整个目录，包括 tokenizer、processor、config、quant description、index、全部输出 shard 和完成 manifest。两端均运行 `verify`，并比较完成 manifest 的 hash。TP/EP 由运行时分配，不手工把权重文件分一半给 P、一半给 D。

本阶段不直接给出双机启动参数：需确认 nightly 的镜像 digest、vLLM/vLLM-Ascend commit、CANN/torch-npu 版本，以及真实的每副本显存和 TP/EP 约束，再修改仓库 PD 模板。目标模型裁剪不会自动裁剪或适配 DSpark 草稿模型；草稿权重、并行参数与接受率测试需要单独验证。

## 6. 验收顺序

| 阶段 | 检查内容 | 通过条件 |
| --- | --- | --- |
| 源文件 | `inspect` 和 `plan`、权重来源、量化格式、非专家精度 | 匹配真实 A3 checkpoint；输入契约完整成立 |
| 文件完整性 | 写后读回、两端 `verify` | 无缺失/额外文件、SHA-256 和索引布局正常 |
| 单副本加载 | 目标版本加载完整副本，记录每 device 显存和 scale_bias 分片兼容性 | 无 missing/unexpected weights、shape 错误、OOM；最小生成无 NaN |
| 1P1D 功能 | 跨机 KV 传输、请求端到端完成 | 多种短请求输出正常，连接/缓存释放正常 |
| GSM8K 精度 | 首轮暂按固定数据版本、原始顺序、不打乱的前 500 条执行 | 报告正确数/500、准确率、失败数、耗时及评测配置 |
| GPQA 探索 | 在 PD 分离服务上尝试，固定 subset 和样本数 | 记录准确率、完成/失败/超时数、实际完成样本数及总耗时 |
| 性能探索 | 尝试固定输入/输出长度与固定接受率，固定并发、warmup、请求数 | 观察 ITL 或 TPOT 效果，注明统计口径；辅助记录 TTFT、吞吐及峰值显存 |

精度基线优先使用同一量化版本的未裁剪 896 专家模型；如果需要更多机器运行，就在足够资源上独立测量。只有 BF16 基线时，观测差异会同时混入量化与裁剪影响，必须说明。不要把参考 README 中 K3/256 专家的历史 GSM8K 数字当作 K3.1 的基线。

需求原话为“精度用例已实践，用 GSM8K 500/1319 条；精度用例在 PD 分离上尝试 GPQA，耗时和数据情况；性能方案可以尝试定长、定采信率，看 ITL 或 TPOT，看效果”。这里将 GSM8K 首轮暂定为前 500 条，与当前用户理解一致；原话中的“500/1319”本身并不能证明历史用例一定按顺序取前 500 条，需要对照既有评测命令或样本 ID 确认。1319 条全量作为后续可选复核，本阶段不强制执行。

前 500 条应在 shuffle、并发分发等步骤之前固定，保留数据版本和样本 ID 清单；不能将随机 500 条与前 500 条的结果直接对比。报告计划样本数和实际完成数，失败请求单独统计，不静默从分母中删除。GPQA 的具体 subset 和样本数仍待确定，不把“尝试”扩大为已具备全量精度验收结果。

GSM8K 与 GPQA 固定具体 subset、样本数、seed、prompt、答案提取、thinking/token budget 等。均匀选择和逐层校准选择都可以测精度，并在报告中注明选择策略。若 256 不达标，优先改进选择/恢复训练，或在容量允许时增加专家数，不能仅通过降低基线要求宣称裁剪成功。

“固定接受率”属于性能控制实验：记录实现方式、请求接受率与实测接受率，不将人工控制接受率的输出用于精度结论。精度使用真实解码路径。ITL 与 TPOT 在流式和推测解码场景下可能口径不同，需固定客户端/服务端时间戳定义，报告分位数，避免仅比较平均值。

nightly 暂定 2026-11-11，待确认具体任务链接和产物；仅日期不足以复现。若其模型规模、机器数或并行策略不同，只能作参考，不能直接据此断言性能提升。

## 7. 本地测试

```bash
python -m unittest discover -s tests/tools/kimi_k31 -v
```

测试使用小尺寸但遵循相同存储契约的合成 checkpoint，逐字节核对非均匀逐层映射、router 重排、量化参数与非专家 tensor，并覆盖格式拒绝、索引错误、源文件变化、文件损坏和目录保护。它不能替代真实 checkpoint、目标 nightly 和 A3 硬件上的验收。

当前 16 项测试属于裁剪脚本的 CPU 单元测试，不调用模型推理，也不包含 GSM8K、GPQA 或 PD 性能压测脚本。合成数据使用 2 个 MoE 层、每层 32 个专家裁至 16 个，以及很小的矩阵；不同专家和 router 行填充不同字节，以检查保留内容与顺序。另检查单组路由配置保留及多组路由拒绝。这些字节用于文件变换校验，不代表有实际语义能力的模型参数。独立 safetensors 读取检查在安装该可选依赖时执行，否则跳过。
