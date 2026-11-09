# 4 核 CPU 上的 SeedBS 精确加速（2026-11-10）

服务器的 `cpu.max=400000 100000` 表示 4 核计算时间配额，内存限额为 8 GiB。
16 个进程不会获得 16 核算力。本次只优化 SeedBS 的 KS 内核，CPU 运行入口和结果目录不变。

## 计算与一致性

原 NumPy 内核为每批重采样构造 `(重复数, 区间长度, 区间长度)` 比较数组，
并创建多份同规模整数中间数组。新的可选 Numba 内核排序后只枚举不同的阈值，
以阈值并列组的末端位置计算总 CDF 计数，再用整数累计扫描切点。
观测序列仍返回每个切点的统计量，保持最先达到最大值的 argmax 规则。

bootstrap 只需要判断最大整数统计量是否 `>= observed_max`，因此新内核一旦找到
符合条件的切点就结束该次检查；该判定仍然精确，不是近似或提前停止 Monte Carlo。
最坏时间复杂度仍为二次，但避免大比较张量和多遍内存扫描，并减少并列阈值的重复计算。

不改变 T=999、T′=1999、B/ζ 网格、随机种子、抽块顺序、bootstrap 分块大小或校准规则。
新旧实现的整数统计量、p 值、切点、选择结果和随机数最终状态通过对照测试。
预计算中的 `seconds` 是运行时间，当然会不同。已有 SeedBS `.pkl` 缓存名称和结构不变，
新旧内核可以复用同一个缓存。内核未启用 fastmath 或内部多线程。

## 本机基准

Windows、Python 3.12.14、NumPy 2.5.4、Numba 0.68.1，单进程。
两个合成序列的有效长度均为 496：null 为均匀随机序列，splice 的前半段替换为 Beta(5,1)。
T=999、T′=1999，单次完整预计算；首次编译/加载预热 0.645 秒未计入下表。
部分测量期间本机另有测试运行，结果只作为本机量级参考，不是服务器性能承诺。

| 合成输入 | B | NumPy 秒 | 加速版秒 | 单篇预计算加速 |
|---|---:|---:|---:|---:|
| null | 6 | 48.91 | 4.55 | 10.75× |
| null | 20 | 45.63 | 3.93 | 11.61× |
| splice | 6 | 53.37 | 3.78 | 14.12× |
| splice | 20 | 43.90 | 2.08 | 21.08× |

四组预计算的计算产物除 `seconds` 外全部一致。以上倍数只适用于相应 SeedBS 预计算，
不适用于 HMM 拟合、整个 CPU 流程或已命中缓存的计算。
Numba 首次编译、服务器 Python/NumPy/Numba 版本和数据构成均会影响实测速度。

## 服务器使用

在已有环境中通过 Git 更新代码、安装可选加速依赖。不要重新安装整套 GPU 环境。

```bash
cd /root/blockdata/HMM
source .venv-q3/bin/activate
source scripts/env.sh
git pull --ff-only
python -m pip install -r requirements/cpu-fast.txt
```

默认 `OPT_HMM_SEEDBS_BACKEND=auto`：Numba 可导入时启用加速，否则沿用 NumPy。
设为 `numba` 可强制使用加速，缺依赖时明确报错；设为 `numpy` 可回到参考实现。
后续启动 CPU 流程时，4 核容器使用 4 个 worker：

```bash
OPT_HMM_SEEDBS_BACKEND=numba WORKERS=4 \
PROGRESS=runs/l2q1-progress.json WEIGHTS=runs/l2q1-weights.json \
bash scripts/l2_cpu.sh l2q1
```

正在运行的旧 fit 不会因更新源码或安装 Numba 就切换内核。不要同时启动第二个同名 CPU 任务。
如果中断尚未完成的 fit 再启动，脚本会复用已落盘的 SeedBS 缓存，但 HMM 拟合仍按整个阶段
重算；本次没有增加 fit 内部检查点。若当前 fit 已接近完成，可等完成后再用于后续启动的阶段。

## 复测

建议在当前重计算结束后测试，避免争抢仅有的 4 核。默认使用合成输入，写项目 `tmp/`，
不读取正式 test 来选参数、不写正式结果目录；即使全部缓存已存在，也会真正计算两种内核。

```bash
python scripts/benchmark_seedbs.py --B 6 20
```

也可以只读训练侧第一篇文章进行对照，保持正式 T、T′；不加载模型：

```bash
python scripts/benchmark_seedbs.py --input runs/l2q1-train-pos --B 20
```

脚本先单独预热编译，分别记录 wall/CPU 秒数，并核对计算产物完全一致，默认报告写入
`tmp/sbbench.json`。`--repeat 3` 可重复测量并报告中位数，但会增加基准耗时。
正确性测试包含长序列、并列值、180/181 整数类型边界、bootstrap 跨块 RNG 状态、
完整预计算、双进程执行、缓存跨内核复用及无 Numba 回退。
