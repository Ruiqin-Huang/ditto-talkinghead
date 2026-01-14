# DITTO 论文分析与使用指南

## 1. 论文概述

### 1.1 解决的问题

DITTO 主要解决音频驱动说话人头生成中的两个关键问题：

#### 问题 1：缺乏精细化的运动控制
- **现有方法缺陷**：传统扩散模型无法对生成结果进行细粒度控制，用户无法直接调整：
  - 面部表情（facial movements）
  - 基础情绪（basic emotions）
  - 头部旋转（head rotations）
  - 眨眼和凝视（blinking and gaze）
- **影响**：生成质量随机性较大，难以获得期望效果，需要反复重新生成

#### 问题 2：推理速度慢
- **现状**：大多数扩散方法难以在单 GPU 上实现实时推理（< 30 FPS）
- **需求**：交互式场景（AI 助手、实时视频流）需要低延迟推理
- **瓶颈**：传统方法在通用 VAE 空间中操作，计算开销大

### 1.2 现有方案及其缺陷

#### 早期方法：GAN-based
- **代表方法**：基于生成对抗网络的说话人头生成
- **优点**：能生成逼真纹理，唇形同步准确
- **缺点**：难以捕获真实表情和头部运动

#### 近期方法：Diffusion-based（如 EMO）
- **代表方法**：EMO、EchoMimic、Hallo、Loopy
- **优点**：生成结果更生动、真实
- **缺点**：
  - 在通用 VAE 空间中操作，冗余且隐式
  - 增加扩散模型学习复杂度
  - 推理速度慢（无法实时）

#### VASA-1
- **创新点**：在 motion-appearance 解耦空间中训练 DiT，实现两阶段生成
- **优点**：显著降低推理时间
- **缺点**：
  - 源代码未公开
  - 使用隐式运动表示，不支持控制和调整

### 1.3 作者提出的方法

#### 核心思想

DITTO 提出基于**运动空间（Motion Space）**的扩散模型，实现**可控的实时**说话人头生成。

#### 关键技术

**1. 运动空间构建（Motion Space）**

基于 LivePortrait [18] 构建运动空间，将单帧图像 `I` 通过运动提取器 `M` 分解为：

```
I → M → {
    c: 规范关键点（canonical keypoints）- 身份特征
    m: 运动表示（motion representation）- 与身份无关
        m = {δ, R, t}
        - δ: 表情变形（expression deformations）∈ R^(K×3)
        - R: 头部姿态旋转矩阵 ∈ R^(3×3)
        - t: 平移向量 ∈ R^3
}
```

**优势**：
- **身份无关**：运动表示与身份解耦，可以跨身份使用
- **显式对应**：运动表示与面部属性有明确对应关系
- **紧凑高效**：相比 VAE 空间更紧凑，降低计算复杂度

**2. 条件扩散 Transformer（Conditional DiT）**

使用条件扩散 Transformer 进行音频到运动生成：

**输入条件**：
- `a`: 音频特征（通过 HuBERT 提取）
- `cref`: 规范关键点（身份特征）
- `s`: 情绪标签（emotion label）
- `e`: 眼部状态（eye state）
- `mref`: 参考初始运动（reference initial motion）

**架构特点**：
- **ECS（External Conditioning Sequence）分支**：处理外部条件信号
  - 输入：`cref`, `s`, `a`, `e`
  - 处理：MLP → (Self-Att + MLP) × 2
  - 输出：作为 Cross-Attention 的键值

- **ICS（Internal Conditioning Sequence）分支**：处理内部条件序列
  - 输入：`mref`（重复）+ `noise`
  - 处理：MLP → (Self-Att + FiLM + Cross-Att + FiLM + MLP + FiLM) × 8
  - 时间步条件：通过 FiLM 层注入

**3. 训练策略**

- **水平翻转**：增强训练数据，平衡左右两侧的音频-运动对应关系
- **自适应损失权重**：根据不同运动组件的运动模式差异，动态调整损失权重
- **验证指标**：使用唇形同步分数作为验证指标，解决损失曲线不能准确反映生成质量的问题

**4. 精细化运动控制**

建立运动表示与面部语义的直接映射：
- **区域控制**：可以限制运动生成到特定局部面部区域
- **幅度控制**：可以对变形值施加约束，防止不自然表情
- **凝视校正**：通过回归 `δe = K(Re)` 建立凝视变化与头部姿态变化的映射

### 1.4 核心公式

#### 运动表示公式

给定参考图像和生成运动，计算隐式 3D 关键点：

```
# 参考关键点
x_ref = cref * Rref + δref + tref    (1)

# 生成关键点
x_hat = cref * R_hat + δ_hat + t_hat  (2)

# 最终图像合成
I_hat = G(f_ref, x_ref, x_hat)        (3)
```

其中：
- `cref`: 规范关键点（身份特征）
- `R`, `δ`, `t`: 旋转、变形、平移
- `G`: 面部渲染器
- `f_ref`: 参考外观特征

#### 损失函数

**主要损失（Denoising Loss）**：
```
Ld = E_{t~U[1,T], m0, C} ||m0 - T(mt, C, t)||²₂  (4)
```

其中：
- `T`: 扩散 Transformer
- `mt`: 添加了 t 步噪声的运动数据
- `C`: 条件信号

**时间稳定性损失**：
```
Lt = ||m̂' - m'||²₂ + ||m̂'' - m''||²₂  (5)
```

**初始运动损失**：
```
L_ini = ||m̂[0] - m_ref||²₂
```

**总损失**：
```
L = Ld + Lt + L_ini  (6)
```

### 1.5 为什么能解决问题？

#### 解决控制问题

1. **显式运动表示**：
   - 使用基于关键点的显式运动表示
   - 建立运动表示与面部语义的映射（如第 34 维控制右眼开合，第 58 维控制嘴部张开）
   - 支持直接修改运动参数

2. **多样化条件信号**：
   - 情绪标签 `s`：直接控制情绪
   - 眼部状态 `e`：控制眨眼和凝视
   - 规范关键点 `cref`：适配目标身份
   - 参考运动 `mref`：控制初始状态和连续性

3. **精细控制机制**：
   - `ctrl_motion()` 函数支持 `delta_pitch/yaw/roll` 直接调整头部朝向
   - 支持区域控制和幅度控制

#### 解决速度问题

1. **紧凑的运动空间**：
   - 相比 VAE 空间，运动表示维度更低（如 63-D 变形向量 vs 数千维 VAE 特征）
   - 降低扩散模型的计算复杂度

2. **两阶段生成**：
   - 第一阶段：音频 → 运动（轻量级，快速）
   - 第二阶段：运动 → 图像（渲染，并行处理）

3. **流式处理优化**：
   - 支持在线模式和离线模式
   - 模块化设计，支持并行处理
   - TensorRT 优化加速

### 1.6 实验结果和结论

**主要成果**：
- ✅ 实现了**实时推理**（单 GPU）
- ✅ 支持**精细化控制**（情绪、头部姿态、表情）
- ✅ 生成质量高（唇形同步准确，表情自然）
- ✅ 开源代码和模型

---

## 2. DITTO 使用方法

### 2.1 环境安装

#### 方法一：使用 Conda（推荐）

```bash
# 1. 克隆代码
git clone https://github.com/antgroup/ditto-talkinghead
cd ditto-talkinghead

# 2. 创建 Conda 环境
conda env create -f environment.yaml
conda activate ditto

# 3. 如果 pip 安装失败，手动安装依赖
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --default-timeout=1000 -r <(cat <<EOF
audioread==3.0.1
cffi==1.17.1
cuda-python==12.6.2.post1
cython==3.0.11
decorator==5.1.1
filetype==1.2.0
imageio==2.36.1
imageio-ffmpeg==0.5.1
joblib==1.4.2
lazy-loader==0.4
librosa==0.10.2.post1
llvmlite==0.43.0
msgpack==1.1.0
numba==0.60.0
nvidia-cublas-cu12==12.6.4.1
nvidia-cuda-runtime-cu12==12.6.77
nvidia-cudnn-cu12==9.6.0.74
opencv-python-headless==4.10.0.84
packaging==24.2
platformdirs==4.3.6
pooch==1.8.2
pycparser==2.22
scikit-image==0.25.0
scikit-learn==1.6.0
scipy==1.15.0
soundfile==0.13.0
soxr==0.5.0.post1
threadpoolctl==3.5.0
tifffile==2024.12.12
tqdm==4.67.1
polygraphy
colored
EOF
)

# 4. 安装 TensorRT（如果需要）
pip install --extra-index-url https://pypi.nvidia.com \
    tensorrt==8.6.1 tensorrt-bindings==8.6.1 tensorrt-libs==8.6.1
```

### 2.2 下载模型

```bash
# 安装 git-lfs（如果没有）
git lfs install

# 下载模型
git clone https://huggingface.co/digital-avatar/ditto-talkinghead checkpoints
```

模型目录结构：
```
./checkpoints/
├── ditto_cfg/
│   ├── v0.4_hubert_cfg_trt.pkl          # TensorRT 离线配置
│   ├── v0.4_hubert_cfg_trt_online.pkl   # TensorRT 在线配置
│   └── v0.4_hubert_cfg_pytorch.pkl      # PyTorch 配置
├── ditto_trt_Ampere_Plus/               # TensorRT 模型（Ampere+ GPU）
│   ├── lmdm_v0.4_hubert_fp32.engine
│   ├── motion_extractor_fp32.engine
│   └── ...
├── ditto_onnx/                          # ONNX 模型
│   └── ...
└── ditto_pytorch/                       # PyTorch 模型
    ├── models/
    │   ├── lmdm_v0.4_hubert.pth
    │   ├── motion_extractor.pth
    │   └── ...
    └── aux_models/
        └── ...
```

### 2.3 基础使用

#### 使用 TensorRT 模型（推荐，速度快）

```bash
python inference.py \
    --data_root "./checkpoints/ditto_trt_Ampere_Plus" \
    --cfg_pkl "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl" \
    --audio_path "./example/audio.wav" \
    --source_path "./example/image.png" \
    --output_path "./tmp/result.mp4"
```

#### 使用 PyTorch 模型（兼容性好）

```bash
python inference.py \
    --data_root "./checkpoints/ditto_pytorch" \
    --cfg_pkl "./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl" \
    --audio_path "./example/audio.wav" \
    --source_path "./example/image.png" \
    --output_path "./tmp/result.mp4"
```

#### 如果 GPU 不支持 Ampere+

```bash
# 1. 从 ONNX 转换为 TensorRT
python scripts/cvt_onnx_to_trt.py \
    --onnx_dir "./checkpoints/ditto_onnx" \
    --trt_dir "./checkpoints/ditto_trt_custom"

# 2. 使用自定义 TensorRT 模型
python inference.py \
    --data_root "./checkpoints/ditto_trt_custom" \
    --cfg_pkl "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl" \
    --audio_path "./example/audio.wav" \
    --source_path "./example/image.png" \
    --output_path "./tmp/result.mp4"
```

---

## 3. 细粒度控制实现与使用

### 3.1 控制机制原理

DITTO 通过 `ctrl_info` 参数实现细粒度控制。控制信息以字典或列表形式提供，键为帧索引，值为控制参数。

#### 控制参数类型

**头部姿态控制**：
- `delta_pitch`: 俯仰角偏移（度数）
- `delta_yaw`: 偏航角偏移（度数）
- `delta_roll`: 翻滚角偏移（度数）
- `alpha_pitch/yaw/roll`: 姿态缩放因子（乘法控制）

**表情控制**：
- `delta_exp`: 表情变形偏移（63-D 向量或部分维度）

**淡入淡出控制**：
- `fade_alpha`: 淡入/淡出透明度（0-1）
- `fade_out_keys`: 需要淡出的键（如 `("exp",)`）

**情绪控制**：
- 在 `setup_kwargs` 中设置 `emo` 参数

### 3.2 代码实现分析

#### 控制函数：`ctrl_motion()`

位置：`core/atomic_components/motion_stitch.py`

```python
def ctrl_motion(x_d_info, **kwargs):
    # 头部姿态偏移控制
    for kk in ["delta_pitch", "delta_yaw", "delta_roll"]:
        if kk in kwargs:
            k = kk[6:]  # 去掉 "delta_" 前缀
            # 将二值化表示转换为度数，然后加上偏移
            x_d_info[k] = bin66_to_degree(x_d_info[k]) + kwargs[kk]
    
    # 头部姿态缩放控制
    for kk in ["alpha_pitch", "alpha_yaw", "alpha_roll"]:
        if kk in kwargs:
            k = kk[6:]  # 去掉 "alpha_" 前缀
            x_d_info[k] = x_d_info[k] * kwargs[kk]
    
    # 表情变形偏移控制
    if "delta_exp" in kwargs:
        x_d_info["exp"] = x_d_info["exp"] + kwargs["delta_exp"]
    
    return x_d_info
```

**原理**：
1. **头部姿态控制**：直接对生成的 `pitch`, `yaw`, `roll` 进行偏移或缩放
2. **表情控制**：对 63-D 表情变形向量（21 个关键点 × 3 维）进行偏移
3. **动态应用**：在 `MotionStitch.__call__()` 中对每一帧应用控制

#### 凝视校正：`_fix_gaze()`

位置：`core/atomic_components/motion_stitch.py`

```python
def _fix_gaze(pose_s, x_d_info):
    x_ratio = 0.26
    y_ratio = 0.28
    
    yaw_s, pitch_s = pose_s  # 参考图像的头部姿态
    yaw_d = bin66_to_degree(x_d_info['yaw']).item()  # 生成图像的头部姿态
    pitch_d = bin66_to_degree(x_d_info['pitch']).item()
    
    # 计算头部姿态变化
    delta_yaw = yaw_d - yaw_s
    delta_pitch = pitch_d - pitch_s
    
    # 根据头部姿态变化调整眼部变形
    dx = delta_yaw * x_ratio
    dy = delta_pitch * y_ratio
    
    # 应用到眼部关键点
    x_d_info['exp'] = _eye_delta(x_d_info['exp'], dx, dy)
    return x_d_info
```

**原理**：
- 根据论文，通过回归建立 `δe = K(Re)` 映射
- 当头部旋转时，自动调整眼部变形，使凝视保持合理方向
- 使用经验比例因子 `x_ratio=0.26`, `y_ratio=0.28`

#### 情绪控制：`ConditionHandler`

位置：`core/atomic_components/condition_handler.py`

```python
class ConditionHandler:
    def setup(self, setup_info, emo, eye_f0_mode=False, ch_info=None):
        # 解析情绪参数
        self.emo_lst = self._parse_emo_seq(emo)
        
        # 情绪标签：0-7
        # 'Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise', 'Contempt'
    
    @staticmethod
    def _parse_emo_seq(emo, seq_len=-1):
        if isinstance(emo, int) and 0 <= emo < 8:
            # 单个情绪标签，如 4 (Neutral)
            emo_seq = _get_emo_avg(emo).reshape(1, 8)
        elif isinstance(emo, list) and emo and isinstance(emo[0], (list, tuple)):
            # 每帧的情绪标签，如 [[4], [3,4], [3], ...]
            emo_seq = np.stack([_get_emo_avg(i) for i in emo], 0)
        # ...
```

**原理**：
- 支持 8 种基础情绪：Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise, Contempt
- 情绪作为条件信号输入到 ECS 分支
- 通过 Cross-Attention 机制影响运动生成

### 3.3 具体使用方法

#### 方法一：通过 `inference.py` 的 `more_kwargs` 参数

创建一个控制配置文件 `control_config.pkl`：

```python
import pickle
import numpy as np

# 准备控制信息
control_config = {
    "setup_kwargs": {
        # 情绪控制：0-7，分别对应 8 种情绪
        # 0: Angry, 1: Disgust, 2: Fear, 3: Happy, 
        # 4: Neutral, 5: Sad, 6: Surprise, 7: Contempt
        "emo": 3,  # 设置为 Happy
        
        # 或者每帧不同情绪：
        # "emo": [[3], [3, 4], [4], ...],  # 前几帧 Happy，逐渐变为 Neutral
        
        # 淡入淡出设置
        "fade_type": "",  # "" | "d0" | "s"
        "fade_out_keys": ("exp",),  # 需要淡出的键
    },
    "run_kwargs": {
        # 淡入淡出帧数
        "fade_in": 10,   # 前 10 帧淡入
        "fade_out": 10,  # 后 10 帧淡出
        
        # 每帧的控制信息
        "ctrl_info": {
            # 第 0 帧：轻微低头
            0: {
                "delta_pitch": -5.0,  # 向下 5 度
            },
            # 第 30 帧：向右转头
            30: {
                "delta_yaw": 15.0,  # 向右 15 度
            },
            # 第 60 帧：恢复正常
            60: {
                "delta_yaw": 0.0,
                "delta_pitch": 0.0,
            },
            # 第 90 帧：向左转头并抬头
            90: {
                "delta_yaw": -10.0,  # 向左 10 度
                "delta_pitch": 8.0,  # 向上 8 度
            },
            # 表情控制示例（需要精确控制时使用）
            # 100: {
            #     "delta_exp": np.zeros((1, 63), dtype=np.float32),  # 63-D 向量
            #     # 或者只控制特定维度
            # },
        }
    }
}

# 保存配置
with open("control_config.pkl", "wb") as f:
    pickle.dump(control_config, f)
```

运行推理：

```bash
python inference.py \
    --data_root "./checkpoints/ditto_trt_Ampere_Plus" \
    --cfg_pkl "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl" \
    --audio_path "./example/audio.wav" \
    --source_path "./example/image.png" \
    --output_path "./tmp/result.mp4"
```

修改 `inference.py` 的 `run()` 调用：

```python
# 在 inference.py 中
run(SDK, audio_path, source_path, output_path, more_kwargs="control_config.pkl")
```

#### 方法二：直接修改 `inference.py`

创建一个新的推理脚本 `inference_controlled.py`：

```python
import librosa
import math
import os
import numpy as np
import pickle
from stream_pipeline_offline import StreamSDK

def run_controlled(SDK, audio_path, source_path, output_path, 
                   emotion=4,  # 情绪：0-7
                   head_movements=None):  # 头部运动控制
    """
    emotion: int | list[int] | list[list[int]]
        - int: 单一情绪（如 3 表示 Happy）
        - list[int]: 混合情绪（如 [3, 4] 表示 Happy 和 Neutral 的混合）
        - list[list[int]]: 每帧的情绪序列
    
    head_movements: dict
        {
            frame_idx: {
                "delta_pitch": float,  # 俯仰角偏移（度）
                "delta_yaw": float,    # 偏航角偏移（度）
                "delta_roll": float,   # 翻滚角偏移（度）
            }
        }
    """
    
    # 准备控制配置
    ctrl_info = {}
    if head_movements:
        for frame_idx, movements in head_movements.items():
            ctrl_info[frame_idx] = movements
    
    more_kwargs = {
        "setup_kwargs": {
            "emo": emotion,
        },
        "run_kwargs": {
            "fade_in": 10,
            "fade_out": 10,
            "ctrl_info": ctrl_info,
        }
    }
    
    SDK.setup(source_path, output_path, **more_kwargs["setup_kwargs"])
    
    audio, sr = librosa.core.load(audio_path, sr=16000)
    num_f = math.ceil(len(audio) / 16000 * 25)
    
    fade_in = more_kwargs["run_kwargs"]["fade_in"]
    fade_out = more_kwargs["run_kwargs"]["fade_out"]
    ctrl_info = more_kwargs["run_kwargs"]["ctrl_info"]
    
    SDK.setup_Nd(N_d=num_f, fade_in=fade_in, fade_out=fade_out, ctrl_info=ctrl_info)
    
    online_mode = SDK.online_mode
    if online_mode:
        chunksize = (3, 5, 2)
        audio = np.concatenate([np.zeros((chunksize[0] * 640,), dtype=np.float32), audio], 0)
        split_len = int(sum(chunksize) * 0.04 * 16000) + 80
        for i in range(0, len(audio), chunksize[1] * 640):
            audio_chunk = audio[i:i + split_len]
            if len(audio_chunk) < split_len:
                audio_chunk = np.pad(audio_chunk, (0, split_len - len(audio_chunk)), mode="constant")
            SDK.run_chunk(audio_chunk, chunksize)
    else:
        aud_feat = SDK.wav2feat.wav2feat(audio)
        SDK.audio2motion_queue.put(aud_feat)
    
    SDK.close()
    
    cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
    os.system(cmd)
    print(f"输出视频: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./checkpoints/ditto_trt_Ampere_Plus")
    parser.add_argument("--cfg_pkl", type=str, default="./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl")
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--emotion", type=int, default=4, help="情绪: 0-7 (0:Angry, 3:Happy, 4:Neutral, ...)")
    parser.add_argument("--head_yaw", type=float, default=None, help="头部偏航角偏移（度）")
    parser.add_argument("--head_pitch", type=float, default=None, help="头部俯仰角偏移（度）")
    parser.add_argument("--head_roll", type=float, default=None, help="头部翻滚角偏移（度）")
    parser.add_argument("--control_frame", type=int, default=0, help="应用控制的帧索引")
    
    args = parser.parse_args()
    
    SDK = StreamSDK(args.cfg_pkl, args.data_root)
    
    # 准备头部运动控制
    head_movements = None
    if any([args.head_yaw, args.head_pitch, args.head_roll]):
        head_movements = {
            args.control_frame: {}
        }
        if args.head_yaw is not None:
            head_movements[args.control_frame]["delta_yaw"] = args.head_yaw
        if args.head_pitch is not None:
            head_movements[args.control_frame]["delta_pitch"] = args.head_pitch
        if args.head_roll is not None:
            head_movements[args.control_frame]["delta_roll"] = args.head_roll
    
    run_controlled(
        SDK, 
        args.audio_path, 
        args.source_path, 
        args.output_path,
        emotion=args.emotion,
        head_movements=head_movements
    )
```

使用示例：

```bash
# 1. 设置情绪为 Happy
python inference_controlled.py \
    --audio_path "./example/audio.wav" \
    --source_path "./example/image.png" \
    --output_path "./tmp/result_happy.mp4" \
    --emotion 3

# 2. 设置头部向右转 15 度（在第 0 帧）
python inference_controlled.py \
    --audio_path "./example/audio.wav" \
    --source_path "./example/image.png" \
    --output_path "./tmp/result_turn_right.mp4" \
    --head_yaw 15.0 \
    --control_frame 0

# 3. 组合控制：Happy 情绪 + 头部动作
python inference_controlled.py \
    --audio_path "./example/audio.wav" \
    --source_path "./example/image.png" \
    --output_path "./tmp/result_controlled.mp4" \
    --emotion 3 \
    --head_yaw 10.0 \
    --head_pitch -5.0 \
    --control_frame 30
```

#### 方法三：在 Python 代码中直接调用

```python
from stream_pipeline_offline import StreamSDK
import librosa
import numpy as np

# 初始化 SDK
SDK = StreamSDK(
    cfg_pkl="./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl",
    data_root="./checkpoints/ditto_trt_Ampere_Plus"
)

# 准备控制参数
setup_kwargs = {
    "emo": 3,  # Happy 情绪
}

run_kwargs = {
    "fade_in": 10,
    "fade_out": 10,
    "ctrl_info": {
        # 动态头部运动：随时间变化
        0: {"delta_pitch": -5.0},      # 开始：轻微低头
        30: {"delta_yaw": 15.0},       # 30 帧：向右转
        60: {"delta_yaw": 0.0, "delta_pitch": 0.0},  # 恢复正常
        90: {"delta_yaw": -10.0, "delta_pitch": 8.0},  # 向左转并抬头
    }
}

# 设置和运行
SDK.setup(
    source_path="./example/image.png",
    output_path="./tmp/result.mp4",
    **setup_kwargs
)

audio, sr = librosa.core.load("./example/audio.wav", sr=16000)
num_f = math.ceil(len(audio) / 16000 * 25)

SDK.setup_Nd(
    N_d=num_f,
    fade_in=run_kwargs["fade_in"],
    fade_out=run_kwargs["fade_out"],
    ctrl_info=run_kwargs["ctrl_info"]
)

aud_feat = SDK.wav2feat.wav2feat(audio)
SDK.audio2motion_queue.put(aud_feat)
SDK.close()

# 合并音频
os.system('ffmpeg -loglevel error -y -i "./tmp/result.mp4.tmp.mp4" -i "./example/audio.wav" -map 0:v -map 1:a -c:v copy -c:a aac "./tmp/result.mp4"')
```

### 3.4 控制参数详细说明

#### 情绪控制（emo）

**类型**：`int | list[int] | list[list[int]] | numpy.ndarray`

**情绪标签**：
- `0`: Angry（愤怒）
- `1`: Disgust（厌恶）
- `2`: Fear（恐惧）
- `3`: Happy（开心）
- `4`: Neutral（中性）⭐ 默认
- `5`: Sad（悲伤）
- `6`: Surprise（惊讶）
- `7`: Contempt（轻蔑）

**使用示例**：
```python
# 单一情绪
"emo": 3  # 整个视频都是 Happy

# 混合情绪
"emo": [3, 4]  # Happy 和 Neutral 的混合

# 每帧不同情绪
"emo": [[3], [3, 4], [4], [5], [4]]  # 前两帧 Happy，逐渐变为 Neutral，然后 Sad，最后 Neutral
```

#### 头部姿态控制

**参数**：
- `delta_pitch`: 俯仰角偏移（度数）
  - 正值：向上抬头
  - 负值：向下低头
- `delta_yaw`: 偏航角偏移（度数）
  - 正值：向右转头
  - 负值：向左转头
- `delta_roll`: 翻滚角偏移（度数）
  - 正值：向右倾斜
  - 负值：向左倾斜
- `alpha_pitch/yaw/roll`: 姿态缩放（乘法）
  - 例如：`alpha_yaw: 1.2` 表示将偏航角放大 1.2 倍

**典型范围**：
- `delta_pitch`: -30° ~ +30°
- `delta_yaw`: -45° ~ +45°
- `delta_roll`: -15° ~ +15°

#### 表情控制（delta_exp）

**类型**：`numpy.ndarray`，形状 `(1, 63)` 或 `(batch, 63)`

**说明**：
- 63-D 向量对应 21 个关键点 × 3 个维度（x, y, z）
- 每个维度控制特定的面部区域
- 需要根据论文中的映射表（如图 3）来调整特定维度

**注意**：表情控制较为复杂，建议先使用情绪控制和头部姿态控制。

#### 淡入淡出控制

**参数**：
- `fade_in`: 淡入帧数（整数）
- `fade_out`: 淡出帧数（整数）
- `fade_alpha`: 手动设置的透明度（0-1，自动计算通常不需要）
- `fade_out_keys`: 需要淡出的键，如 `("exp",)` 表示只淡出表情

### 3.5 实际应用示例

#### 示例 1：演讲场景（头部自然摆动）

```python
ctrl_info = {
    # 开始：正视前方
    0: {"delta_pitch": 0.0, "delta_yaw": 0.0},
    # 10 帧后：轻微向右转
    10: {"delta_yaw": 8.0},
    # 20 帧后：回到中间
    20: {"delta_yaw": 0.0},
    # 30 帧后：向左转
    30: {"delta_yaw": -8.0},
    # 40 帧后：回到中间
    40: {"delta_yaw": 0.0},
    # 50 帧后：轻微抬头
    50: {"delta_pitch": 5.0},
    # 60 帧后：恢复正常
    60: {"delta_pitch": 0.0},
}
```

#### 示例 2：情感表达（开心 + 点头）

```python
setup_kwargs = {
    "emo": 3,  # Happy
}

run_kwargs = {
    "ctrl_info": {
        # 点头动作
        0: {"delta_pitch": 0.0},
        15: {"delta_pitch": -8.0},  # 低头
        30: {"delta_pitch": 5.0},   # 抬头
        45: {"delta_pitch": 0.0},   # 恢复
    }
}
```

#### 示例 3：对话场景（回应对方）

```python
ctrl_info = {
    # 开始：向右看（看向对话者）
    0: {"delta_yaw": 12.0, "delta_pitch": 2.0},
    # 说话时：转向镜头
    30: {"delta_yaw": 0.0, "delta_pitch": 0.0},
    # 结束：轻微点头表示理解
    90: {"delta_pitch": -5.0},
    100: {"delta_pitch": 0.0},
}
```

---

## 4. 与 LecturePPT 项目集成

在你的 `LecturePPT` 项目中，`inference_ditto.py` 已经集成了 DITTO：

```python
# 当前实现（无控制）
command = [
    ditto_python,
    'inference_ditto.py',
    '--data_root', ditto_data_root,
    '--cfg_pkl', ditto_cfg_pkl,
    '--audio_path', audio_path,
    '--source_path', source_image_path,
    '--output_path', output_path
]
```

**添加控制功能**：

修改 `talkingteacher3.py` 中的 `generate_video_ditto()` 函数：

```python
def generate_video_ditto(audio_path, source_image_path, output_path, section_number,
                         emotion=4,  # 新增：情绪控制
                         head_movements=None):  # 新增：头部运动控制
    """使用Ditto模型生成视频（支持控制）"""
    
    ditto_python = '/home/hrq/miniconda3/envs/ditto/bin/python'
    ditto_data_root = "./checkpoints_ditto/ditto_pytorch"
    ditto_cfg_pkl = "./checkpoints_ditto/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 准备控制配置文件（如果需要）
    if emotion != 4 or head_movements:
        control_config_path = f"temp_ditto_control_{section_number}.pkl"
        control_config = {
            "setup_kwargs": {"emo": emotion},
            "run_kwargs": {
                "ctrl_info": head_movements or {},
                "fade_in": 5,
                "fade_out": 5,
            }
        }
        with open(control_config_path, "wb") as f:
            pickle.dump(control_config, f)
    else:
        control_config_path = None
    
    command = [
        ditto_python,
        'inference_ditto.py',
        '--data_root', ditto_data_root,
        '--cfg_pkl', ditto_cfg_pkl,
        '--audio_path', audio_path,
        '--source_path', source_image_path,
        '--output_path', output_path,
    ]
    
    if control_config_path:
        command.extend(['--control_config', control_config_path])
    
    # ... 执行命令 ...
```

---

## 5. 总结

### DITTO 的核心优势

1. **实时推理**：在单 GPU 上实现实时生成
2. **精细控制**：支持情绪、头部姿态、表情的细粒度控制
3. **高质量生成**：运动自然，唇形同步准确
4. **易于集成**：代码开源，API 清晰

### 控制功能总结

| 控制类型 | 参数 | 说明 | 使用场景 |
|---------|------|------|---------|
| **情绪** | `emo` | 0-7 的情绪标签 | 控制整体情绪表达 |
| **头部姿态** | `delta_pitch/yaw/roll` | 角度偏移（度） | 控制头部朝向和动作 |
| **表情** | `delta_exp` | 63-D 向量 | 精细控制面部表情 |
| **淡入淡出** | `fade_in/out` | 帧数 | 平滑过渡效果 |

### 推荐使用方式

1. **基础使用**：直接使用默认参数，生成自然的说话人头视频
2. **情绪控制**：根据内容设置合适情绪（如演讲用 Neutral，娱乐用 Happy）
3. **头部动作**：添加自然的头部摆动，增强真实感
4. **精细调整**：对于特定需求，使用表情控制进行微调

---

**参考文献**：
- DITTO 论文：https://arxiv.org/abs/2411.19509
- 项目主页：https://digital-avatar.github.io/ai/Ditto/
- GitHub：https://github.com/antgroup/ditto-talkinghead
- HuggingFace：https://huggingface.co/digital-avatar/ditto-talkinghead

