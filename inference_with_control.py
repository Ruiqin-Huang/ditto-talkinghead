"""
DITTO 带控制功能的推理脚本

支持的情绪：0-7
- 0: Angry（愤怒）
- 1: Disgust（厌恶）
- 2: Fear（恐惧）
- 3: Happy（开心）
- 4: Neutral（中性）- 默认
- 5: Sad（悲伤）
- 6: Surprise（惊讶）
- 7: Contempt（轻蔑）

支持的头部姿态控制：
- delta_pitch: 俯仰角偏移（度，正数向上，负数向下）
- delta_yaw: 偏航角偏移（度，正数向右，负数向左）
- delta_roll: 翻滚角偏移（度，正数向右倾斜，负数向左倾斜）
"""

import librosa
import math
import os
import numpy as np
import pickle
from stream_pipeline_offline import StreamSDK


def create_control_config(emotion=4, head_movements=None, fade_in=10, fade_out=10):
    """
    创建控制配置
    
    Args:
        emotion: 情绪标签或情绪序列
            - int: 单一情绪（0-7）
            - list[int]: 混合情绪，如 [3, 4]
            - list[list[int]]: 每帧的情绪序列，如 [[3], [3, 4], [4], ...]
        head_movements: 头部运动控制字典
            {
                frame_idx: {
                    "delta_pitch": float,  # 俯仰角偏移（度）
                    "delta_yaw": float,    # 偏航角偏移（度）
                    "delta_roll": float,   # 翻滚角偏移（度）
                }
            }
        fade_in: 淡入帧数
        fade_out: 淡出帧数
    
    Returns:
        dict: 控制配置
    """
    control_config = {
        "setup_kwargs": {
            "emo": emotion,
        },
        "run_kwargs": {
            "fade_in": fade_in,
            "fade_out": fade_out,
            "ctrl_info": head_movements or {},
        }
    }
    return control_config


def run_with_control(SDK, audio_path, source_path, output_path, control_config=None):
    """
    使用控制配置运行推理
    
    Args:
        SDK: StreamSDK 实例
        audio_path: 音频文件路径
        source_path: 源图像/视频路径
        output_path: 输出视频路径
        control_config: 控制配置字典
    """
    if control_config is None:
        control_config = create_control_config()
    
    setup_kwargs = control_config.get("setup_kwargs", {})
    run_kwargs = control_config.get("run_kwargs", {})
    
    # 设置
    SDK.setup(source_path, output_path, **setup_kwargs)
    
    # 加载音频并计算帧数
    audio, sr = librosa.core.load(audio_path, sr=16000)
    num_f = math.ceil(len(audio) / 16000 * 25)
    
    # 设置控制参数
    fade_in = run_kwargs.get("fade_in", -1)
    fade_out = run_kwargs.get("fade_out", -1)
    ctrl_info = run_kwargs.get("ctrl_info", {})
    
    SDK.setup_Nd(N_d=num_f, fade_in=fade_in, fade_out=fade_out, ctrl_info=ctrl_info)
    
    # 运行推理
    online_mode = SDK.online_mode
    if online_mode:
        chunksize = run_kwargs.get("chunksize", (3, 5, 2))
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
    
    # 合并音频
    cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
    print(f"执行: {cmd}")
    os.system(cmd)
    
    print(f"✅ 输出视频: {output_path}")


def create_natural_head_movement(num_frames, interval=30):
    """
    创建自然的头部运动序列
    
    Args:
        num_frames: 总帧数
        interval: 头部动作间隔（帧）
    
    Returns:
        dict: 头部运动控制字典
    """
    head_movements = {}
    
    for i in range(0, num_frames, interval):
        # 随机选择头部动作
        action = np.random.choice(["left", "right", "up", "down", "center"])
        
        if action == "left":
            head_movements[i] = {"delta_yaw": -10.0 + np.random.uniform(-3, 3)}
        elif action == "right":
            head_movements[i] = {"delta_yaw": 10.0 + np.random.uniform(-3, 3)}
        elif action == "up":
            head_movements[i] = {"delta_pitch": 5.0 + np.random.uniform(-2, 2)}
        elif action == "down":
            head_movements[i] = {"delta_pitch": -5.0 + np.random.uniform(-2, 2)}
        else:  # center
            head_movements[i] = {"delta_yaw": 0.0, "delta_pitch": 0.0}
        
        # 下一帧恢复
        if i + 10 < num_frames:
            head_movements[i + 10] = {"delta_yaw": 0.0, "delta_pitch": 0.0}
    
    return head_movements


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DITTO 带控制功能的推理脚本")
    parser.add_argument("--data_root", type=str, default="./checkpoints/ditto_trt_Ampere_Plus",
                        help="模型根目录路径")
    parser.add_argument("--cfg_pkl", type=str, default="./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl",
                        help="配置文件路径")
    parser.add_argument("--audio_path", type=str, required=True,
                        help="输入音频文件路径")
    parser.add_argument("--source_path", type=str, required=True,
                        help="输入图像/视频路径")
    parser.add_argument("--output_path", type=str, required=True,
                        help="输出视频路径")
    
    # 情绪控制
    parser.add_argument("--emotion", type=int, default=4,
                        help="情绪标签 (0:Angry, 1:Disgust, 2:Fear, 3:Happy, 4:Neutral, 5:Sad, 6:Surprise, 7:Contempt)")
    
    # 头部姿态控制
    parser.add_argument("--head_yaw", type=float, default=None,
                        help="头部偏航角偏移（度，正数向右，负数向左）")
    parser.add_argument("--head_pitch", type=float, default=None,
                        help="头部俯仰角偏移（度，正数向上，负数向下）")
    parser.add_argument("--head_roll", type=float, default=None,
                        help="头部翻滚角偏移（度）")
    parser.add_argument("--control_frame", type=int, default=0,
                        help="应用头部控制的帧索引")
    
    # 自动头部运动
    parser.add_argument("--auto_head_movement", action="store_true",
                        help="启用自动头部运动（自然的头部摆动）")
    
    # 淡入淡出
    parser.add_argument("--fade_in", type=int, default=10,
                        help="淡入帧数")
    parser.add_argument("--fade_out", type=int, default=10,
                        help="淡出帧数")
    
    # 控制配置文件
    parser.add_argument("--control_config", type=str, default=None,
                        help="控制配置文件路径（.pkl 格式）")
    
    args = parser.parse_args()
    
    # 初始化 SDK
    SDK = StreamSDK(args.cfg_pkl, args.data_root)
    
    # 准备控制配置
    if args.control_config:
        # 从文件加载
        print(f"📁 从文件加载控制配置: {args.control_config}")
        with open(args.control_config, "rb") as f:
            control_config = pickle.load(f)
    else:
        # 从命令行参数创建
        head_movements = None
        
        # 自动头部运动
        if args.auto_head_movement:
            audio, sr = librosa.core.load(args.audio_path, sr=16000)
            num_frames = math.ceil(len(audio) / 16000 * 25)
            head_movements = create_natural_head_movement(num_frames)
            print(f"🎬 启用自动头部运动，共 {len(head_movements)} 个控制点")
        
        # 手动头部控制
        elif any([args.head_yaw, args.head_pitch, args.head_roll]):
            head_movements = {
                args.control_frame: {}
            }
            if args.head_yaw is not None:
                head_movements[args.control_frame]["delta_yaw"] = args.head_yaw
                print(f"↔️  设置偏航角: {args.head_yaw}° (帧 {args.control_frame})")
            if args.head_pitch is not None:
                head_movements[args.control_frame]["delta_pitch"] = args.head_pitch
                print(f"↕️  设置俯仰角: {args.head_pitch}° (帧 {args.control_frame})")
            if args.head_roll is not None:
                head_movements[args.control_frame]["delta_roll"] = args.head_roll
                print(f"↻  设置翻滚角: {args.head_roll}° (帧 {args.control_frame})")
        
        # 情绪标签说明
        emotion_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise", "Contempt"]
        print(f"😊 设置情绪: {emotion_names[args.emotion]} ({args.emotion})")
        
        control_config = create_control_config(
            emotion=args.emotion,
            head_movements=head_movements,
            fade_in=args.fade_in,
            fade_out=args.fade_out
        )
    
    # 运行推理
    print(f"🚀 开始生成视频...")
    print(f"   音频: {args.audio_path}")
    print(f"   图像: {args.source_path}")
    print(f"   输出: {args.output_path}")
    
    run_with_control(SDK, args.audio_path, args.source_path, args.output_path, control_config)
    
    print("✅ 完成！")

