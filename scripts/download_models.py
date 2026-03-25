#!/usr/bin/env python3
"""下载 FunASR 核心模型到本地缓存。

用法:
    # 下载所有缺少的模型
    python scripts/download_models.py

    # 下载指定模型
    python scripts/download_models.py --models sensevoice emotion2vec

    # 列出所有可用模型及状态
    python scripts/download_models.py --list

    # 强制重新下载
    python scripts/download_models.py --force
"""

import argparse
import os
import sys

MODELS = {
    # 别名: (ModelScope ID, 简要说明)
    "paraformer-zh": (
        "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "SeACo-Paraformer 中文离线 ASR（支持热词）",
    ),
    "paraformer": (
        "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "Paraformer-large 中文离线 ASR",
    ),
    "paraformer-streaming": (
        "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
        "Paraformer-large 中文流式 ASR",
    ),
    "fsmn-vad": (
        "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "FSMN-VAD 语音活动检测",
    ),
    "ct-punc": (
        "iic/punc_ct-transformer_cn-en-common-vocab471067-large",
        "CT-Transformer 中英文标点恢复",
    ),
    "cam++": (
        "iic/speech_campplus_sv_zh-cn_16k-common",
        "CAM++ 说话人验证",
    ),
    "sensevoice": (
        "iic/SenseVoiceSmall",
        "SenseVoice 多任务模型（ASR+情绪+事件+语种）",
    ),
    "fun-asr-nano": (
        "FunAudioLLM/Fun-ASR-Nano-2512",
        "Fun-ASR-Nano 31语言+7方言（800M 参数，较大）",
    ),
    "whisper-large-v3": (
        "iic/Whisper-large-v3",
        "Whisper-large-v3 多语言 ASR（1550M 参数，较大）",
    ),
    "emotion2vec": (
        "iic/emotion2vec_plus_large",
        "emotion2vec 语音情绪识别",
    ),
}


def get_cache_dir():
    """获取 ModelScope 缓存目录。"""
    return os.environ.get(
        "MODELSCOPE_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "modelscope"),
    )


def is_model_cached(model_id: str) -> bool:
    """检查模型是否已在本地缓存。"""
    cache_dir = get_cache_dir()
    # ModelScope 缓存路径: {cache_dir}/hub/models/{model_id}/
    model_path = os.path.join(cache_dir, "hub", "models", model_id)
    if os.path.isdir(model_path):
        # 至少有一些文件才算已下载
        files = os.listdir(model_path)
        return len(files) > 1
    return False


def list_models():
    """列出所有模型及其本地状态。"""
    print(f"{'别名':<24} {'状态':<8} 说明")
    print("-" * 80)
    for alias, (model_id, desc) in MODELS.items():
        cached = is_model_cached(model_id)
        status = "已下载" if cached else "未下载"
        print(f"{alias:<24} {status:<8} {desc}")
    print()
    print(f"缓存目录: {get_cache_dir()}")


def download_model(model_id: str, alias: str, desc: str):
    """下载单个模型。"""
    from modelscope.hub.snapshot_download import snapshot_download

    print(f"\n{'='*60}")
    print(f"下载: {alias}")
    print(f"  ID: {model_id}")
    print(f"  说明: {desc}")
    print(f"{'='*60}")

    try:
        path = snapshot_download(model_id)
        print(f"  完成! 保存到: {path}")
        return True
    except Exception as e:
        print(f"  失败: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="下载 FunASR 核心模型")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        help="指定要下载的模型别名（默认下载所有缺少的模型）",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有模型及下载状态",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载（即使已缓存）",
    )
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    targets = args.models or list(MODELS.keys())

    to_download = []
    for alias in targets:
        model_id, desc = MODELS[alias]
        if args.force or not is_model_cached(model_id):
            to_download.append((alias, model_id, desc))
        else:
            print(f"  跳过 {alias} (已缓存)")

    if not to_download:
        print("所有模型已下载，无需更新。")
        return

    print(f"\n将下载 {len(to_download)} 个模型:")
    for alias, model_id, desc in to_download:
        print(f"  - {alias}: {desc}")

    success, failed = 0, 0
    for alias, model_id, desc in to_download:
        if download_model(model_id, alias, desc):
            success += 1
        else:
            failed += 1

    print(f"\n完成: {success} 成功, {failed} 失败")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
