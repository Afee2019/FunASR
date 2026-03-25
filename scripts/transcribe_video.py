#!/usr/bin/env python3
"""将视频/音频文件的音轨转为文字，智能分段后保存为文本文件。

用法:
    # 完整流程：识别 + 智能分段
    python scripts/transcribe_video.py video.mkv

    # 只做智能分段（跳过识别，复用已有的 raw 文件）
    python scripts/transcribe_video.py --raw existing_raw.txt

    # 只做识别，不做分段
    python scripts/transcribe_video.py video.mkv --no-segment
"""

import argparse
import os
import json
from datetime import datetime
from urllib.request import Request, urlopen

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_API_KEY = "sk-bc09bb6516a040c28ecd63032fb82b69"
DEEPSEEK_MODEL = "deepseek-chat"


def smart_segment(raw_text: str) -> str:
    """调用 DeepSeek API 对 ASR 原始文本进行智能分段。"""
    print("\n正在调用 DeepSeek 进行智能分段...")

    prompt = f"""请对以下语音识别的原始文本进行智能分段处理。要求：

1. 根据语义和话题切换，将文本分成合理的段落
2. 每个段落之间用空行分隔
3. 修正明显的语音识别错误（如同音字错误）
4. 保留原文的所有信息，不要删减内容
5. 不要添加标题、编号或任何额外的标注
6. 直接输出分段后的文本，不要有任何解释说明

原始文本：
{raw_text}"""

    body = json.dumps({
        "model": DEEPSEEK_MODEL,
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()

    req = Request(
        f"{DEEPSEEK_BASE_URL}/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        },
    )

    with urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())

    if "choices" not in data or not data["choices"]:
        print(f"API 响应异常: {json.dumps(data, ensure_ascii=False, indent=2)[:1000]}")
        raise RuntimeError("DeepSeek API 返回了意外的响应格式")

    result_text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    print(f"分段完成 (prompt_tokens: {usage.get('prompt_tokens', '?')}, completion_tokens: {usage.get('completion_tokens', '?')})")
    return result_text


def transcribe(video_path: str) -> str:
    """语音识别，返回原始文本。"""
    from funasr import AutoModel

    model = AutoModel(
        model="paraformer-zh",
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        device="cpu",
        ncpu=4,
        disable_update=True,
    )
    result = model.generate(input=video_path, batch_size_s=300)
    return "\n".join(item["text"] for item in result if item["text"])


def main():
    parser = argparse.ArgumentParser(description="视频/音频转文字 + 智能分段")
    parser.add_argument("input", help="视频/音频文件路径")
    parser.add_argument("--raw", help="已有的原始识别文件（跳过识别步骤）")
    parser.add_argument("--no-segment", action="store_true", help="只做识别，不做智能分段")
    parser.add_argument("-o", "--output-dir", help="输出目录（默认与输入文件同目录）")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = args.output_dir or os.path.dirname(input_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Step 1: 语音识别
    if args.raw and os.path.exists(args.raw):
        print(f"=== Step 1: 跳过识别，读取已有文件: {args.raw} ===")
        with open(args.raw, "r", encoding="utf-8") as f:
            raw_text = f.read()
    else:
        print("=== Step 1: 语音识别 ===")
        raw_text = transcribe(input_path)

        raw_path = os.path.join(output_dir, f"{base_name}_{timestamp}_raw.txt")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(raw_text)
        print(f"\n原始结果已保存到: {raw_path}")

    if args.no_segment:
        return

    # Step 2: 智能分段
    print("\n=== Step 2: 智能分段 ===")
    segmented_text = smart_segment(raw_text)

    seg_path = os.path.join(output_dir, f"{base_name}_{timestamp}.txt")
    with open(seg_path, "w", encoding="utf-8") as f:
        f.write(segmented_text)

    print(f"分段结果已保存到: {seg_path}")
    print(f"\n{'='*40}")
    print(segmented_text[:500] + ("..." if len(segmented_text) > 500 else ""))


if __name__ == "__main__":
    main()
