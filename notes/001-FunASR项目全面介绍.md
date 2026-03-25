# 001 - FunASR 项目全面介绍

## 一句话概括

FunASR 是阿里达摩院开源的语音识别工具包，能把人说的话变成文字，同时还能判断谁在说话、说话人的情绪、语音中的事件（如掌声、笑声），支持从学术研究到工业部署的完整链路。

---

## 一、这个项目到底是干嘛的？

想象一下你对着手机说了一段话，手机把你说的内容一字不差地显示在屏幕上——这就是语音识别（ASR）最核心的功能。

但现实场景远比这复杂：

- 一段录音可能有好几个人在说话，你需要知道**谁说了什么**
- 一段长达数小时的会议录音，你需要它自动分段、加标点、标注时间戳
- 你可能在嘈杂环境中说话，需要先检测**哪些片段有人声**
- 你可能需要识别说话人的**情绪**（愤怒、开心、悲伤）
- 你可能需要一个**关键词唤醒**系统（类似"小爱同学"、"Alexa"）

FunASR 把这些能力打包在一起，提供了一站式解决方案：

| 能力 | 技术名称 | 做什么 |
|------|---------|--------|
| 语音识别 | ASR | 语音转文字 |
| 语音活动检测 | VAD | 检测音频中哪些片段有人在说话 |
| 标点恢复 | Punctuation | 给识别出的文字自动加标点 |
| 说话人验证 | Speaker Verification | 判断"这段话是不是张三说的" |
| 说话人分离 | Speaker Diarization | 区分一段录音中不同说话人 |
| 情绪识别 | Emotion Recognition | 识别说话人的情绪状态 |
| 关键词检测 | KWS | 检测音频中是否包含特定关键词 |
| 时间戳对齐 | Timestamp | 精确标注每个字/词的起止时间 |
| 语言识别 | LID | 自动判断说的是哪种语言 |

---

## 二、输入是什么？输出是什么？

### 输入

FunASR 的输入非常灵活：

**音频格式**：WAV、MP3 及 FFmpeg 支持的几乎所有音频格式，采样率自动重采样到模型所需（通常 16kHz）

**输入方式**：
- 本地文件路径：`input="meeting.wav"`
- HTTP/HTTPS 远程 URL：`input="https://example.com/audio.wav"`
- 音频字节流：适合实时麦克风采集场景
- NumPy 数组：已加载的音频波形数据
- PyTorch 张量：已提取的声学特征
- 批量文件列表：Kaldi 格式的 wav.scp、JSONL 文件
- 视频文件：通过 Runtime 服务支持，自动提取音轨

**典型代码**：
```python
from funasr import AutoModel

model = AutoModel(model="paraformer-zh",
                  vad_model="fsmn-vad",
                  punc_model="ct-punc")

# 单个文件
result = model.generate(input="meeting.wav")

# 批量处理
result = model.generate(input="wavs.scp")
```

### 输出

输出是结构化的字典列表，根据配置的模型组合不同，包含以下字段：

```python
[{
    "key": "meeting",                           # 音频标识
    "text": "今天的会议到此结束，谢谢大家。",       # 识别文本（含标点）
    "timestamp": [[380, 560], [560, 800], ...],  # 每个字的起止时间（毫秒）
    "spk_id": "speaker_0",                       # 说话人标识（如配置了说话人模型）
    "emotion": "neutral",                        # 情绪标签（如配置了情绪模型）
    "language": "zh",                            # 语言标识
}]
```

简单说：**输入是声音，输出是带有丰富元信息的结构化文本**。

---

## 三、具体能用在哪些场景？

### 1. 会议纪要自动生成

**场景**：一场 2 小时的多人会议，需要自动生成带说话人标注和时间戳的会议纪要。

**流程**：VAD（切分有效语音段） → ASR（语音转文字） → 标点恢复 → 说话人分离

**输出示例**：
```
[00:01:23] 张总：下面我们讨论一下第三季度的销售目标。
[00:01:35] 李经理：根据目前的数据，我建议将目标定在八千万。
[00:01:42] 张总：这个数字是怎么算出来的？
```

### 2. 客服质检

**场景**：电商客服每天处理上万通电话，需要自动转录并分析服务质量。

**用到的能力**：ASR + 情绪识别 + 关键词检测

**价值**：自动检测客户是否愤怒、客服是否使用了禁用话术、是否遗漏了必说话术。

### 3. 字幕生成

**场景**：视频平台需要为海量视频自动生成字幕。

**用到的能力**：ASR + 时间戳对齐 + 标点恢复

**价值**：精确到字级别的时间戳让字幕与画面完美同步，标点恢复让字幕可读性更好。支持中、英、日、韩等多语言。

### 4. 智能语音助手 / 车载语音

**场景**：用户对着设备说"导航到人民广场"，系统需要实时识别。

**用到的能力**：关键词唤醒（KWS） + 流式 ASR

**特点**：FunASR 的 Paraformer-online 支持流式识别，延迟可控制在 480~600ms，满足实时交互需求。

### 5. 有声内容转文稿

**场景**：播客、有声书、培训课程等音频内容需要转成文字用于搜索、归档、二次编辑。

**用到的能力**：VAD + ASR + 标点恢复 + 时间戳

**优势**：Paraformer-large-long 模型专门针对长音频优化，不限音频时长。

### 6. 多语言内容审核

**场景**：社交平台需要审核用户上传的多语种音频内容。

**用到的能力**：语言识别 + 多语言 ASR + 情绪/事件检测

**优势**：SenseVoice 模型一次推理同时输出语言类型、识别文本、情绪和声学事件（掌声、笑声、音乐等）。

### 7. 医疗 / 法律 / 金融领域的语音文档化

**场景**：医生口述病历、律师录制庭审笔录、金融合规录音回溯。

**用到的能力**：ASR + 热词定制（SeACo-Paraformer）

**价值**：通过热词功能可以显著提升专业术语的识别准确率，无需重新训练模型。

### 8. 方言识别

**场景**：面向全国用户的语音服务，需要支持各地方言。

**优势**：Fun-ASR-Nano 模型支持中文 7 大方言（上海话、四川话、粤语等）及 26 种地方口音，这在开源项目中极为少见。

---

## 四、核心模型介绍

| 模型 | 参数量 | 特点 | 适用场景 |
|------|--------|------|---------|
| **Paraformer-large** | 220M | 非自回归架构，精度高、速度快 | 离线中文 ASR 首选 |
| **Paraformer-large-online** | 220M | 流式版本，支持实时识别 | 实时转写 |
| **SeACo-Paraformer** | 220M | 支持热词定制 | 专业术语场景 |
| **SenseVoice** | 234M | 多任务：ASR+情绪+事件+语种 | 需要丰富元信息的场景 |
| **Fun-ASR-Nano** | 800M | 31种语言，7种中文方言 | 多语言多方言场景 |
| **Whisper-large-v3** | 1550M | OpenAI 模型，99+语言 | 小语种覆盖 |
| **FSMN-VAD** | 0.4M | 极轻量级语音检测 | 前置切分 |
| **CT-Transformer** | 290M | 中英文标点恢复 | 文本后处理 |
| **CAM++** | 7.2M | 说话人特征提取 | 说话人相关任务 |
| **emotion2vec** | 300M | 5类情绪分类 | 情绪分析 |

---

## 五、同类产品对比

### 主要竞品

| 项目 | 开发者 | 定位 |
|------|--------|------|
| **OpenAI Whisper** | OpenAI | 多语言通用 ASR |
| **WeNet** | 出门问问 + 西工大 | 端到端流式/离线 ASR |
| **ESPnet** | 约翰霍普金斯大学 | 学术研究框架 |
| **PaddleSpeech** | 百度 | 语音全栈工具包 |
| **NeMo** | NVIDIA | GPU 优化的语音 AI |
| **SpeechBrain** | Mila 研究所 | 通用语音工具包 |
| **Faster-Whisper** | SYSTRAN | Whisper 的 CTranslate2 加速版 |

### FunASR 的差异化优势

**1. 非自回归架构带来的速度优势**

Whisper、WeNet 等主流模型采用自回归（autoregressive）解码——逐字生成，前一个字生成后才能生成下一个字。Paraformer 采用非自回归（non-autoregressive）架构，所有字并行生成，推理速度比同精度的自回归模型快数倍。

这意味着在同等硬件条件下，FunASR 能处理更多并发请求。

**2. 工业级全链路 Pipeline**

多数开源项目只提供 ASR 模型本身。FunASR 提供了从 VAD → ASR → 标点 → 说话人 → 情绪 的完整 Pipeline，通过 AutoModel 一行代码即可组装：

```python
model = AutoModel(
    model="paraformer-zh",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    spk_model="cam++",
)
```

竞品通常需要自己拼接多个独立项目才能实现类似效果。

**3. 中文识别精度领先**

在权威中文语音基准上，Paraformer-large 的表现：

| 测试集 | Paraformer-large (CER) |
|--------|----------------------|
| AISHELL-1 test | 1.94% |
| AISHELL-2 test_ios | 2.84% |
| WenetSpeech test_net | 6.66% |

这些数字在开源模型中处于领先水平。CER（字错误率）越低越好，1.94% 意味着每识别 100 个字只错不到 2 个。

**4. 方言和口音支持**

这是 FunASR 最独特的优势之一。Fun-ASR-Nano 支持中文 7 大方言 + 26 种地方口音，基于数千万小时真实语音数据训练。竞品（包括 Whisper）对中文方言的支持非常有限。

**5. 热词定制无需重训练**

SeACo-Paraformer 支持运行时传入热词列表，无需任何额外训练即可提升特定词汇的识别率。这对医疗（药名）、法律（法条编号）、金融（产品名称）等专业领域至关重要。

竞品实现类似功能通常需要语言模型重打分（LM rescoring）或微调。

**6. 生产级部署方案齐全**

FunASR 的 `runtime/` 目录提供了开箱即用的部署方案：

- WebSocket 实时流式服务（支持 SSL）
- HTTP REST API 服务
- gRPC 高性能服务
- ONNX Runtime C++ 推理（去除 Python 依赖）
- Triton GPU 推理服务
- Docker 容器化部署（CPU / GPU 版本）
- 客户端 SDK：Android、iOS、C#、Java、Go、HTML5

很少有开源 ASR 项目能提供如此完整的跨平台部署支持。

**7. 灵活的训练和微调**

- 支持 LoRA 参数高效微调（少量数据即可适配新领域）
- 支持 DeepSpeed 多机多卡分布式训练
- 支持 DDP 和 FSDP 训练策略
- 提供分阶段微调建议：<1000h 只调适配层，<5000h 调编码器+适配层，>10000h 全参数微调

**8. 模型生态丰富**

ModelScope 和 HuggingFace 上提供 100+ 预训练模型，覆盖不同语言、不同规模、不同任务，可通过模型名直接下载使用。

---

## 六、与竞品的取舍建议

| 需求 | 推荐方案 |
|------|---------|
| 中文离线转写（追求精度） | FunASR Paraformer-large |
| 中文实时流式识别 | FunASR Paraformer-online |
| 中文方言识别 | FunASR Fun-ASR-Nano |
| 需要完整 Pipeline（VAD+ASR+标点+说话人） | FunASR（原生支持最完整） |
| 99+ 小语种覆盖 | Whisper（FunASR 也集成了 Whisper） |
| 纯英文高精度 | Whisper-large-v3 或 NeMo |
| 学术研究 / 自定义模型结构 | ESPnet 或 FunASR |
| NVIDIA GPU 深度优化 | NeMo |
| 纯离线、极简部署 | Faster-Whisper |

---

## 七、技术架构速览

```
用户音频输入
    │
    ▼
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  VAD    │ →  │  ASR    │ →  │ 标点恢复 │ →  │ 说话人   │
│ 切分有效 │    │ 语音转文字│    │ 加标点   │    │ 分离标注  │
│ 语音片段 │    │ +时间戳  │    │         │    │         │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
                                                  │
                                                  ▼
                                          结构化输出结果
                                    (文本+时间戳+说话人+情绪)
```

**AutoModel** 是核心入口，通过注册表机制（Registry Pattern）自动发现和加载 50+ 种模型实现，用户只需指定模型名称，无需关心底层实现细节。

---

## 八、快速体验

```python
# 安装
# pip install funasr

# 最简用法：3 行代码完成语音识别
from funasr import AutoModel
model = AutoModel(model="paraformer-zh")
result = model.generate(input="your_audio.wav")
print(result[0]["text"])

# 完整 Pipeline：VAD + ASR + 标点 + 说话人
model = AutoModel(
    model="paraformer-zh",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    spk_model="cam++",
)
result = model.generate(
    input="meeting.wav",
    batch_size_s=300,  # 每批处理 300 秒音频
)
for item in result:
    print(f"[{item.get('spk_id', '?')}] {item['text']}")
```

---

*文档编号：001 | 项目：FunASR v1.3.1 | 更新日期：2026-03-24*
