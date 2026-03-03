# Demo: GPTQ W4A16 量化提交包

本 demo 展示如何通过 `prepare_env.sh` + `prepare_model.sh` 接口，对 MiniCPM-SALA 模型进行 RTN W4A16 量化，并以 GPTQ 格式在 SGLang 中加载运行。

## 文件说明

| 文件 | 说明 |
|---|---|
| `prepare_env.sh` | 环境准备：patch SGLang 注意力后端 dtype、设置 `SGLANG_SERVER_ARGS` |
| `prepare_model.sh` | 模型预处理入口，调用 `quantize_gptq.py` |
| `quantize_gptq.py` | RTN W4A16 量化脚本，将权重转为 GPTQ 格式 |

## 执行流程

1. **`prepare_env.sh`**（由平台 `source` 执行）
   - 将 SGLang 注意力后端中硬编码的 `bfloat16` 替换为 `float16`，以兼容 GPTQ 量化
   - 导出 `SGLANG_SERVER_ARGS`，添加 `--quantization gptq --dtype float16 --disable-cuda-graph`

2. **`prepare_model.sh --input <模型路径> --output <输出路径>`**
   - 对所有线性层权重执行 RTN (Round-To-Nearest) 4-bit 量化
   - 生成 SGLang 兼容的 GPTQ 格式权重（qweight / scales / qzeros / g_idx）
   - 写入 `config.json` 和 `quantize_config.json`

## 注意事项

- 本 demo 使用 RTN 量化（无需校准数据），精度不如真正的 GPTQ，仅作为格式示例
- 量化后模型约为原始模型的 1/4 大小
- `group_size=1024`，可在 `quantize_gptq.py` 中调整
