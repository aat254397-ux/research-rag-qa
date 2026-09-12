# Simple Document QA

第三周开始构建的简单文档问答助手。

初始功能范围：

1. 读取 TXT、Markdown，后续增加普通文本型 PDF；
2. 文本切分；
3. Embedding 与相似度检索；
4. 根据检索内容回答问题；
5. 展示引用的原文片段；
6. 可选的简单 Gradio 界面。

暂不加入多智能体、复杂 Memory、网页搜索、模型微调或生产级部署。

## 当前阶段

Day 15：建立项目结构并实现 TXT、Markdown 文档读取。

计划结构：

- `sample_docs/`：用于本地验证的示例文档。
- `tests/`：项目测试。
- `document_loader.py`：读取并验证文档。
- `text_chunker.py`：后续加入文本切片。
- `retrieval.py`：后续加入索引与检索。
- `prompt_builder.py`：后续加入 Prompt 组装。
- `app.py`：后续串联完整问答流程。
