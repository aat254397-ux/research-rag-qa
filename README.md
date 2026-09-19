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

Day 19：文档索引与问题检索已经完成，正在把检索结果和问题组装成模型可用的 Prompt。

计划结构：

- `sample_docs/`：用于本地验证的示例文档。
- `tests/`：项目测试。
- `document_loader.py`：读取并验证文档。
- `text_chunker.py`：按字符范围切片并保留来源和位置。
- `retrieval.py`：建立 TF-IDF 索引，并正在加入问题检索。
- `prompt_builder.py`：正在加入检索上下文与 Prompt 组装。
- `app.py`：后续串联完整问答流程。
