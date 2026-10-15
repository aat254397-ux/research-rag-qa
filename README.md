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

Day 20：已将文档读取、切片、索引、检索和 Prompt 构建串联成完整主流程。

计划结构：

- `sample_docs/`：用于本地验证的示例文档。
- `tests/`：项目测试。
- `document_loader.py`：读取并验证文档。
- `text_chunker.py`：按字符范围切片并保留来源和位置。
- `retrieval.py`：建立 TF-IDF 索引并执行问题检索。
- `prompt_builder.py`：构建检索上下文与完整 Prompt。
- `app.py`：串联文档读取、切片、索引、检索与 Prompt 构建流程。
## 模拟模型回答流程

已连接文档读取、切片、检索、Prompt 构建和回答生成流程。

- `model_client.py`：调用传入的客户端，提取回答并检查空内容。
- `app.py`：返回问题、答案和本次提供给模型的参考资料。
- 当前演示使用模拟客户端，答案为预设内容，不发送网络请求。
- 参考资料来自实际检索，但模拟答案不能用于验证模型回答质量。
- 当前共 42 项自动化测试通过。

运行演示：

    python app.py

运行全部测试：

    python -m unittest discover -s tests -v
