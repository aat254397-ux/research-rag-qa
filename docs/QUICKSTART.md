# 启动与交付指南

本项目是本机、单服务进程的科研文献 RAG 应用。以下先区分“已有环境继续使用”和“第一次准备环境”，避免重复下载模型或覆盖当前文献库。

## 已有环境：直接继续使用

在项目目录、已激活 `.venv` 的终端中启动服务：

```powershell
python -m uvicorn api:api --reload --host 127.0.0.1 --port 8000
```

若服务已经在另一个终端运行，不要重复启动。打开：

- 网页工作台：http://127.0.0.1:8000/
- 接口调试文档：http://127.0.0.1:8000/docs

已有索引从硬盘恢复，无需再次加载文献或重建 E5。首次检索会按需加载本地模型；后续请求在同一服务进程中复用。网页“只检索”在本地运行 E5 和可选重排，“生成答案”还会调用 DeepSeek 并产生费用。

## 第一次准备环境

以下命令在 `simple-document-qa` 项目目录中执行。已验证的开发环境为 Windows PowerShell、Python 3.10.19；本节是依据现有代码整理的启动路径，尚未在另一台全新机器重装验证。

### 1. 安装两组依赖

仅没有虚拟环境时创建：

```powershell
python -m venv .venv
```

直接使用项目虚拟环境的解释器安装，无需依赖当前终端是否已激活环境：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-embedding.txt
```

第一组是服务与基础测试依赖；第二组提供本地 E5 和重排模型所需的 Sentence Transformers 等依赖。它们是补充关系，不是二选一。

### 2. 准备本地 DeepSeek 配置

仅当 `.env` 不存在时复制模板，避免覆盖已有密钥：

```powershell
if (-not (Test-Path -LiteralPath ".env")) {
    Copy-Item -LiteralPath ".env.example" -Destination ".env"
}
```

用编辑器在本地填写 `DEEPSEEK_API_KEY` 和 `DEEPSEEK_MODEL`。当前默认模型为 `deepseek-flash`，`DEEPSEEK_BASE_URL` 默认指向 DeepSeek 官网。不要把真实密钥写入模板、README、截图或提交记录。只运行基础自动化测试不需要真实 API 密钥。

#### 使用阿里云百炼调用 DeepSeek

学生优惠券属于阿里云账户，调用必须经过百炼才能产生可由该券抵扣的账单。先在优惠券详情中核对适用商品、订单类型与有效期；支持“阿里云按量付费账单”表示可抵扣符合商品条件的按量费用。百炼官方确认云工开物 300 元抵扣金可直接抵扣阿里直供模型的后付费推理费用，无需先购买资源包或节省计划；模型处于活动折扣价期间，其按量调用不能使用该券。选择模型时核对“阿里直供”及当前活动价格，详见 [百炼账单与抵扣金 FAQ](https://help.aliyun.com/zh/model-studio/bill-query-and-cost-management) 和 [优惠券使用方式](https://help.aliyun.com/zh/user-center/how-to-use-coupons)。

在百炼控制台选择华北2（北京），获取同一地域的 API Key 和业务空间 ID，在本地 `.env` 中更新三项。例如目标型号确认为 `deepseek-v4.1-flash-1017` 时：

```dotenv
DEEPSEEK_API_KEY=在本地填写百炼的真实密钥
DEEPSEEK_MODEL=deepseek-v4.1-flash-1017
DEEPSEEK_BASE_URL=https://你的业务空间ID.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
```

上面的密钥和业务空间 ID 都是占位文字，需要在本地替换。字段仍命名为 `DEEPSEEK_API_KEY`，填入的是百炼创建的密钥；密钥、接入地址和模型名必须属于同一套服务配置。官方文档确认该型号支持项目问答所用的 Responses API，也支持独立工具示例使用的 Chat Completions API；若换其他型号，需重新确认其接口支持情况。参考 [获取 API Key](https://help.aliyun.com/zh/model-studio/get-api-key) 和 [DeepSeek 调用说明](https://help.aliyun.com/zh/model-studio/deepseek-api)。

修改 `.env` 后，在原服务终端按 Ctrl+C，再执行本页的 Uvicorn 启动命令。程序优先使用进程中已有的环境变量，仅刷新网页或等待代码自动重载不足以保证读取新配置。重启后已有索引仍从硬盘恢复，无需重建。通过原网页提交一次问题，随后在阿里云账单及优惠券抵扣明细核对实际计费结果；免费额度尚未用尽时可能优先消耗免费额度，未发生券抵扣不代表配置失败。券耗尽或过期后，符合条件的调用仍可能继续按量计费。

发送真实问答前，服务终端会显示“模型请求目标：host=…；model=…”，用于核对实际客户端指向。百炼地址应为上述 `maas.aliyuncs.com` 域名，DeepSeek 官网地址则是 `api.deepseek.com`；日志不包含密钥。2026-10-26 用户清理旧配置并重新测试，终端同时显示百炼域名、模型 `deepseek-v4.1-flash-1017` 以及 `/ask` 返回 200，网页正常展示答案与来源，已确认百炼调用成功。学生券抵扣情况单独通过账单核对，不以接口成功代替计费验证。

### 3. 首次准备两个模型缓存

这一步需要网络与足够的硬盘空间。服务端只读取已有缓存，不会自行补下载。已有完整缓存时无需重复运行。

先按服务组件指定的模型名、固定 revision 和缓存目录准备 E5 权重；这样不依赖在线示例未来默认指向的版本。以下为一条完整命令：

```powershell
.\.venv\Scripts\python.exe -c "from sentence_transformers import SentenceTransformer; from embedding_retrieval import MODEL_NAME, MODEL_REVISION, MODEL_CACHE_DIR; SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION, device='cpu', cache_folder=str(MODEL_CACHE_DIR), local_files_only=False, trust_remote_code=False)"
```

再复用已有重排组件的公开加载方法准备重排权重，不读取论文或候选文件，也不执行重排推理：

```powershell
.\.venv\Scripts\python.exe -c "from reranking import CrossEncoderReranker; CrossEncoderReranker(local_files_only=False).load()"
```

上述两个命令沿用组件内固定的模型版本、CPU 设置和项目缓存目录 `data/model_cache/`，不需要另写下载器，也不执行文本编码或重排推理。它们会加载权重到各自的命令进程，命令结束后释放该进程的模型内存；以后启动服务时从硬盘缓存再次加载。若想观察编码结果，可再单独运行 embedding_demo.py，但它不是此处准备指定版本权重的前提。

注意：`rerank_demo.py` 默认读取 `data/rerank_examples/mjo_zh_top10.json`，这是原学习环境的私有检索记录，已经加入忽略规则，不会随仓库分发。因此新环境不能把“不带参数运行这个脚本”作为准备重排模型的前提。若要运行独立重排示例，应先保存自己的 embedding 模式 `/search` 完整结果，再用 `--input` 指定文件。

### 4. 启动服务，加载自己的资料

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:api --reload --host 127.0.0.1 --port 8000
```

打开首页后：

1. 在“加载 / 更换文献”中输入本机资料的完整路径，一行一个文件，然后提交。
2. 建立 E5 语义索引；此步骤在本地编码并保存，不调用 DeepSeek。
3. 查看“只检索”返回的原文，或点击“生成答案”进行真实问答。

加载操作是整体替换当前文献库，不是追加。已有环境想保留当前论文时，不要为了复现指南再次执行替换。仓库不附带个人论文、索引、模型权重或私有候选记录；换一篇资料后，切片数、排名和答案也会不同。

## 自动化回归与真实问答分开验证

在第二个终端、项目目录中运行：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

现有测试使用受控模型客户端、手工向量、示例文档和临时目录，检查应用逻辑、接口、持久化、排序与页面资源，不需要真实付费模型调用或下载权重。接口测试使用独立的 TestClient 应用，不占用运行服务的 8000 端口，也不覆盖正式论文索引。

2026-10-25 用户实际运行结果为 80 项测试、OK。这不等同于真实模型问答质量评测，也不表示已验证所有新机器或生产部署环境。之后的 API 错误诊断改动新增两项测试，尚待用户回归；可先运行 `python -m unittest discover -s tests -p "test_deepseek_api.py" -v` 核对本次相关接口行为。

## 演示和提交前

- 按 [项目展示与讲解](PROJECT_SHOWCASE.md) 演示当前版本，不需要再次下载模型或替换已验证资料。
- 索引、模型缓存和私有候选记录已列入 `.gitignore`；它们不会自动成为可复现仓库的一部分，需要使用者自行准备。
- 用户自行检查 Git 文件清单，确认不包含 `.env`、真实密钥、私人论文或缓存。忽略规则不能自动移除过去已被 Git 跟踪的敏感文件。
- 保留当前已验证版本的提交记录；本指南不自动执行安装、下载、测试、模型调用或 Git 操作。
