"""只重排已保存的 /search 结果，不调用 API、不读取或修改索引。"""

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

from reranking import (
    MODEL_CACHE_DIR,
    MODEL_NAME,
    MODEL_REVISION,
    CrossEncoderReranker,
    RerankerUnavailableError,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "data" / "rerank_examples" / "mjo_zh_top10.json"


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("参数必须是正整数")
    return number


def load_candidates(file_path):
    # utf-8-sig 兼容 Windows 编辑器保存的带 BOM 文本。
    with Path(file_path).open("r", encoding="utf-8-sig") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, dict) or payload.get("retriever") != "embedding":
        raise ValueError("输入必须是保存的 embedding 模式 /search JSON 结果")
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("输入结果缺少非空 question")
    candidates = payload.get("references")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("输入结果缺少非空 references 列表")
    seen = set()
    for position, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            raise ValueError(f"第 {position} 个候选不是 JSON 对象")
        if type(candidate.get("rank")) is not int or candidate["rank"] != position:
            raise ValueError("候选必须保留 /search 原有的连续 rank 和顺序")
        if any(
            not isinstance(candidate.get(field), str) or not candidate[field].strip()
            for field in ("source", "path", "text")
        ):
            raise ValueError(f"第 {position} 个候选缺少来源、路径或正文")
        chunk_id = candidate.get("chunk_id")
        if type(chunk_id) is not int or chunk_id < 1:
            raise ValueError(f"第 {position} 个候选的 chunk_id 无效")
        if "page" in candidate:
            page = candidate["page"]
            if type(page) is not int or page < 1:
                raise ValueError(f"第 {position} 个候选的 page 无效")
        score = candidate.get("score")
        if type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError(f"第 {position} 个候选的 E5 分数无效")
        key = (candidate["path"], chunk_id)
        if key in seen:
            raise ValueError("输入结果含重复切片，请保留原始 /search 结果")
        seen.add(key)
    return question, candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="已保存的 embedding 模式 /search JSON 文件")
    parser.add_argument("--top-k", type=positive_int, default=3,
                        help="展示全文的最终片段数量，默认 3")
    parser.add_argument("--batch-size", type=positive_int, default=4,
                        help="每批计算的问题与片段对数，默认 4")
    parser.add_argument("--local-files-only", action="store_true",
                        help="只使用已有缓存，不联网检查或下载")
    args = parser.parse_args()

    try:
        question, candidates = load_candidates(args.input)
        if args.top_k > len(candidates):
            raise ValueError("top-k 不能大于输入候选数量")
        print("输入文件：", args.input.resolve())
        print("问题：", question)
        print(f"固定候选数：{len(candidates)}；最终展示数：{args.top_k}")
        print("这些是之前的真实检索结果；本次不重新运行 E5，也不启动服务。")
        print("\n重排前（E5 顺序）：")
        for candidate in candidates:
            print(f"原第 {candidate['rank']:2} 名 | 切片 {candidate['chunk_id']:2}"
                  f" | PDF页 {candidate.get('page', '-')}"
                  f" | E5分数 {candidate['score']:.6f}")

        print("\n准备加载重排模型：", MODEL_NAME)
        print("固定权重版本：", MODEL_REVISION)
        print("模型缓存目录：", MODEL_CACHE_DIR)
        if args.local_files_only:
            print("仅使用本地缓存；缓存不完整时将报错。", flush=True)
        else:
            print("首次运行会联网下载；主要权重约 471 MB，另有分词器等文件。",
                  flush=True)

        # 与正式接口复用组件；只有独立例子默认允许补下载模型。
        reranker = CrossEncoderReranker(
            local_files_only=args.local_files_only,
            batch_size=args.batch_size,
        )
        loading_started = perf_counter()
        reranker.load()
        print(f"模型加载用时（含可能的下载）：{perf_counter() - loading_started:.2f} 秒")
        inference_started = perf_counter()
        ranked = reranker.rerank(question, candidates, show_progress_bar=True)
        print(f"长度检查及重排用时：{perf_counter() - inference_started:.2f} 秒")
        print("\n重排后（不预设哪个片段应排第一）：")
        for new_rank, candidate in enumerate(ranked, start=1):
            print(f"新第 {new_rank:2} 名 | 原第 {candidate['rank']:2} 名"
                  f" | 切片 {candidate['chunk_id']:2}"
                  f" | PDF页 {candidate.get('page', '-')}"
                  f" | E5 {candidate['score']:.6f}"
                  f" | 重排分数 {candidate['rerank_score']:+.6f}")
        print(f"\n最终前 {args.top_k} 个片段的原文：")
        for new_rank, candidate in enumerate(ranked[:args.top_k], start=1):
            print(f"\n[{new_rank}] {candidate['source']}"
                  f" | PDF页 {candidate.get('page', '-')}"
                  f" | 切片 {candidate['chunk_id']}")
            print(candidate["text"])
        print("\n重排分数不是余弦相似度或正确率，不与 E5 分数比较大小。")
        print("本例只打印结果，不覆盖输入、不修改索引、不训练模型、不调用 DeepSeek。")
    except (ImportError, OSError, ValueError, RerankerUnavailableError) as error:
        parser.exit(1, f"\n运行未完成：{error}\n请保留报错排查；本例未修改现有索引。\n")


if __name__ == "__main__":
    main()
