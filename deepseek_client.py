import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


def create_deepseek_client():
    # 从项目目录读取配置；已有环境变量优先
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(env_path, override=False)

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    model = os.getenv(
        "DEEPSEEK_MODEL",
        "deepseek-flash",
    ).strip()

    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY must not be empty")

    if not model:
        raise ValueError("DEEPSEEK_MODEL must not be empty")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=60.0,
        max_retries=0,
    )

    return client, model
