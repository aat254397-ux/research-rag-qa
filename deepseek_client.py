import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, OpenAI


def model_service_error_detail(error):
    """保留可用于定位的状态和已知错误码，不回显上游正文或密钥。"""
    if isinstance(error, APIConnectionError):
        return "模型服务调用失败：无法连接模型接口，请检查 API 地址、网络和代理设置。"

    code_hints = {
        "InvalidApiKey": "密钥校验失败，请确认使用百炼密钥，且地域、业务空间和 API 地址匹配。",
        "invalid_api_key": "密钥校验失败，请确认密钥与当前模型服务及 API 地址匹配。",
        "Arrearage": "上游报告账户计费异常，请核对密钥所属账号的账单与可用额度；优惠券不是现金余额。",
        "AllocationQuota.FreeTierOnly": "免费额度已耗尽且开启了用完即停，请在百炼核对该开关与后续计费设置。",
        "AccessDenied.Unpurchased": "模型服务尚未开通，请在百炼控制台确认服务开通状态。",
        "Model.AccessDenied": "当前密钥或业务空间没有该模型的调用权限，请检查模型授权。",
        "Workspace.AccessDenied": "当前密钥没有该业务空间的访问权限，请核对业务空间与 API 地址。",
        "ModelNotFound": "模型不存在或不可访问，请核对百炼模型 ID、所在地域及开通状态。",
        "model_not_found": "模型不存在或不可访问，请核对百炼模型 ID、所在地域及开通状态。",
        "InvalidParameter": "上游拒绝了请求参数，请核对所选模型是否支持当前接口及非流式调用。",
        "invalid_parameter_error": "上游拒绝了请求参数，请核对模型开通状态、接口及非流式调用支持。",
        "insufficient_quota": "上游报告可用额度不足，请在模型服务控制台核对账户和额度状态。",
    }
    status_hints = {
        400: "上游拒绝了请求参数，请核对模型名、接口协议及该模型的参数要求。",
        401: "鉴权失败，请核对密钥、地域、业务空间及 API 地址；修改配置后需完整重启服务。",
        402: "上游报告账户余额或额度不足，请先核对账单和优惠券适用范围。",
        403: "上游拒绝访问，请核对模型权限、服务开通状态及账户可用额度。",
        404: "上游未找到模型或接口，请核对模型 ID 和 API 地址，并确认模型支持 Responses API。",
        429: "上游限制了请求频率或额度，请查看控制台限流与额度状态，稍后重试。",
    }

    body = error.body if isinstance(error.body, dict) else {}
    payload = body.get("error")
    if not isinstance(payload, dict):
        payload = body
    code = error.code or payload.get("code")
    # 仅展示白名单中的固定错误码；未知字符串可能包含请求内容。
    known_code = code if isinstance(code, str) and code in code_hints else None
    status = getattr(error, "status_code", None)
    status = status if type(status) is int and 100 <= status <= 599 else None
    facts = []
    if status is not None:
        facts.append(f"上游 HTTP {status}")
    if known_code:
        facts.append(known_code)
    hint = code_hints.get(known_code) or status_hints.get(status)
    if hint is None:
        hint = "上游服务暂时不可用，请检查服务状态、账户和配置。"
    suffix = f"（{'；'.join(facts)}）" if facts else ""
    return f"模型服务调用失败{suffix}。{hint}"


def create_deepseek_client():
    # 从项目目录读取配置；已有环境变量优先
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(env_path, override=False)

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    model = os.getenv(
        "DEEPSEEK_MODEL",
        "deepseek-flash",
    ).strip()
    base_url = os.getenv(
        "DEEPSEEK_BASE_URL",
        "https://api.deepseek.com",
    ).strip()

    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY must not be empty")

    if not model:
        raise ValueError("DEEPSEEK_MODEL must not be empty")

    if not base_url:
        raise ValueError("DEEPSEEK_BASE_URL must not be empty")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=60.0,
        max_retries=0,
    )

    # 记录实际客户端的请求目标，便于核对服务商；不记录密钥或完整 URL。
    logged_model = (
        model if re.fullmatch(r"deepseek[-a-zA-Z0-9._]{0,100}", model)
        else "[其他模型，名称略]"
    )
    logging.getLogger("uvicorn.error").info(
        "模型请求目标：host=%s；model=%s",
        client.base_url.host,
        logged_model,
    )

    return client, model
