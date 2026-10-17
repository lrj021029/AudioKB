"""
工具模块，负责提供 reranker 相关的辅助能力。
"""
from FlagEmbedding import FlagReranker
from pathlib import Path

from app.shared.config.reranker_config import reranker_config
from app.shared.runtime.logger import logger

_reranker_model: FlagReranker | None = None


def _resolve_reranker_path(configured_path: str) -> str:
    model_path = Path(configured_path).expanduser()
    if (model_path / "config.json").is_file():
        return str(model_path)

    cached_models = sorted(
        candidate.parent
        for candidate in model_path.glob("*/*/config.json")
        if candidate.is_file()
    )
    if len(cached_models) == 1:
        resolved_path = cached_models[0]
        logger.info(f"从模型缓存根目录解析到 Reranker：{resolved_path}")
        return str(resolved_path)
    if not cached_models:
        raise FileNotFoundError(
            f"Reranker 模型配置不存在：{model_path}/config.json，"
            "请将 BGE_RERANKER_LARGE 指向模型目录或仅包含一个模型的缓存根目录"
        )
    raise ValueError(
        f"Reranker 缓存目录包含多个模型，请将 BGE_RERANKER_LARGE 指向明确的模型目录："
        f"{model_path}"
    )


def get_reranker_model() -> FlagReranker:
    """
    获取重排模型单例对象。

    Returns:
        FlagReranker: 初始化完成的重排模型实例。
    """
    global _reranker_model
    if _reranker_model is None:
        logger.info("开始初始化重排模型")
        _reranker_model = FlagReranker(
            model_name_or_path=_resolve_reranker_path(reranker_config.bge_reranker_large),
            device=reranker_config.bge_reranker_device,
            use_fp16=reranker_config.bge_reranker_fp16,
        )
        logger.success("重排模型初始化成功")
    return _reranker_model
