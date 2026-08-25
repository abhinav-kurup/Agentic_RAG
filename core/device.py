import logging

from core.config import Config

logger = logging.getLogger(__name__)


def cuda_available() -> bool:
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return True
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            return True
    except Exception:
        pass
    return False


def torch_device() -> str:
    """Device for PyTorch models (embeddings). Not CTranslate2."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def whisper_device_and_compute() -> tuple:
    requested = (Config.WHISPER_DEVICE or "auto").strip().lower()
    compute = (Config.WHISPER_COMPUTE_TYPE or "").strip()
    if requested in ("", "auto"):
        if cuda_available():
            device = "cuda"
            compute = compute or "float16"
        else:
            device = "cpu"
            compute = compute or "int8"
    elif requested == "cuda":
        device = "cuda"
        compute = compute or "float16"
    else:
        device = requested
        compute = compute or "int8"
    logger.info("Voice compute: whisper device=%s compute=%s", device, compute)
    return device, compute
