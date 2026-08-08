import logging
from functools import lru_cache

import redis
from rq import Queue

from core.config import Config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_redis_connection() -> redis.Redis:
    # protocol=2: RESP2 only. Needed for Redis < 6 (e.g. old Windows 3.x) that
    # reject the HELLO handshake redis-py 5+ sends by default.
    conn = redis.from_url(Config.REDIS_URL, decode_responses=False, protocol=2)
    conn.ping()
    logger.info("Connected to Redis at %s", Config.REDIS_URL)
    return conn


@lru_cache(maxsize=1)
def get_ingest_queue() -> Queue:
    return Queue(
        Config.INGEST_QUEUE_NAME,
        connection=get_redis_connection(),
        default_timeout=Config.INGEST_JOB_TIMEOUT_SECONDS,
    )


def check_redis() -> tuple[bool, str]:
    try:
        get_redis_connection().ping()
        queue = get_ingest_queue()
        return True, f"connected (queue={queue.name}, depth={queue.count})"
    except Exception as e:
        return False, str(e)
