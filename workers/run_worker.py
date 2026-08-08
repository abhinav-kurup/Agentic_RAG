#!/usr/bin/env python3
"""
Start the DocuMind ingest worker.

Usage (from project root):
    python workers/run_worker.py

Or via RQ directly:
    rq worker documind:ingest --url redis://localhost:6379/0
"""
import logging
import os
import sys

# Ensure project root is on sys.path when run as a script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rq import Queue, SimpleWorker, Worker

from core.config import Config
from backend.queue.redis_client import get_redis_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("documind.worker")


def main() -> None:
    os.makedirs(Config.JOBS_DIR, exist_ok=True)
    os.makedirs(Config.DOCUMENTS_DIR, exist_ok=True)

    # Default Worker uses os.fork(); Windows needs in-process SimpleWorker.
    worker_cls = SimpleWorker if sys.platform == "win32" else Worker

    logger.info("Starting ingest worker (%s)", worker_cls.__name__)
    logger.info("  Redis URL  : %s", Config.REDIS_URL)
    logger.info("  Queue      : %s", Config.INGEST_QUEUE_NAME)
    logger.info("  Jobs dir   : %s", Config.JOBS_DIR)
    logger.info("  Image proc : %s", Config.ENABLE_IMAGE_PROCESSING)

    conn = get_redis_connection()

    queues = [Queue(Config.INGEST_QUEUE_NAME, connection=conn)]
    worker = worker_cls(queues, connection=conn, name="documind-ingest-worker-2")
    logger.info("Worker listening on queue '%s'...", Config.INGEST_QUEUE_NAME)
    worker.work(with_scheduler=False)


if __name__ == "__main__":
    main()
