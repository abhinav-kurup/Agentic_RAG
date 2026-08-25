"""Native tool-calling surface. The LLM decides when to retrieve."""
from __future__ import annotations

import ast
import asyncio
import json
import logging
import math
import operator
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agents.retrieval_base import BaseRetrievalAgent
from core.config import Config
from core.llm import get_llm

logger = logging.getLogger(__name__)

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}


def _safe_eval(expression: str) -> str:
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        return f"Invalid expression: {exc}"

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = _eval(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            args = [_eval(arg) for arg in node.args]
            if name == "sqrt":
                return math.sqrt(args[0])
            if name == "log":
                return math.log(*args)
            if name == "log10":
                return math.log10(args[0])
            if name in {"abs", "round", "min", "max"}:
                return getattr(__builtins__, name)(*args)
        raise ValueError("Only arithmetic is allowed")

    try:
        return str(_eval(tree))
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


def pack_docs(docs: List[Dict[str, Any]], limit: int = 6) -> str:
    chunks = []
    for doc in docs[:limit]:
        meta = doc.get("metadata") or {}
        chunks.append(
            {
                "source": meta.get("source"),
                "page": meta.get("page_number"),
                "score": doc.get("score"),
                "content": (doc.get("content") or "")[:1500],
            }
        )
    return json.dumps({"hit_count": len(docs), "chunks": chunks}, ensure_ascii=False)


class SearchArgs(BaseModel):
    query: str = Field(description="Focused search string for the PDFs.")
    source: str = Field(default="", description="Optional PDF filename to restrict search.")


class ReadPagesArgs(BaseModel):
    source: str = Field(description="PDF filename from list_documents or a prior hit.")
    page: int = Field(description="1-based page number.")


class ExtractArgs(BaseModel):
    instruction: str = Field(description="What table or numbers to pull from working memory.")


class CalcArgs(BaseModel):
    expression: str = Field(description="Arithmetic only, e.g. '(0.91-0.84)/0.84'.")


class DocumentToolkit(BaseRetrievalAgent):
    def __init__(self, vector_store):
        super().__init__(vector_store=vector_store)
        self._evidence_getter = lambda: []

    def bind_evidence_getter(self, getter):
        self._evidence_getter = getter

    async def search_docs(self, query: str, source: str = "") -> str:
        docs = await self._retrieve_single_query(query, query, k=10)
        if source:
            docs = [d for d in docs if (d.get("metadata") or {}).get("source") == source]
        if docs:
            try:
                docs = await self._post_process_and_select(query, docs, [docs], [query])
            except Exception:
                logger.exception("Rerank/select failed; using raw hits")
        logger.info("search_docs q=%r source=%r hits=%s", query, source, len(docs))
        return pack_docs(docs)

    async def list_documents(self) -> str:
        names = await asyncio.to_thread(self.vector_store.get_processed_documents)
        return json.dumps({"documents": names or []})

    async def read_pages(self, source: str, page: int) -> str:
        docs = await asyncio.to_thread(
            self.vector_store.hybrid_search,
            f"{source} page {page}",
            8,
            {"source": source, "page_number": page},
        )
        if not docs:
            docs = await asyncio.to_thread(
                self.vector_store.hybrid_search,
                f"{source} page {page}",
                8,
                {"source": source},
            )
            docs = [d for d in docs if (d.get("metadata") or {}).get("page_number") == page]
        return pack_docs(docs, limit=8)

    async def extract_tables(self, instruction: str) -> str:
        evidence = self._evidence_getter() or []
        if not evidence:
            return json.dumps({"error": "working memory is empty; call search_docs first"})
        context = "\n\n".join(
            f"[{(d.get('metadata') or {}).get('source')} p.{(d.get('metadata') or {}).get('page_number')}]\n"
            f"{d.get('content', '')}"
            for d in evidence[:8]
        )
        llm = get_llm(Config.EXTRACTION_MODEL, temperature=0.0)
        resp = await llm.ainvoke(
            [
                HumanMessage(
                    content=(
                        "Extract only the requested table or numbers. Markdown tables. "
                        "If absent, say so.\n\n"
                        f"Instruction: {instruction}\n\nEvidence:\n{context}"
                    )
                )
            ]
        )
        return json.dumps({"extracted": getattr(resp, "content", str(resp))})

    def as_langchain_tools(self) -> List[StructuredTool]:
        return [
            StructuredTool.from_function(
                coroutine=self.search_docs,
                name="search_docs",
                description="Hybrid search + rerank over uploaded PDFs. Use a focused query.",
                args_schema=SearchArgs,
            ),
            StructuredTool.from_function(
                coroutine=self.list_documents,
                name="list_documents",
                description="List indexed PDF filenames.",
            ),
            StructuredTool.from_function(
                coroutine=self.read_pages,
                name="read_pages",
                description="Read one PDF page when a search snippet is truncated.",
                args_schema=ReadPagesArgs,
            ),
            StructuredTool.from_function(
                coroutine=self.extract_tables,
                name="extract_tables",
                description="Extract tables/numbers from chunks already in working memory.",
                args_schema=ExtractArgs,
            ),
            StructuredTool.from_function(
                func=_safe_eval,
                name="calculator",
                description="Arithmetic after numbers have been retrieved from documents.",
                args_schema=CalcArgs,
            ),
        ]
