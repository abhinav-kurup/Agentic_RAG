from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    ADVERSARIAL = "adversarial"


class QueryType(str, Enum):
    FACTOID = "factoid"
    MULTI_HOP = "multi_hop"
    SUMMARIZATION = "summarization"
    COMPARISON = "comparison"
    CITATION_VERIFICATION = "citation_verification"
    BROAD_SEMANTIC = "broad_semantic"


@dataclass
class ExpectedSource:
    source_document: str
    page_number: int
    citation: str
    relevant_chunk_ids: List[str] = field(default_factory=list)
    relevance_grade: int = 3

    def __post_init__(self) -> None:
        self.page_number = int(self.page_number)
        self.relevance_grade = int(self.relevance_grade)


@dataclass
class BenchmarkItem:
    id: str
    question: str
    ground_truth_answer: str
    expected_sources: List[ExpectedSource]
    difficulty_level: DifficultyLevel
    query_type: QueryType
    expected_route: str = "document"
    expected_tools: List[str] = field(default_factory=list)
    expected_agent_path: List[str] = field(
        default_factory=lambda: ["router", "retrieval", "extraction", "analysis"]
    )
    answer_keywords: List[str] = field(default_factory=list)
    unacceptable_answers: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.expected_sources = [
            src if isinstance(src, ExpectedSource) else ExpectedSource(**src)
            for src in self.expected_sources
        ]
        self.difficulty_level = DifficultyLevel(self.difficulty_level)
        self.query_type = QueryType(self.query_type)


@dataclass
class RetrievedContext:
    content: str
    source_document: Optional[str] = None
    page_number: Optional[int] = None
    citation: Optional[str] = None
    chunk_id: Optional[str] = None
    score: Optional[float] = None

    def __post_init__(self) -> None:
        if self.page_number is not None:
            self.page_number = int(self.page_number)
        if self.score is not None:
            self.score = float(self.score)


@dataclass
class PredictionItem:
    id: str
    question: str
    answer: str
    citations: List[str] = field(default_factory=list)
    retrieved_contexts: List[RetrievedContext] = field(default_factory=list)
    route: Optional[str] = None
    selected_tools: List[str] = field(default_factory=list)
    agent_path: List[str] = field(default_factory=list)
    workflow_completed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.retrieved_contexts = [
            ctx if isinstance(ctx, RetrievedContext) else RetrievedContext(**ctx)
            for ctx in self.retrieved_contexts
        ]
