from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class ExperimentConfig:
    chunk_size: int
    chunk_overlap_pct: int
    top_k: int
    embedding_model: str
    retrieval_method: str

    @property
    def experiment_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def load_matrix(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_full_factorial(matrix: Dict) -> Iterable[ExperimentConfig]:
    factors = matrix["factors"]
    for chunk_size, overlap, top_k, embedding_model, retrieval_method in itertools.product(
        factors["chunk_size"],
        factors["chunk_overlap_pct"],
        factors["top_k"],
        factors["embedding_model"],
        factors["retrieval_method"],
    ):
        yield ExperimentConfig(
            chunk_size=int(chunk_size),
            chunk_overlap_pct=int(overlap),
            top_k=int(top_k),
            embedding_model=embedding_model,
            retrieval_method=retrieval_method,
        )


def write_manifest(matrix_path: Path, output_path: Path) -> List[Dict]:
    matrix = load_matrix(matrix_path)
    rows = [
        {"experiment_id": config.experiment_id, **asdict(config)}
        for config in iter_full_factorial(matrix)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
        handle.write("\n")
    return rows


def main() -> None:
    matrix_path = Path("evaluation/experiment_matrix.json")
    output_path = Path("evaluation/results/experiment_manifest.json")
    rows = write_manifest(matrix_path, output_path)
    print(f"Wrote {len(rows)} experiment configs to {output_path}")


if __name__ == "__main__":
    main()

