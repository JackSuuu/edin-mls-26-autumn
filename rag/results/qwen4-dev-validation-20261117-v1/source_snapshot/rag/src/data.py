from __future__ import annotations

import csv
import json
from pathlib import Path


def load_jsonl(path: str | Path):
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class CourseData:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

        corpus_path = self.data_dir / "corpus.jsonl"
        rows = load_jsonl(corpus_path)
        self.records = {}
        for row in rows:
            rid = str(row["record_id"])
            text = str(row["text"])
            if rid in self.records:
                raise ValueError(f"duplicate record_id: {rid}")
            self.records[rid] = text

    def record_text(self, record_id: str) -> str:
        return self.records[record_id]

    def load_audio_manifest(self):
        return load_jsonl(self.data_dir / "student_audio_manifest.jsonl")

    def load_qrels(self, split: str):
        path = self.data_dir / f"qrels_{split}.tsv"
        qrels = {}
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                qid = str(row["query_id"])
                rid = str(row["record_id"])
                score = float(row.get("score", 1))
                if score > 0:
                    qrels.setdefault(qid, {})[rid] = score
        return qrels

    def audio_items(self, split: str):
        return [
            row for row in self.load_audio_manifest()
            if row["split"] == split
        ]
