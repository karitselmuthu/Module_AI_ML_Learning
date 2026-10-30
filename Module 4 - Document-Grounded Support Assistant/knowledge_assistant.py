"""Ingest local documents, retrieve evidence, answer with citations, and evaluate RAG."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import HashingVectorizer

ROOT = Path(__file__).resolve().parent
DEFAULT_DOCS = ROOT / "sample_docs"
DEFAULT_INDEX = ROOT / "artifacts" / "knowledge_index"
DEFAULT_EVAL = ROOT / "eval" / "questions.jsonl"
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
FLAN = "google/flan-t5-small"
ABSTAIN = "I don't have enough evidence in the uploaded documents to answer that."
SUPPORTED = {".md", ".txt", ".pdf"}
STOPWORDS = set("a an and are as at be can do does for from how i in is it my of on or our the this to what when where which who with you your support ticket demo".split())


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOPWORDS


def read_documents(folder: Path) -> list[dict]:
    folder = folder.resolve()
    if not folder.is_dir():
        raise ValueError(f"Document folder does not exist: {folder}")
    pages = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.suffix.lower() not in SUPPORTED:
            continue
        relative = path.relative_to(folder).as_posix()
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(path)
            if reader.is_encrypted:
                raise ValueError(f"Encrypted PDF is not supported: {relative}")
            for page_number, page in enumerate(reader.pages, 1):
                content = (page.extract_text() or "").strip()
                if content:
                    pages.append({"source": relative, "page": page_number, "text": content})
        else:
            content = path.read_text(encoding="utf-8").strip()
            if content:
                pages.append({"source": relative, "page": None, "text": content})
    if not pages:
        raise ValueError(f"No readable .md, .txt, or text PDF documents in {folder}")
    return pages


def make_chunks(pages: list[dict], chunk_size: int = 500, overlap: int = 80) -> list[dict]:
    if chunk_size < 100 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be at least 100 and overlap smaller than chunk_size")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for page in pages:
        for number, text in enumerate(splitter.split_text(page["text"]), 1):
            chunks.append({"id": f"{page['source']}#p{page['page'] or 0}-c{number:02d}",
                           "source": page["source"], "page": page["page"], "text": text})
    return chunks


class Embeddings:
    def __init__(self, backend: str):
        if backend not in {"hash", "minilm"}:
            raise ValueError("embedding backend must be hash or minilm")
        self.backend = backend
        if backend == "hash":
            self.model = HashingVectorizer(n_features=512, alternate_sign=False, norm="l2", ngram_range=(1, 2))
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(MINILM)

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.backend == "hash":
            vectors = self.model.transform(texts).toarray().astype("float32")
        else:
            vectors = np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype="float32")
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        faiss.normalize_L2(vectors)
        return vectors


@lru_cache(maxsize=2)
def get_embeddings(backend: str) -> Embeddings:
    return Embeddings(backend)


def ingest(docs: Path = DEFAULT_DOCS, output: Path = DEFAULT_INDEX, *, backend: str = "hash",
           chunk_size: int = 500, overlap: int = 80, overwrite: bool = False) -> dict:
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"Index already exists in {output}; pass --overwrite or choose another --index")
    pages = read_documents(Path(docs))
    chunks = make_chunks(pages, chunk_size, overlap)
    encoder = get_embeddings(backend)
    vectors = encoder.encode([chunk["text"] for chunk in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    output.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output / "vectors.faiss"))
    (output / "chunks.json").write_text(json.dumps(chunks, indent=2) + "\n")
    source_hashes = {page["source"]: hashlib.sha256((Path(docs) / page["source"]).read_bytes()).hexdigest()
                     for page in pages}
    manifest = {"embedding_backend": backend, "embedding_model": MINILM if backend == "minilm" else "sklearn HashingVectorizer",
                "chunk_size": chunk_size, "overlap": overlap, "documents": len(source_hashes),
                "chunks": len(chunks), "source_sha256": source_hashes}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_index(folder: Path):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    chunks = json.loads((folder / "chunks.json").read_text())
    index = faiss.read_index(str(folder / "vectors.faiss"))
    if index.ntotal != len(chunks):
        raise ValueError("FAISS index and chunk metadata have different lengths")
    return manifest, chunks, index


def retrieve(question: str, folder: Path = DEFAULT_INDEX, k: int = 3) -> list[dict]:
    if not question.strip() or k < 1:
        raise ValueError("question must be nonempty and k must be positive")
    manifest, chunks, index = load_index(folder)
    vector = get_embeddings(manifest["embedding_backend"]).encode([question])
    scores, positions = index.search(vector, min(k, len(chunks)))
    return [{**chunks[int(position)], "score": round(float(score), 4)}
            for score, position in zip(scores[0], positions[0]) if position >= 0]


def extract_answer(question: str, hits: list[dict]) -> tuple[str, dict | None]:
    query_terms = tokens(question)
    candidates = []
    for hit in hits:
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", hit["text"]):
            sentence = sentence.strip("#* -\t")
            overlap = len(query_terms & tokens(sentence))
            # One shared word (for example, "next") is too weak for a grounded answer.
            if overlap >= 2:
                candidates.append((overlap + max(hit["score"], 0), sentence, hit))
    if not candidates:
        return ABSTAIN, None
    _, sentence, hit = max(candidates, key=lambda row: row[0])
    return sentence, {"source": hit["source"], "page": hit["page"],
                      "chunk_id": hit["id"], "score": hit["score"], "evidence": hit["text"]}


def flan_candidate(question: str, evidence: str) -> str:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch
    tokenizer = AutoTokenizer.from_pretrained(FLAN)
    model = AutoModelForSeq2SeqLM.from_pretrained(FLAN, use_safetensors=True)
    prompt = ("Answer the question using only the context. If it is not answered, say 'I don't know'.\n"
              f"Context: {evidence[:1400]}\nQuestion: {question}\nAnswer:")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


def answer(question: str, folder: Path = DEFAULT_INDEX, *, k: int = 3, mode: str = "extractive") -> dict:
    if mode not in {"extractive", "flan"}:
        raise ValueError("mode must be extractive or flan")
    hits = retrieve(question, folder, k)
    text, citation = extract_answer(question, hits)
    if citation is None:
        return {"status": "insufficient_evidence", "answer": ABSTAIN, "citations": [],
                "mode": mode, "retrieved": [{"source": h["source"], "score": h["score"]} for h in hits]}
    selected_mode = "extractive"
    if mode == "flan":
        candidate = flan_candidate(question, citation["evidence"])
        # Only publish a generated answer when it is a verbatim span of the retrieved source.
        if candidate and candidate.lower() in citation["evidence"].lower():
            text = candidate
            selected_mode = "flan_verified"
        else:
            selected_mode = "extractive_fallback"
    return {"status": "answered", "answer": text, "citations": [citation],
            "mode": selected_mode, "retrieved": [{"source": h["source"], "score": h["score"]} for h in hits]}


def evaluate(folder: Path = DEFAULT_INDEX, questions: Path = DEFAULT_EVAL, k: int = 3) -> dict:
    rows = [json.loads(line) for line in Path(questions).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Evaluation file is empty")
    results = []
    for case in rows:
        hits = retrieve(case["question"], folder, k)
        response = answer(case["question"], folder, k=k)
        gold = case.get("source")
        found_at = next((i + 1 for i, h in enumerate(hits) if h["source"] == gold), None) if gold else None
        grounded = (response["status"] == "answered" and bool(response["citations"])
                    and response["answer"].lower() in response["citations"][0]["evidence"].lower())
        expected = case.get("expected_phrase", "")
        results.append({"question": case["question"], "answerable": bool(gold),
                        "gold_source_rank": found_at, "top_source": hits[0]["source"] if hits else None,
                        "status": response["status"], "grounded": grounded,
                        "expected_phrase_found": expected.lower() in response["answer"].lower() if gold else None})
    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]
    summary = {"cases": len(results), "retrieval_recall_at_k": sum(r["gold_source_rank"] is not None for r in answerable) / len(answerable),
               "retrieval_mrr": sum(1 / r["gold_source_rank"] if r["gold_source_rank"] else 0 for r in answerable) / len(answerable),
               "answer_phrase_rate": sum(r["expected_phrase_found"] for r in answerable) / len(answerable),
               "grounding_rate": sum(r["grounded"] for r in answerable) / len(answerable),
               "abstention_rate": sum(r["status"] == "insufficient_evidence" for r in unanswerable) / len(unanswerable) if unanswerable else None,
               "results": results}
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ingest_parser = commands.add_parser("ingest")
    ingest_parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    ingest_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ingest_parser.add_argument("--embedding", choices=("hash", "minilm"), default="hash")
    ingest_parser.add_argument("--chunk-size", type=int, default=500)
    ingest_parser.add_argument("--overlap", type=int, default=80)
    ingest_parser.add_argument("--overwrite", action="store_true")
    ask_parser = commands.add_parser("ask")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ask_parser.add_argument("--mode", choices=("extractive", "flan"), default="extractive")
    ask_parser.add_argument("--k", type=int, default=3)
    eval_parser = commands.add_parser("evaluate")
    eval_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    eval_parser.add_argument("--questions", type=Path, default=DEFAULT_EVAL)
    eval_parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    if args.command == "ingest":
        result = ingest(args.docs, args.index, backend=args.embedding, chunk_size=args.chunk_size,
                        overlap=args.overlap, overwrite=args.overwrite)
    elif args.command == "ask":
        result = answer(args.question, args.index, k=args.k, mode=args.mode)
    else:
        result = evaluate(args.index, args.questions, args.k)
    print(json.dumps(result, indent=2))
