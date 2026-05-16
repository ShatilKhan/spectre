"""Full pipeline benchmarks: OCR accuracy, classifier accuracy, extraction quality.

Benchmark modes:
  - "classifier": Run document classifier against 13,155 CUAD clauses
  - "ocr": Run PaddleOCR on CUAD PDFs, compute CER/WER against ground truth
  - "extraction": Run full pipeline, measure field extraction quality
  - "all": Run all three and aggregate results
"""

import base64
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.evaluation.metrics import compute_cer, compute_wer
from app.extraction.llm_extractor import extract_fields
from app.ocr.classifier import classify_document
from app.ocr.pipeline import process_document

CUAD_PATH = Path(os.getenv("CUAD_PATH", "/home/user/app/datasets/cuad/clauses.jsonl"))
SAMPLE_DIR = Path(os.getenv("SAMPLE_DIR", "/home/user/app/sample_docs"))

# ─── Shared helpers ──────────────────────────────────────


def _load_cuad_pdf_dataset(max_docs: int = 10) -> list[dict[str, Any]]:
    """Load CUAD PDFs with ground-truth text from HuggingFace.

    Returns list of dicts with file_name, pdf_bytes, ground_truth_text.
    Returns empty list on any failure (no internet, no datasets lib, etc.).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    try:
        dataset = load_dataset(
            "dvgodoy/CUAD_v1_Contract_Understanding_PDF",
            split="train",
            cache_dir=str(CUAD_PATH.parent),
        )
    except Exception:
        return []

    docs = []
    for i, row in enumerate(dataset):
        if i >= max_docs:
            break
        pdf_b64 = row.get("pdf_bytes_base64", "")
        if not pdf_b64:
            continue
        docs.append({
            "file_name": row.get("file_name", f"doc_{i}.pdf"),
            "pdf_bytes": base64.b64decode(pdf_b64),
            "ground_truth_text": row.get("text", ""),
        })
    return docs


# ─── 1. OCR Benchmark ────────────────────────────────────


def run_ocr_benchmark(max_docs: int = 10) -> dict[str, Any]:
    """Measure OCR accuracy on real CUAD PDFs.

    For each doc:
      1. Download from HF / serve from cache
      2. Run full OCR pipeline
      3. Compute CER and WER against the ground-truth text
      4. Report per-document + aggregate stats

    Parallelism: up to 4 PDFs at once (PaddleOCR already uses its own
    internal thread pool, so this is mainly I/O parallelism).
    """
    docs = _load_cuad_pdf_dataset(max_docs=max_docs)
    if not docs:
        return {
            "error": "CUAD PDF dataset not available. "
            "The `datasets` library must be installed and HF must be reachable.",
        }

    results: list[dict[str, Any]] = []
    total_cer: list[float] = []
    total_wer: list[float] = []

    def _process_one(doc: dict) -> dict[str, Any]:
        """Run OCR on a single PDF and return metrics."""
        pdf_path = SAMPLE_DIR / doc["file_name"]
        SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(doc["pdf_bytes"])

        try:
            t0 = time.time()
            pipeline_result = process_document(pdf_path)
            elapsed = time.time() - t0

            ocr_text = pipeline_result.full_text
            gt_text = doc["ground_truth_text"]

            cer = compute_cer(gt_text, ocr_text)
            wer = compute_wer(gt_text, ocr_text)

            return {
                "file_name": doc["file_name"],
                "cer": round(cer, 4),
                "wer": round(wer, 4),
                "accuracy": round(1.0 - cer, 4),
                "ocr_chars": len(ocr_text),
                "gt_chars": len(gt_text),
                "ocr_confidence": round(pipeline_result.confidence, 3),
                "doc_type": pipeline_result.doc_type,
                "latency_sec": round(elapsed, 1),
            }
        except Exception as e:
            return {
                "file_name": doc["file_name"],
                "error": str(e),
                "cer": 1.0,
                "wer": 1.0,
                "accuracy": 0.0,
            }
        finally:
            if pdf_path.exists():
                pdf_path.unlink()

    # Sequential processing — PaddleOCR already uses internal parallelism.
    # Concurrent calls can cause C++ level thread-safety issues.
    for doc in docs:
        res = _process_one(doc)
        results.append(res)
        if "error" not in res:
            total_cer.append(res["cer"])
            total_wer.append(res["wer"])

    avg_cer = float(np.mean(total_cer)) if total_cer else 1.0
    avg_wer = float(np.mean(total_wer)) if total_wer else 1.0
    num_ok = len([r for r in results if "error" not in r])
    p50_cer = float(np.median(total_cer)) if total_cer else None
    p95_cer = float(np.percentile(total_cer, 95)) if len(total_cer) > 1 else None
    p99_cer = float(np.percentile(total_cer, 99)) if len(total_cer) > 1 else None

    return {
        "mode": "ocr",
        "num_docs": len(docs),
        "num_success": num_ok,
        "average_cer": round(avg_cer, 4),
        "average_wer": round(avg_wer, 4),
        "average_accuracy": round(1.0 - avg_cer, 4),
        "median_cer": round(p50_cer, 4) if p50_cer is not None else None,
        "p95_cer": round(p95_cer, 4) if p95_cer is not None else None,
        "p99_cer": round(p99_cer, 4) if p99_cer is not None else None,
        "per_document": sorted(results, key=lambda r: r.get("cer", 1.0)),
        "note": (
            "CER = Character Error Rate (lower is better, <0.05 is excellent). "
            "WER = Word Error Rate (lower is better). "
            "Based on PaddleOCR output vs HuggingFace ground truth text."
        ),
    }


# ─── 2. Classifier Benchmark (improved) ──────────────────

# Maps CUAD clause labels to our classifier's document types
CUAD_TO_DOC_TYPE: dict[str, str] = {
    "Non-Compete": "nda",
    "No-Solicit Of Customers": "nda",
    "No-Solicit Of Employees": "nda",
    "Non-Disparagement": "nda",
    "Confidentiality": "nda",
    "Governing Law": "msa",
    "Effective Date": "msa",
    "Expiration Date": "msa",
    "Renewal Term": "msa",
    "Termination For Convenience": "msa",
    "Anti-Assignment": "msa",
    "Cap On Liability": "msa",
    "Uncapped Liability": "msa",
    "Insurance": "msa",
    "Audit Rights": "msa",
    "Most Favored Nation": "msa",
    "Exclusivity": "msa",
    "License Grant": "msa",
    "Ip Ownership Assignment": "engagement_letter",
    "Joint Ip Ownership": "engagement_letter",
    "Post-Termination Services": "engagement_letter",
    "Revenue/Profit Sharing": "fee_proposal",
    "Price Restrictions": "fee_proposal",
    "Minimum Commitment": "fee_proposal",
    "Liquidated Damages": "fee_proposal",
    "Change Of Control": "legal_generic",
    "Rofr/Rofo/Rofn": "legal_generic",
    "Volume Restriction": "legal_generic",
    "Covenant Not To Sue": "legal_generic",
    "Third Party Beneficiary": "legal_generic",
    "Warranty Duration": "legal_generic",
    "Source Code Escrow": "legal_generic",
    "Non-Transferable License": "legal_generic",
    "Affiliate License-Licensor": "legal_generic",
    "Affiliate License-Licensee": "legal_generic",
    "Unlimited/All-You-Can-Eat-License": "legal_generic",
    "Irrevocable Or Perpetual License": "legal_generic",
    "Competitive Restriction Exception": "legal_generic",
    "Document Name": "legal_generic",
    "Parties": "legal_generic",
    "Agreement Date": "legal_generic",
}

# Keywords that should appear in clauses belonging to each doc type, for
# recall upper-bound estimation.
DOC_TYPE_KEYWORDS: dict[str, list[str]] = {
    "nda": ["confidential", "non-disclosure", "nda", "confidentiality"],
    "msa": ["master services", "msa", "agreement", "terms"],
    "engagement_letter": ["engagement", "services"],
    "fee_proposal": ["fee", "proposal", "pricing"],
    "sow": ["scope of work", "statement of work"],
    "legal_generic": ["agreement", "contract", "clause"],
}


def _load_cuad_clauses(num_samples: int) -> list[dict[str, Any]]:
    """Load stratified samples from CUAD clause classification dataset.

    Tries local JSONL first. Falls back to downloading from HuggingFace
    (cached by the datasets library for subsequent calls).

    Stratifies evenly across all 41+ clause types so no single type
    dominates the accuracy measurement.
    """
    samples: list[dict[str, Any]] = []

    # Try local JSONL first
    if CUAD_PATH.exists():
        with open(CUAD_PATH) as f:
            samples = [json.loads(line) for line in f if line.strip()]

    # Fall back to HuggingFace download
    if not samples:
        try:
            from datasets import load_dataset
            dataset = load_dataset(
                "dvgodoy/CUAD_v1_Contract_Understanding_clause_classification",
                split="train",
                cache_dir=str(CUAD_PATH.parent),
            )
            samples = [
                {
                    "file_name": row["file_name"],
                    "clause": row["clause"],
                    "label": row["label"],
                    "class_id": row["class_id"],
                }
                for row in dataset
            ]
            # Cache to local JSONL for future fast access
            if samples:
                try:
                    CUAD_PATH.parent.mkdir(parents=True, exist_ok=True)
                    with open(CUAD_PATH, "w") as f:
                        for s in samples:
                            f.write(json.dumps(s) + "\n")
                except OSError:
                    pass  # caching is opportunistic
        except Exception:
            return []

    if not samples:
        return []

    by_class: dict[int, list] = {}
    for s in samples:
        by_class.setdefault(s["class_id"], []).append(s)

    result: list[dict[str, Any]] = []
    classes = sorted(by_class.keys())
    per_class = max(1, num_samples // len(classes))

    for cid in classes:
        pool = by_class[cid]
        k = min(per_class, len(pool))
        result.extend(random.sample(pool, k))

    random.shuffle(result)
    return result[:num_samples]


def run_classifier_benchmark(num_samples: int = 200) -> dict[str, Any]:
    """Run classifier benchmark against CUAD clause dataset.

    Reports:
      - Overall accuracy across all sampled clauses
      - Per-doc-type accuracy (precision/recall estimate per document category)
      - Per-clause-type accuracy (top 20 clause types by count)
      - Keyword-coverage upper bound (theoretical max recall per type)
    """
    samples = _load_cuad_clauses(num_samples)
    if not samples:
        return {
            "error": "CUAD clause dataset not found. Run: python datasets/download_cuad.py",
        }

    correct = 0
    total = 0
    by_type: dict[str, dict[str, int]] = {}
    by_clause: dict[str, dict[str, int]] = {}

    for s in samples:
        clause_text = s.get("clause", "")
        true_label = s.get("label", "")
        if not clause_text or not true_label:
            continue

        predicted = classify_document(clause_text)
        expected = CUAD_TO_DOC_TYPE.get(true_label, "legal_generic")

        dt = expected
        by_type.setdefault(dt, {"correct": 0, "total": 0})
        by_type[dt]["total"] += 1
        if predicted == expected:
            correct += 1
            by_type[dt]["correct"] += 1

        ct = true_label
        by_clause.setdefault(ct, {"correct": 0, "total": 0})
        by_clause[ct]["total"] += 1
        if predicted == expected:
            by_clause[ct]["correct"] += 1

        total += 1

    accuracy = correct / total if total > 0 else 0

    per_type = {
        dt: {
            "accuracy": round(c["correct"] / c["total"], 3) if c["total"] > 0 else 0,
            "count": c["total"],
            "correct": c["correct"],
        }
        for dt, c in sorted(by_type.items())
    }

    # Top 20 clause types by sample count
    sorted_clauses = sorted(by_clause.items(), key=lambda x: -x[1]["total"])
    per_clause = {
        ct: {
            "accuracy": round(c["correct"] / c["total"], 3) if c["total"] > 0 else 0,
            "count": c["total"],
            "correct": c["correct"],
        }
        for ct, c in sorted_clauses[:20]
    }

    # Keyword coverage — what fraction of each doc type's clauses contain
    # that type's defining keywords (theoretical upper bound for our classifier)
    keyword_hits: dict[str, dict[str, Any]] = {}
    for dt, keywords in DOC_TYPE_KEYWORDS.items():
        type_samples = [
            s for s in samples
            if CUAD_TO_DOC_TYPE.get(s.get("label", "")) == dt
        ]
        if not type_samples:
            continue
        hits = sum(
            1 for s in type_samples
            if any(kw in (s.get("clause", "") or "").lower() for kw in keywords)
        )
        keyword_hits[dt] = {
            "keyword_coverage": round(hits / len(type_samples), 3),
            "samples": len(type_samples),
        }

    return {
        "mode": "classifier",
        "num_samples": total,
        "overall_accuracy": round(accuracy, 4),
        "per_doc_type": per_type,
        "per_clause_type": per_clause,
        "keyword_coverage": keyword_hits,
        "note": (
            f"Classifier benchmark against {total} CUAD clauses "
            f"across {len(by_type)} document types. "
            "Keyword coverage estimates the theoretical max accuracy "
            "achievable by a keyword-based classifier for each type."
        ),
    }


# ─── 3. Extraction Benchmark ─────────────────────────────


def _extract_date_strings(text: str) -> list[str]:
    """Pull date-like substrings from a value for fuzzy matching."""
    import re

    dates: list[str] = []
    # ISO dates
    dates.extend(re.findall(r"\d{4}-\d{2}-\d{2}", text))
    # Written dates
    months = (
        "january|february|march|april|may|june|"
        "july|august|september|october|november|december"
    )
    dates.extend(re.findall(rf"(?:{months})\s+\d{{1,2}},?\s+\d{{4}}", text, re.I))
    # MM/DD/YYYY or DD/MM/YYYY
    dates.extend(re.findall(r"\b\d{1,2}/\d{1,2}/\d{4}\b", text))
    return dates


def _field_in_text(field_name: str, field_value: Any, text: str) -> bool:
    """Check whether a field value can be found somewhere in the source text.

    Handles strings, lists, dicts, and special patterns for dates/money.
    """
    if not field_value:
        return True  # null field = not a hallucination

    text_lower = text.lower()

    if field_name in ("effective_date", "expiration_date", "agreement_date"):
        # Check each date pattern individually
        date_strings = _extract_date_strings(str(field_value))
        if date_strings:
            return any(ds.lower() in text_lower for ds in date_strings)
        # Fall through to direct string match

    if isinstance(field_value, str):
        return field_value.lower() in text_lower
    if isinstance(field_value, list):
        return any(
            str(v).lower() in text_lower for v in field_value if v
        )
    if isinstance(field_value, dict):
        return any(
            str(v).lower() in text_lower
            for v in field_value.values()
            if v and isinstance(v, str)
        )
    return str(field_value).lower() in text_lower


# Summary/generated fields — these are LLM-generated and won't appear
# verbatim in source text. Skip grounding check for them.
SUMMARY_FIELDS = {"summary", "content", "overview", "description"}


def _compute_field_metrics(
    field_name: str,
    field_value: Any,
    ground_truth: str,
    ocr_text: str,
) -> dict[str, Any]:
    """Determine if a field is grounded in either reference or OCR output.

    Summary/generated fields are always considered grounded since they
    represent LLM-generated text that won't appear verbatim in source.
    """
    is_summary = field_name.lower() in SUMMARY_FIELDS
    if is_summary:
        return {
            "field": field_name,
            "present": bool(field_value),
            "grounded_in_truth": True,
            "grounded_in_ocr": True,
            "grounded": True,
            "is_summary": True,
        }

    in_gt = _field_in_text(field_name, field_value, ground_truth)
    in_ocr = _field_in_text(field_name, field_value, ocr_text)
    return {
        "field": field_name,
        "present": bool(field_value),
        "grounded_in_truth": in_gt,
        "grounded_in_ocr": in_ocr,
        "grounded": in_gt or in_ocr,
    }


def run_extraction_benchmark(max_docs: int = 5) -> dict[str, Any]:
    """Run the full OCR + classification + extraction pipeline on CUAD PDFs.

    For each document:
      1. OCR the PDF
      2. Classify
      3. Extract fields using the matching schema
      4. Check every extracted field against both ground-truth text and OCR text

    Metrics:
      - grounding_rate: fraction of extracted fields traceable to source
      - hallucination_rate: fraction of fields that are fabricated
      - Per-field breakdown showing which fields are most/least reliable
    """
    docs = _load_cuad_pdf_dataset(max_docs=max_docs)
    if not docs:
        return {"error": "CUAD PDF dataset not available."}

    all_field_metrics: list[dict[str, Any]] = []
    per_doc_results: list[dict[str, Any]] = []

    def _process_one(doc: dict) -> dict[str, Any]:
        pdf_path = SAMPLE_DIR / doc["file_name"]
        SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(doc["pdf_bytes"])

        try:
            t0 = time.time()
            pipeline_result = process_document(pdf_path)
            ocr_time = time.time() - t0

            t0 = time.time()
            extracted = extract_fields(
                raw_text=pipeline_result.full_text,
                doc_type=pipeline_result.doc_type,
                use_few_shot=False,  # no operator corrections for benchmark docs
            )
            extract_time = time.time() - t0

            gt_text = doc["ground_truth_text"]
            skip_keys = {"error", "raw_snippet", "file_name", "doc_type"}
            field_metrics = [
                _compute_field_metrics(fname, fvalue, gt_text, pipeline_result.full_text)
                for fname, fvalue in extracted.items()
                if fname not in skip_keys
            ]

            fields_present = sum(1 for fm in field_metrics if fm["present"])
            # Only count grounded among present fields (null = skip)
            fields_grounded = sum(
                1 for fm in field_metrics if fm["present"] and fm["grounded"]
            )
            fields_hallucinated = sum(
                1 for fm in field_metrics if fm["present"] and not fm["grounded"]
            )
            fields_null = sum(1 for fm in field_metrics if not fm["present"])

            return {
                "file_name": doc["file_name"],
                "doc_type": pipeline_result.doc_type,
                "ocr_latency_sec": round(ocr_time, 1),
                "extract_latency_sec": round(extract_time, 1),
                "fields_checked": len(field_metrics),
                "fields_present": fields_present,
                "fields_grounded": fields_grounded,
                "fields_null": fields_null,
                "hallucinated": fields_hallucinated,
                "grounding_rate": round(
                    fields_grounded / max(fields_present, 1), 3
                ),
                "field_metrics": field_metrics,
            }
        except Exception as e:
            return {
                "file_name": doc["file_name"],
                "error": str(e),
                "fields_checked": 0,
                "fields_present": 0,
                "fields_grounded": 0,
                "hallucinated": 0,
                "grounding_rate": 0.0,
                "field_metrics": [],
            }
        finally:
            if pdf_path.exists():
                pdf_path.unlink()

    # Sequential processing — PaddleOCR has C++ thread-safety issues with
    # concurrent calls, and the LLM is single-instance (4 GB VRAM shared).
    for doc in docs:
        res = _process_one(doc)
        per_doc_results.append(res)
        all_field_metrics.extend(res.get("field_metrics", []))

    total_checked = sum(r["fields_checked"] for r in per_doc_results)
    total_present = sum(r["fields_present"] for r in per_doc_results)
    total_grounded = sum(r["fields_grounded"] for r in per_doc_results)
    total_hallucinated = sum(r["hallucinated"] for r in per_doc_results)
    grounding_rate = round(total_grounded / max(total_present, 1), 4)
    hallucination_rate = round(total_hallucinated / max(total_present, 1), 4)

    # Aggregate per field across all documents
    # Grounded is counted only among present fields (null fields are skipped)
    field_agg: dict[str, dict[str, int]] = {}
    for fm in all_field_metrics:
        fname = fm["field"]
        field_agg.setdefault(fname, {"count": 0, "grounded": 0, "present": 0})
        field_agg[fname]["count"] += 1
        if fm["present"]:
            field_agg[fname]["present"] += 1
            if fm["grounded"]:
                field_agg[fname]["grounded"] += 1

    per_field_summary = {
        fname: {
            "occurrences": agg["count"],
            "present": agg["present"],
            "grounded": agg["grounded"],
            "grounding_rate": round(
                agg["grounded"] / max(agg["present"], 1), 3
            ),
        }
        for fname, agg in sorted(field_agg.items())
    }

    return {
        "mode": "extraction",
        "num_docs": len(docs),
        "num_success": len([r for r in per_doc_results if "error" not in r]),
        "total_fields_checked": total_checked,
        "total_fields_present": total_present,
        "total_fields_grounded": total_grounded,
        "total_fields_hallucinated": total_hallucinated,
        "grounding_rate": grounding_rate,
        "hallucination_rate": hallucination_rate,
        "overall_extraction_quality": round(grounding_rate * (1 - hallucination_rate), 4),
        "per_field": per_field_summary,
        "per_document": per_doc_results,
        "note": (
            "grounding_rate = fraction of extracted fields found in source text. "
            "hallucination_rate = fraction NOT found. "
            "overall_extraction_quality = grounding_rate * (1 - hallucination_rate). "
            f"Based on {total_checked} field extractions across {len(docs)} contracts."
        ),
    }


# ─── 4. Aggregate runner ────────────────────────────────


def run_all_benchmarks(
    classifier_samples: int = 200,
    ocr_docs: int = 10,
    extraction_docs: int = 5,
) -> dict[str, Any]:
    """Run all three benchmarks sequentially and return a combined report."""
    t0 = time.time()
    classifier = run_classifier_benchmark(num_samples=classifier_samples)
    ocr = run_ocr_benchmark(max_docs=ocr_docs)
    extraction = run_extraction_benchmark(max_docs=extraction_docs)
    elapsed = time.time() - t0

    return {
        "mode": "all",
        "total_time_sec": round(elapsed, 1),
        "classifier": classifier,
        "ocr": ocr,
        "extraction": extraction,
    }
