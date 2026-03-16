from __future__ import annotations

import json
import logging
import os
from typing import Dict, Iterable, List, Mapping, Sequence

from openai import OpenAI

from .dimensions import DEFAULT_DIMENSIONS
from .extraction import extract_code_dimension as extract_code_dimension_call
from .extraction import extract_paper_dimension as extract_paper_dimension_call
from .models import (
    AnalysisDetail,
    CodeFile,
    ComparisonRecord,
    DimensionEvidence,
    MatchDecision,
)
from .rag import EmbeddingIndex, build_code_chunks, build_paper_chunks, looks_like_code

LOG = logging.getLogger("codebot_flow")


def _compact_json(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(obj)


def _parse_json_or_fallback(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        return {"raw": text}


class LLMComparisonStrategy:
    """Default LLM-based strategy with optional staged flow."""

    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-5.1",
        dimensions: Mapping[str, str] | None = None,
        reasoning_effort: str = "medium",
    ):
        self.client = client
        self.model = model
        self.dimensions = dict(dimensions or DEFAULT_DIMENSIONS)
        self.reasoning_effort = reasoning_effort

    # --- staged pieces ---
    def extract_paper_dimension(self, analysis: AnalysisDetail, dimension: str, definition: str, paper_text: str) -> DimensionEvidence:
        return extract_paper_dimension_call(
            analysis,
            dimension,
            definition,
            paper_text,
            client=self.client,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
        )

    def extract_code_dimension(
        self,
        analysis: AnalysisDetail,
        dimension: str,
        definition: str,
        code_files: Sequence[CodeFile],
        paper_value: str | None = None,
        repo_tree: str | None = None,
        project_yaml: str | None = None,
    ) -> DimensionEvidence:
        return extract_code_dimension_call(
            analysis,
            dimension,
            definition,
            code_files,
            paper_value=paper_value,
            repo_tree=repo_tree,
            project_yaml=project_yaml,
            client=self.client,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
        )

    def judge_dimension(
        self,
        analysis: AnalysisDetail,
        dimension: str,
        definition: str,
        paper_value: str,
        code_value: str,
    ) -> MatchDecision:
        LOG.info("API: judge dimension | analysis_id=%s | dimension=%s", analysis.analysis_id, dimension)
        prompt = (
            "Below, you will see a description of a specific analysis conducted within an academic paper, a specific dimension of analysis to be considered, specific quotes from the academic paper that relate to this analyis-dimension pairing, and specific code snippets from the associated analysis code related to this dimension-analysis pairing.\n"
            "Your task is to determine whether the code implementation matches the paper description for this analysis-dimension pairing. You may make one of three judgements: 'match', 'mismatch', or 'unknown'.\n"
            "'match' indicates that the code implementation appears to align with the paper description for the analysis-dimension pairing. 'mismatch' indicates that the code implementation appears to differ from the paper description for the analysis-dimension pairing. 'unknown' indicates that it is unclear whether the code implementation matches the paper description for the analysis-dimension pairing (e.g., due to insufficient information in either source).\n"
            "Return STRICT JSON with keys: status, explanation (<=2 sentences), evidence.\n"
            "status refers to the match/mismatch/unknown judgement. explanation refers to a brief rationale for the judgement. evidence refers to specific supporting information from the paper and code that informed the judgement (e.g., direct quotes, file names, line numbers, etc.).\n\n"
            f"Here is the brief description of the analysis of interest: {_compact_json(analysis.__dict__)}\n\n"
            f"Here is the dimension of interest: {dimension}, which is defined as {definition}\n\n"
            f"Here are the contents extracted from the paper in relation to this: {paper_value}\n\n"
            f"Here are the contents extracted from the code in relation to this: {code_value}\n\n"
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            messages=[
                {"role": "system", "content": "You are a strict adjudicator returning JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = resp.choices[0].message.content or ""
        payload = _parse_json_or_fallback(raw)
        status = payload.get("status") or payload.get("match_status") or "unknown"
        explanation = payload.get("explanation") or payload.get("reason", "")
        evidence = payload.get("evidence") or {}
        return MatchDecision(
            paper_id=analysis.analysis_id,
            analysis_id=analysis.analysis_id,
            dimension=dimension,
            status=status,
            explanation=explanation,
            paper_value=paper_value,
            code_value=code_value,
            evidence=evidence,
            llm_raw=raw,
        )

    # --- combined legacy flow ---
    # Note from Jamie: this is not currently used in the main pipeline, but retained for potential future use
    def combined_dimension(
        self,
        analysis: AnalysisDetail,
        dimension: str,
        definition: str,
        paper_text: str,
        code_files: Sequence[CodeFile],
        repo_tree: str | None = None,
        project_yaml: str | None = None,
    ) -> MatchDecision:
        LOG.info("API: combined comparison | analysis_id=%s | dimension=%s", analysis.analysis_id, dimension)
        prompt = (
            "Below, you will see the content of an academic paper, a description of a specific analysis conducted within that paper, "
            "a specific analysis dimension to consider, repository structure metadata, and the full contents of relevant code files.\n"
            "Your task is to (1) extract direct quotes from the paper relevant to the analysis + dimension, "
            "(2) extract direct code snippets from the codebase relevant to the analysis + dimension, and "
            "(3) judge whether the code matches the paper along this dimension.\n"
            "Return STRICT JSON with keys:\n"
            "{\n"
            "  \"status\": \"match\"|\"mismatch\"|\"unknown\",\n"
            "  \"paper_value\": \"direct quote(s) from the paper\",\n"
            "  \"code_value\": \"direct code snippet(s)\",\n"
            "  \"explanation\": \"<=2 sentences\",\n"
            "  \"evidence\": {\n"
            "     \"paper_span\": \"supporting quote(s)\",\n"
            "     \"location\": \"where in the paper the quotes appear (section/page/line if possible)\",\n"
            "     \"code_path\": \"file path\",\n"
            "     \"code_lines\": \"line range or approximate lines\"\n"
            "  }\n"
            "}\n"
            "paper_value should be the most relevant direct quote(s) for this dimension. paper_span can be the same as paper_value "
            "or a longer surrounding excerpt. code_value must include actual code (not only comments). If you cannot find an "
            "appropriate paper or code excerpt, use an empty string for that field and consider status=\"unknown\".\n\n"
            f"Here is the brief description of the analysis: {_compact_json(analysis.__dict__)}\n\n"
            f"Here is the dimension to compare: {dimension}, which is defined as {definition}\n\n"
            "Here is the paper text:\n" + paper_text + "\n\n"
            "Here is the repository tree (if available):\n" + (repo_tree or "<not provided>") + "\n\n"
            "Here is the repository project.yaml (if available):\n" + (project_yaml or "<missing>") + "\n\n"
            "Here are the code files (path + content), as a JSON list:\n" + _compact_json([cf.__dict__ for cf in code_files])
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            messages=[
                {"role": "system", "content": "You are a rigorous extraction and comparison assistant. Return only JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = resp.choices[0].message.content or ""
        payload = _parse_json_or_fallback(raw)
        status = payload.get("status") or "unknown"
        explanation = payload.get("explanation") or payload.get("reason", "")
        evidence = payload.get("evidence") or {}
        paper_value = payload.get("paper_value") or ""
        code_value = payload.get("code_value") or ""
        return MatchDecision(
            paper_id=analysis.analysis_id,
            analysis_id=analysis.analysis_id,
            dimension=dimension,
            status=status,
            explanation=explanation,
            paper_value=paper_value,
            code_value=code_value,
            evidence=evidence,
            llm_raw=raw,
        )


class RAGComparisonStrategy(LLMComparisonStrategy):
    """RAG-first extraction strategy with code-snippet appropriateness screening."""

    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-5.1",
        dimensions: Mapping[str, str] | None = None,
        reasoning_effort: str = "medium",
        embedding_model: str = os.getenv("CODEBOT_EMBEDDING_MODEL", "text-embedding-3-large"),
        paper_top_k: int = 5,
        code_top_k: int = 10,
        code_keep_k: int = 5,
    ):
        super().__init__(client=client, model=model, dimensions=dimensions, reasoning_effort=reasoning_effort)
        self.embedding_model = embedding_model
        self.paper_top_k = paper_top_k
        self.code_top_k = code_top_k
        self.code_keep_k = code_keep_k
        self._paper_index: EmbeddingIndex | None = None
        self._code_index: EmbeddingIndex | None = None
        self._paper_signature: tuple[int, int] | None = None
        self._code_signature: tuple[tuple[str, int], ...] | None = None

    def prepare_context(self, *, paper_text: str | None = None, code_files: Sequence[CodeFile] | None = None) -> None:
        if paper_text is not None:
            signature = (len(paper_text), hash(paper_text))
            if self._paper_signature != signature:
                chunks = build_paper_chunks(paper_text)
                self._paper_index = EmbeddingIndex.build(
                    client=self.client,
                    model=self.embedding_model,
                    chunks=chunks,
                )
                self._paper_signature = signature
        if code_files is not None:
            signature = tuple((cf.path, len(cf.content)) for cf in code_files)
            if self._code_signature != signature:
                chunks = build_code_chunks(code_files)
                self._code_index = EmbeddingIndex.build(
                    client=self.client,
                    model=self.embedding_model,
                    chunks=chunks,
                )
                self._code_signature = signature

    @staticmethod
    def _query_text(analysis: AnalysisDetail, dimension: str, definition: str) -> str:
        return (
            f"Analysis description: {analysis.brief_description}\n"
            f"Dimension: {dimension}\n"
            f"Definition: {definition}\n"
            "Find direct textual/code evidence relevant to this analysis-dimension pairing."
        )

    def extract_paper_dimension(self, analysis: AnalysisDetail, dimension: str, definition: str, paper_text: str) -> DimensionEvidence:
        if self._paper_index is None:
            self.prepare_context(paper_text=paper_text)
        if not self._paper_index or self._paper_index.is_empty():
            return super().extract_paper_dimension(analysis, dimension, definition, paper_text)

        query = self._query_text(analysis, dimension, definition)
        hits = self._paper_index.query(query, top_k=self.paper_top_k)
        quotes = [h.chunk.text for h in hits if h.chunk.text.strip()]
        locations = [h.chunk.location for h in hits if h.chunk.location]
        paper_value = "\n\n".join(quotes).strip()
        evidence = {
            "paper_span": paper_value,
            "location": "; ".join(locations),
        }
        return DimensionEvidence(
            paper_id=analysis.analysis_id,  # overwritten by orchestrator
            analysis_id=analysis.analysis_id,
            dimension=dimension,
            value=paper_value,
            source="paper",
            evidence=evidence,
            llm_raw=None,
        )

    @staticmethod
    def _non_comment_code_lines(snippet: str) -> list[str]:
        lines: list[str] = []
        for line in snippet.splitlines():
            raw = line.rstrip()
            stripped = raw.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            lines.append(raw)
        if lines:
            return lines
        return [line.rstrip() for line in snippet.splitlines() if line.strip()]

    def _fallback_keep_indices(self, candidates: Sequence[dict]) -> list[int]:
        fallback: list[int] = []
        for i, cand in enumerate(candidates):
            text = str(cand.get("snippet", ""))
            if looks_like_code(text):
                fallback.append(i)
            if len(fallback) >= self.code_keep_k:
                break
        if fallback:
            return fallback
        return [0] if candidates else []

    def _fallback_informative_lines(self, selected: Sequence[dict]) -> list[str]:
        lines: list[str] = []
        for item in selected:
            snippet = str(item.get("snippet", ""))
            lines.extend(self._non_comment_code_lines(snippet))
        return lines

    def _screen_code_candidates(
        self,
        analysis: AnalysisDetail,
        dimension: str,
        definition: str,
        paper_value: str | None,
        candidates: Sequence[dict],
    ) -> tuple[list[int], list[str], str, dict]:
        prompt = (
            "You are screening candidate code snippets retrieved via semantic search.\n"
            "Task A: choose which snippets to keep.\n"
            "Task B: from those snippets, select only the most informative raw code lines for this analysis+dimension.\n"
            "Important constraints:\n"
            "- informative lines must be exact lines copied from the candidates (verbatim)\n"
            "- no added text, no paraphrase, no comments from you, no line numbers\n"
            "- you may only remove irrelevant lines\n"
            "Return STRICT JSON with exactly two top-level elements and no other keys:\n"
            "{\n"
            "  \"snippet_selection\": {\"keep_indices\": [int]},\n"
            "  \"informative_code_lines\": [{\"index\": int, \"lines\": [str]}]\n"
            "}\n"
            f"Keep at most {self.code_keep_k} indices.\n\n"
            f"Analysis: {_compact_json(analysis.__dict__)}\n"
            f"Dimension: {dimension}\n"
            f"Definition: {definition}\n"
            f"Related paper quote(s): {paper_value or 'n/a'}\n\n"
            "Candidates (JSON list):\n"
            f"{_compact_json(candidates)}"
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            messages=[
                {"role": "system", "content": "You are a strict selector. Return only JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = resp.choices[0].message.content or ""
        payload = _parse_json_or_fallback(raw)

        snippet_selection = payload.get("snippet_selection", {}) if isinstance(payload, dict) else {}
        idxs = snippet_selection.get("keep_indices", []) if isinstance(snippet_selection, dict) else []
        keep: list[int] = []
        if isinstance(idxs, list):
            for item in idxs:
                if isinstance(item, int) and 0 <= item < len(candidates):
                    keep.append(item)
        keep = keep[: self.code_keep_k]
        if not keep:
            keep = self._fallback_keep_indices(candidates)

        allowed_lines_by_idx: dict[int, set[str]] = {}
        for idx in keep:
            snippet = str(candidates[idx].get("snippet", ""))
            allowed_lines_by_idx[idx] = {line.rstrip() for line in snippet.splitlines() if line.strip()}

        info = payload.get("informative_code_lines", []) if isinstance(payload, dict) else []
        informative_lines: list[str] = []
        if isinstance(info, list):
            for entry in info:
                if not isinstance(entry, dict):
                    continue
                idx = entry.get("index")
                if not isinstance(idx, int) or idx not in keep:
                    continue
                lines = entry.get("lines")
                if not isinstance(lines, list):
                    continue
                allowed = allowed_lines_by_idx.get(idx, set())
                for line in lines:
                    if not isinstance(line, str):
                        continue
                    normalized = line.rstrip()
                    if not normalized.strip():
                        continue
                    if normalized in allowed:
                        informative_lines.append(normalized)

        return keep, informative_lines, raw, payload if isinstance(payload, dict) else {}

    def extract_code_dimension(
        self,
        analysis: AnalysisDetail,
        dimension: str,
        definition: str,
        code_files: Sequence[CodeFile],
        paper_value: str | None = None,
        repo_tree: str | None = None,
        project_yaml: str | None = None,
    ) -> DimensionEvidence:
        if self._code_index is None:
            self.prepare_context(code_files=code_files)
        if not self._code_index or self._code_index.is_empty():
            return super().extract_code_dimension(
                analysis,
                dimension,
                definition,
                code_files,
                paper_value=paper_value,
                repo_tree=repo_tree,
                project_yaml=project_yaml,
            )

        query = self._query_text(analysis, dimension, definition)
        hits = self._code_index.query(query, top_k=self.code_top_k)
        candidates: list[dict] = []
        for i, hit in enumerate(hits):
            chunk = hit.chunk
            candidates.append(
                {
                    "index": i,
                    "score": round(hit.score, 4),
                    "path": chunk.path,
                    "line_range": f"{chunk.start_line}-{chunk.end_line}" if chunk.start_line and chunk.end_line else "",
                    "snippet": chunk.text,
                }
            )
        if not candidates:
            return DimensionEvidence(
                paper_id=analysis.analysis_id,
                analysis_id=analysis.analysis_id,
                dimension=dimension,
                value="",
                source="code",
                evidence={"code_path": "", "code_lines": ""},
                llm_raw=None,
            )

        keep_idxs, informative_lines, selector_raw, selector_payload = self._screen_code_candidates(
            analysis,
            dimension,
            definition,
            paper_value,
            candidates,
        )
        selected = [candidates[i] for i in keep_idxs if 0 <= i < len(candidates)]
        if not informative_lines:
            informative_lines = self._fallback_informative_lines(selected)
        code_value = "\n".join(informative_lines).strip()

        paths = [str(item.get("path", "")) for item in selected if item.get("path")]
        ranges = [str(item.get("line_range", "")) for item in selected if item.get("line_range")]
        line_ranges = "\n".join(ranges)
        evidence = {
            "code_path": "\n".join(paths),
            "code_lines": line_ranges,
            "code_line_ranges": line_ranges,
            "informative_code_lines": informative_lines,
            "screener_output": selector_payload,
        }
        return DimensionEvidence(
            paper_id=analysis.analysis_id,  # overwritten by orchestrator
            analysis_id=analysis.analysis_id,
            dimension=dimension,
            value=code_value,
            source="code",
            evidence=evidence,
            llm_raw=selector_raw,
        )

    def combined_dimension(
        self,
        analysis: AnalysisDetail,
        dimension: str,
        definition: str,
        paper_text: str,
        code_files: Sequence[CodeFile],
        repo_tree: str | None = None,
        project_yaml: str | None = None,
    ) -> MatchDecision:
        paper_ev = self.extract_paper_dimension(analysis, dimension, definition, paper_text)
        code_ev = self.extract_code_dimension(
            analysis,
            dimension,
            definition,
            code_files,
            paper_value=paper_ev.value,
            repo_tree=repo_tree,
            project_yaml=project_yaml,
        )
        decision = self.judge_dimension(analysis, dimension, definition, paper_ev.value, code_ev.value)
        merged_evidence = decision.evidence if isinstance(decision.evidence, dict) else {}
        if not merged_evidence.get("paper_span"):
            merged_evidence["paper_span"] = (paper_ev.evidence or {}).get("paper_span") or paper_ev.value
        if not merged_evidence.get("location"):
            merged_evidence["location"] = (paper_ev.evidence or {}).get("location", "")
        if not merged_evidence.get("code_path"):
            merged_evidence["code_path"] = (code_ev.evidence or {}).get("code_path", "")
        if not merged_evidence.get("code_lines"):
            merged_evidence["code_lines"] = (code_ev.evidence or {}).get("code_lines", "")
        if not merged_evidence.get("code_line_ranges"):
            merged_evidence["code_line_ranges"] = (code_ev.evidence or {}).get("code_line_ranges", "")
        if not merged_evidence.get("informative_code_lines"):
            merged_evidence["informative_code_lines"] = (code_ev.evidence or {}).get("informative_code_lines", [])
        if not merged_evidence.get("screener_output"):
            merged_evidence["screener_output"] = (code_ev.evidence or {}).get("screener_output", {})
        decision.evidence = merged_evidence
        return decision


# ---------- Orchestration helpers ----------

def run_staged(
    analyses: Sequence[AnalysisDetail],
    paper_text: str,
    code_files: Sequence[CodeFile],
    *,
    strategy: LLMComparisonStrategy,
    paper_id: str,
    dimensions: Mapping[str, str] | None = None,
    repo_tree: str | None = None,
    project_yaml: str | None = None,
) -> tuple[List[DimensionEvidence], List[DimensionEvidence], List[MatchDecision]]:
    dims = dimensions or strategy.dimensions
    if hasattr(strategy, "prepare_context"):
        try:
            strategy.prepare_context(paper_text=paper_text, code_files=code_files)  # type: ignore[attr-defined]
        except Exception:
            LOG.exception("RAG context preparation failed; continuing with on-demand extraction.")

    paper_evidence: List[DimensionEvidence] = []
    total_analyses = len(analyses)
    total_dims = len(dims)
    for a_idx, analysis in enumerate(analyses, start=1):
        for d_idx, (dim, definition) in enumerate(dims.items(), start=1):
            LOG.info("Stage: paper evidence | analysis %s/%s | dimension %s/%s (%s)", a_idx, total_analyses, d_idx, total_dims, dim)
            ev = strategy.extract_paper_dimension(analysis, dim, definition, paper_text)
            ev.paper_id = paper_id
            paper_evidence.append(ev)

    code_evidence: List[DimensionEvidence] = []
    paper_lookup = {(e.analysis_id, e.dimension): e for e in paper_evidence}
    for a_idx, analysis in enumerate(analyses, start=1):
        for d_idx, (dim, definition) in enumerate(dims.items(), start=1):
            LOG.info("Stage: code evidence | analysis %s/%s | dimension %s/%s (%s)", a_idx, total_analyses, d_idx, total_dims, dim)
            paper_ev = paper_lookup.get((analysis.analysis_id, dim))
            ev = strategy.extract_code_dimension(
                analysis,
                dim,
                definition,
                code_files,
                paper_value=paper_ev.value if paper_ev else None,
                repo_tree=repo_tree,
                project_yaml=project_yaml,
            )
            ev.paper_id = paper_id
            code_evidence.append(ev)

    matches: List[MatchDecision] = []
    code_lookup = {(e.analysis_id, e.dimension): e for e in code_evidence}
    for a_idx, analysis in enumerate(analyses, start=1):
        for d_idx, (dim, definition) in enumerate(dims.items(), start=1):
            LOG.info("Stage: adjudication | analysis %s/%s | dimension %s/%s (%s)", a_idx, total_analyses, d_idx, total_dims, dim)
            paper_ev = paper_lookup.get((analysis.analysis_id, dim))
            code_ev = code_lookup.get((analysis.analysis_id, dim))
            paper_val = paper_ev.value if paper_ev else ""
            code_val = code_ev.value if code_ev else ""
            decision = strategy.judge_dimension(analysis, dim, definition, paper_val, code_val)
            merged_evidence = decision.evidence if isinstance(decision.evidence, dict) else {}
            if not merged_evidence.get("paper_span"):
                merged_evidence["paper_span"] = ((paper_ev.evidence or {}).get("paper_span") if paper_ev else "") or paper_val
            if not merged_evidence.get("location"):
                merged_evidence["location"] = ((paper_ev.evidence or {}).get("location") if paper_ev else "") or ""
            if not merged_evidence.get("code_path"):
                merged_evidence["code_path"] = ((code_ev.evidence or {}).get("code_path") if code_ev else "") or ""
            if not merged_evidence.get("code_lines"):
                merged_evidence["code_lines"] = ((code_ev.evidence or {}).get("code_lines") if code_ev else "") or ""
            if not merged_evidence.get("code_line_ranges"):
                merged_evidence["code_line_ranges"] = ((code_ev.evidence or {}).get("code_line_ranges") if code_ev else "") or ""
            if not merged_evidence.get("informative_code_lines"):
                merged_evidence["informative_code_lines"] = ((code_ev.evidence or {}).get("informative_code_lines") if code_ev else []) or []
            if not merged_evidence.get("screener_output"):
                merged_evidence["screener_output"] = ((code_ev.evidence or {}).get("screener_output") if code_ev else {}) or {}
            decision.evidence = merged_evidence
            decision.paper_id = paper_id
            matches.append(decision)

    return paper_evidence, code_evidence, matches


def run_combined(
    analyses: Sequence[AnalysisDetail],
    paper_text: str,
    code_files: Sequence[CodeFile],
    *,
    strategy: LLMComparisonStrategy,
    paper_id: str,
    dimensions: Mapping[str, str] | None = None,
    repo_tree: str | None = None,
    project_yaml: str | None = None,
) -> List[MatchDecision]:
    dims = dimensions or strategy.dimensions
    if hasattr(strategy, "prepare_context"):
        try:
            strategy.prepare_context(paper_text=paper_text, code_files=code_files)  # type: ignore[attr-defined]
        except Exception:
            LOG.exception("RAG context preparation failed; continuing with on-demand extraction.")
    decisions: List[MatchDecision] = []
    total_analyses = len(analyses)
    total_dims = len(dims)
    for a_idx, analysis in enumerate(analyses, start=1):
        for d_idx, (dim, definition) in enumerate(dims.items(), start=1):
            LOG.info("Stage: combined comparison | analysis %s/%s | dimension %s/%s (%s)", a_idx, total_analyses, d_idx, total_dims, dim)
            decision = strategy.combined_dimension(
                analysis,
                dim,
                definition,
                paper_text,
                code_files,
                repo_tree=repo_tree,
                project_yaml=project_yaml,
            )
            decision.paper_id = paper_id
            decisions.append(decision)
    return decisions


def to_comparison_records(
    analyses: Sequence[AnalysisDetail],
    decisions: Iterable[MatchDecision],
) -> List[ComparisonRecord]:
    brief_lookup = {a.analysis_id: a.brief_description for a in analyses}
    records: List[ComparisonRecord] = []
    for dec in decisions:
        records.append(
            ComparisonRecord(
                paper_id=dec.paper_id,
                analysis_id=dec.analysis_id,
                brief_description=brief_lookup.get(dec.analysis_id, ""),
                dimension=dec.dimension,
                paper_value=dec.paper_value,
                code_value=dec.code_value,
                match_status=dec.status,
                explanation=dec.explanation,
                evidence=dec.evidence,
                llm_raw=dec.llm_raw,
            )
        )
    return records


__all__ = [
    "LLMComparisonStrategy",
    "RAGComparisonStrategy",
    "run_staged",
    "run_combined",
    "to_comparison_records",
]
