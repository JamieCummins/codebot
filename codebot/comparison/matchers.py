import re
from typing import Iterable, List, Tuple

from codebot.models import CodeAnalysisIR, MatchEdge, PaperAnalysisIR, RepoFile
from codebot.utils import normalize_space, take

# (name, regex, family, formula_capture_group)
MODEL_PATTERNS: List[Tuple[str, str, str, int | None]] = [
    ("glmer_binom", r"glmer\s*\(\s*([^)]*),\s*family\s*=\s*binomial", "logistic", 1),
    ("glm_binom", r"glm\s*\(\s*([^)]*),\s*family\s*=\s*binomial", "logistic", 1),
    ("coxph", r"coxph\s*\(\s*([^)]*)\)", "cox", 1),
    ("t_test", r"t\.test\s*\(\s*([^)]*)\)", "t-test", 1),
    ("chisq", r"chisq\.test\s*\(\s*([^)]*)\)", "chi-square", 1),
    ("poisson_glm", r"glm\s*\(\s*([^)]*),\s*family\s*=\s*poisson", "poisson", 1),
    ("massed_stats", r"(mean\s*\(|median\s*\(|sd\s*\()", "counts/ct", None),
    ("matchit", r"matchit\s*\(", "psm", None),
]


def is_comment_line(line: str) -> bool:
    """Check if a line is a comment (starts with #)."""
    return line.strip().startswith("#")


def is_empty_or_comment(line: str) -> bool:
    """Check if a line is empty or a comment (not actual code)."""
    stripped = line.strip()
    return stripped == "" or stripped.startswith("#")


def extract_code_lines(lines: List[str]) -> str:
    """Extract only non-comment, non-empty lines from a list of lines."""
    return "\n".join(line for line in lines if not is_empty_or_comment(line))


def has_actual_code(lines: List[str]) -> bool:
    """Check if snippet contains at least one line of actual code (non-comment, non-empty)."""
    return any(not is_empty_or_comment(line) for line in lines)


def extract_code_tokens(snippet: str) -> List[str]:
    """Extract variable tokens from code only (excluding comments)."""
    lines = snippet.split("\n")
    code_only = extract_code_lines(lines)
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", code_only)
    return tokens


def extract_comment_tokens(snippet: str) -> List[str]:
    """Extract variable tokens from comments only."""
    lines = snippet.split("\n")
    comment_lines = [line for line in lines if is_comment_line(line)]
    comment_text = "\n".join(comment_lines)
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", comment_text)
    return tokens


def mine_code_ir(r_files: Iterable[RepoFile]) -> List[CodeAnalysisIR]:
    code_irs: List[CodeAnalysisIR] = []
    cid = 1
    for file in r_files:
        path, content = file.file_path, file.content
        lines = content.splitlines()
        for pat_name, pat, fam, grp in MODEL_PATTERNS:
            for m in re.finditer(pat, content, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL):
                char_idx = m.start()
                line_guess = content[:char_idx].count("\n")
                window = 15
                s = max(0, line_guess - window)
                e = min(len(lines), line_guess + window)
                snippet_lines = take(lines, s, e)

                # Skip snippets that don't contain actual code
                if not has_actual_code(snippet_lines):
                    continue

                snippet = "\n".join(snippet_lines)
                formula_hint = None
                if grp is not None and m.lastindex and m.lastindex >= grp:
                    formula_hint = normalize_space(m.group(grp))[:500]

                # Extract tokens from code only (not comments)
                code_tokens = extract_code_tokens(snippet)
                variables_hint = sorted(set(code_tokens))[:20]

                code_irs.append(
                    CodeAnalysisIR(
                        analysis_id=f"C-{cid:03d}",
                        file_path=path,
                        line_start=s + 1,
                        line_end=e + 1,
                        snippet=snippet,
                        model_family=fam,
                        formula_hint=formula_hint,
                        variables_hint=variables_hint,
                    )
                )
                cid += 1
    return code_irs


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def family_compat(pfam: str | None, cfam: str | None) -> float:
    if not pfam or not cfam:
        return 0.2
    return 1.0 if pfam == cfam else 0.0


def match_paper_to_code(papers: List[PaperAnalysisIR], codes: List[CodeAnalysisIR], top_k: int = 3) -> List[MatchEdge]:
    edges: List[MatchEdge] = []
    for pa in papers:
        p_tokens = set([t.lower() for t in pa.variables_hint])
        cand_scores = []
        for ca in codes:
            # Code tokens (from variables_hint, which now excludes comments)
            c_code_tokens = set([t.lower() for t in ca.variables_hint])
            code_var_sim = jaccard(p_tokens, c_code_tokens)

            # Comment tokens for secondary signal
            c_comment_tokens = set([t.lower() for t in extract_comment_tokens(ca.snippet)])
            comment_var_sim = jaccard(p_tokens, c_comment_tokens)

            # Model family compatibility
            fam_sim = family_compat(pa.model_family_hint, ca.model_family)

            # Formula hint bonus
            formula_hit = 0.0
            if ca.formula_hint and any(v.lower() in ca.formula_hint.lower() for v in pa.variables_hint[:5]):
                formula_hit = 0.2

            # Weighted scoring: prioritize code tokens and model family
            # 60% code tokens, 30% model family, 10% comment tokens, + formula bonus
            score = 0.6 * code_var_sim + 0.3 * fam_sim + 0.1 * comment_var_sim + formula_hit

            reasons = {
                "code_var_sim": code_var_sim,
                "comment_var_sim": comment_var_sim,
                "fam_sim": fam_sim,
                "formula_hit": formula_hit,
            }
            cand_scores.append((ca, score, reasons))

        cand_scores.sort(key=lambda x: x[1], reverse=True)
        for ca, sc, reasons in cand_scores[:top_k]:
            edges.append(MatchEdge(paper_id=pa.analysis_id, code_id=ca.analysis_id, score=sc, reasons=reasons))
    return edges


def greedy_unique_bipartite(edges: List[MatchEdge], min_score: float = 0.35) -> List[MatchEdge]:
    by_score = sorted(edges, key=lambda e: e.score, reverse=True)
    seen_p, seen_c, chosen = set(), set(), []
    for e in by_score:
        if e.score < min_score:
            continue
        if e.paper_id in seen_p or e.code_id in seen_c:
            continue
        chosen.append(e)
        seen_p.add(e.paper_id)
        seen_c.add(e.code_id)
    return chosen

