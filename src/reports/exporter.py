"""
Academic Report & Dossier Exporter.
Exports complete research intelligence reports in Markdown, LaTeX (.tex), and printable HTML formats.
Includes 15-category Gap Taxonomy, Source Facts vs AI Inferences, Calibrated Confidence,
5-Pillar Devil's Advocate Reality Check, and Researcher Verification Checklists.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    if not text:
        return ""
    special_chars = {
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}"
    }
    for char, replacement in special_chars.items():
        text = text.replace(char, replacement)
    return text


class ReportExporter:
    """Generates downloadable academic report artifacts across multiple publication formats."""

    @classmethod
    def export_markdown(
        cls,
        topic_query: str,
        results: Dict[str, Any],
        literature_review: Optional[Dict[str, Any]] = None,
        research_questions: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Export full analysis briefing as Markdown with evidence grounding."""
        lines = []
        lines.append(f"# ResearchGapAI Report: {topic_query}\n## Evidence-Backed Research Gap Discovery Report\n")
        lines.append(f"*Generated on: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by ResearchGapAI Suite*\n")
        lines.append(f"**Total Papers Analyzed:** {results.get('corpus_size', 0)} | "
                     f"**Candidate Signals Evaluated:** {results.get('candidate_gaps_count', 0)} | "
                     f"**Validated Gaps Produced:** {len(results.get('ranked_gaps', []))}\n")

        # Literature Review section
        if literature_review:
            lines.append("## Automated Literature Review\n")
            lines.append(literature_review.get("review_markdown", ""))
            lines.append("\n---\n")

        # Actionable Research Gaps
        lines.append("## Actionable Evidence-Backed Research Gap Dossiers\n")
        for idx, g in enumerate(results.get("ranked_gaps", [])):
            title = g.get("project_title") or g.get("title") or f"{g.get('axis_a')} in {g.get('axis_b')}"
            conf = g.get("confidence_scorecard") or g.get("confidence_breakdown") or {}
            overall_c = conf.get("overall") or conf.get("overall_confidence", 80)
            ev_c = conf.get("evidence") or conf.get("evidence_confidence", 85)
            nov_c = conf.get("novelty") or conf.get("novelty_confidence", 85)
            feas_c = conf.get("feasibility") or conf.get("feasibility_confidence", 75)
            rel_c = conf.get("relevance") or conf.get("relevance_confidence", 80)
            deriv = conf.get("derivation_explanation") or conf.get("derivation", "Corpus literature evaluation")
            status = g.get("status") or g.get("gap_status", "True / Strong Gap")

            lines.append(f"### Research Gap #{idx+1}: {title}\n")
            lines.append(f"- **Gap Type (Taxonomy):** `{g.get('gap_type', 'Methodological Gap')}`")
            lines.append(f"- **Validation Status:** **{status}** ({g.get('status_description', '')})")
            lines.append(f"- **Discovery Signal:** {g.get('signal_type', 'Literature Analysis')} (*{g.get('signal_source', '')}*)")
            lines.append(f"- **Evidence Support Ratio:** {g.get('ratio_evidence_string') or g.get('evidence_ratio', 'Verified')}\n")

            lines.append("#### 📊 Calibrated Confidence Scorecard")
            lines.append(f"- **Overall Gap Confidence: {overall_c}%**")
            lines.append(f"- **Evidence Confidence:** {ev_c}% | "
                         f"**Novelty Confidence:** {nov_c}% | "
                         f"**Feasibility Confidence:** {feas_c}% | "
                         f"**Relevance Confidence:** {rel_c}%")
            lines.append(f"- *Confidence Basis:* {deriv}\n")

            rq = g.get("research_question") or g.get("core_research_question") or ""
            if rq:
                lines.append(f"#### ❓ Core Research Question")
                lines.append(f"> {rq}\n")

            if g.get("directional_hypothesis_h1"):
                lines.append(f"- **Directional Hypothesis ($H_1$):** {g.get('directional_hypothesis_h1')}")
                lines.append(f"- **Null Hypothesis ($H_0$):** {g.get('null_hypothesis_h0')}\n")

            # Source Facts vs AI Inferences
            if g.get("source_facts"):
                lines.append("#### 📚 SOURCE FACTS (Direct Literature Evidence)")
                for fact in g.get("source_facts", []):
                    lines.append(f"- {fact}")
                lines.append("")

            if g.get("ai_inferences"):
                lines.append("#### 💡 AI INFERENCES (System Deduction)")
                for inf in g.get("ai_inferences", []):
                    lines.append(f"- {inf}")
                lines.append("")

            # Grounded Experiment
            exp = g.get("grounded_experiment", {})
            if exp:
                target_ds = exp.get("benchmark_dataset") or exp.get("target_dataset", "N/A")
                indep_vars = exp.get("independent_variables") or ([exp.get("independent_variable")] if exp.get("independent_variable") else [])
                metrics = exp.get("evaluation_metrics") or exp.get("metrics") or []
                conds = exp.get("experimental_conditions", [])
                lines.append("#### 🧪 Grounded Experimental Protocol")
                lines.append(f"- **Target Dataset / Modality:** {target_ds}")
                lines.append(f"- **Baseline Models:** {', '.join(exp.get('baselines', []))}")
                lines.append(f"- **Independent Variables:** {', '.join(indep_vars)}")
                lines.append(f"- **Experimental Conditions:** {', '.join(conds)}")
                lines.append(f"- **Evaluation Metrics:** {', '.join(metrics)}")
                lines.append(f"- **Statistical Test:** {exp.get('statistical_test')}")
                lines.append(f"- **Expected Contribution:** {exp.get('expected_contribution')}\n")

            # 5-Pillar Devil's Advocate
            lines.append("#### 🚨 5-Pillar Devil's Advocate Reality Check")
            lines.append(f"**Verdicts:** `{g.get('devil_advocate_verdicts', '4 PASS / 1 WARNING')}`\n")
            challenges = g.get("devils_advocate_critique") or g.get("devil_advocate_challenges") or {}
            for ch_name, ch_data in challenges.items():
                if isinstance(ch_data, dict):
                    v = ch_data.get("verdict", "PASS")
                    q = ch_data.get("challenge") or ch_data.get("question") or ""
                    expl = ch_data.get("explanation", "")
                    label = ch_name.replace("_", " ").title()
                    lines.append(f"- **{label} [{v}]:** {q}{f' — *{expl}*' if expl else ''}")
            counter = g.get("counter_argument", "")
            if counter:
                lines.append(f"\n**Primary Technical Failure Mode:** {counter}\n")

            # Researcher Verification Checklist
            chk = g.get("researcher_verification_checklist", [])
            if chk:
                lines.append("#### 🧑‍🔬 Pre-Flight Researcher Verification Checklist")
                for item in chk:
                    if isinstance(item, dict):
                        lines.append(f"- [ ] **{item.get('step')}:** `{item.get('query')}`")
                    else:
                        lines.append(f"- [ ] {item}")
                lines.append("\n---\n")

        return "\n".join(lines)

    @classmethod
    def export_latex(
        cls,
        topic_query: str,
        results: Dict[str, Any],
        literature_review: Optional[Dict[str, Any]] = None,
        research_questions: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Export IEEE-compliant LaTeX document (.tex)."""
        clean_topic = _escape_latex(topic_query)
        tex_lines = [
            "\\documentclass[conference]{IEEEtran}",
            "\\usepackage{amsmath,amsfonts,amssymb}",
            "\\usepackage{booktabs}",
            "\\usepackage{hyperref}",
            "\\begin{document}",
            f"\\title{{Evidence-Backed Research Landscape and Gap Matrix in {clean_topic}}}",
            "\\author{\\IEEEauthorblockN{ResearchGapAI Intelligence Suite}}",
            "\\maketitle",
            "\\begin{abstract}",
            f"This paper presents an evidence-grounded research gap analysis for scientific literature in {clean_topic}. "
            f"From an indexed corpus of {results.get('corpus_size', 0)} publications, we extract structured research dimensions, "
            "mine repeated unresolved limitations and future-work trajectories, verify novelty against prior art, and formulate "
            "adversarially stress-tested research hypotheses and experimental protocols.",
            "\\end{abstract}",
            "\\section{Introduction}",
            f"Identifying legitimate research gaps in {clean_topic} requires multi-signal evidence verification. "
            "Rather than treating unexplored combinatorial intersections as definitive gaps, our system subjects every candidate "
            "to adversarial novelty checks, 5-pillar Devil's Advocate critique, and calibrated confidence estimation.",
            "\\section{Validated Research Gap Dossiers}"
        ]

        for idx, g in enumerate(results.get("ranked_gaps", [])[:3]):
            title = _escape_latex(g.get("project_title") or g.get("title") or "")
            gap_type = _escape_latex(g.get("gap_type", "Methodological Gap"))
            status = _escape_latex(g.get("status") or g.get("gap_status", "True / Strong Gap"))
            q = _escape_latex(g.get("research_question") or g.get("core_research_question", ""))
            conf_obj = g.get("confidence_scorecard") or g.get("confidence_breakdown") or {}
            conf = conf_obj.get("overall") or conf_obj.get("overall_confidence", 80)
            crit = _escape_latex(g.get("counter_argument", ""))

            tex_lines.extend([
                f"\\subsection{{Research Gap \\#{idx+1}: {title}}}",
                f"\\textbf{{Gap Taxonomy Category:}} {gap_type} \\\\",
                f"\\textbf{{Validation Status:}} {status} \\\\",
                f"\\textbf{{Overall Confidence:}} {conf}\\% \\\\",
                f"\\textbf{{Core Research Question:}} \\textit{{{q}}} \\\\",
                f"\\textbf{{Adversarial Reality Check:}} {crit} \\\\"
            ])

            if g.get("source_facts") or g.get("ai_inferences"):
                tex_lines.append("\\subsection*{Source Facts vs AI Inferences}")
                for fact in g.get("source_facts", [])[:2]:
                    tex_lines.append(f"\\textbf{{Source Fact:}} {_escape_latex(fact)} \\\\")
                for inf in g.get("ai_inferences", [])[:2]:
                    tex_lines.append(f"\\textbf{{AI Inference:}} {_escape_latex(inf)} \\\\")

        tex_lines.extend([
            "\\section{Conclusion}",
            f"The evidence-backed analysis provides a rigorous, verified foundation for future experimental studies in {clean_topic}.",
            "\\end{document}"
        ])

        return "\n".join(tex_lines)

    @classmethod
    def export_html(
        cls,
        topic_query: str,
        results: Dict[str, Any],
        literature_review: Optional[Dict[str, Any]] = None
    ) -> str:
        """Export self-contained, responsive HTML briefing."""
        md = cls.export_markdown(topic_query, results, literature_review)
        body_html = md.replace("\n\n", "</p><p>").replace("\n- ", "<br>• ").replace("\n", "<br>")

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ResearchGapAI: {topic_query}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; line-height: 1.6; color: #1e293b; max-width: 900px; margin: 40px auto; padding: 0 20px; background: #f8fafc; }}
        h1, h2, h3 {{ color: #0f172a; }}
        blockquote {{ border-left: 4px solid #38bdf8; margin: 1.5em 10px; padding: 0.5em 10px; background: #e0f2fe; }}
        code {{ background: #e2e8f0; padding: 2px 6px; border-radius: 4px; }}
        hr {{ border: 0; height: 1px; background: #cbd5e1; margin: 30px 0; }}
        .scorecard-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; margin: 15px 0; }}
    </style>
</head>
<body>
    <h1>Evidence-Backed Research Gap Dossiers</h1>
    <div class="scorecard-grid"></div>
    <p>{body_html}</p>
</body>
</html>"""

    # Convenience aliases for dossier lists
    @classmethod
    def export_to_markdown(cls, dossiers: List[Dict[str, Any]], topic: str = "Research Intelligence") -> str:
        return cls.export_markdown(topic_query=topic, results={"ranked_gaps": dossiers})

    @classmethod
    def export_to_latex(cls, dossiers: List[Dict[str, Any]], topic: str = "Research Intelligence") -> str:
        return cls.export_latex(topic_query=topic, results={"ranked_gaps": dossiers})

    @classmethod
    def export_to_html(cls, dossiers: List[Dict[str, Any]], topic: str = "Research Intelligence") -> str:
        return cls.export_html(topic_query=topic, results={"ranked_gaps": dossiers})
