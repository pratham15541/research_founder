"""
Academic Report & Dossier Exporter.
Exports complete research intelligence reports in Markdown, LaTeX (.tex), and printable HTML formats.
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
        """Export full analysis briefing as Markdown."""
        lines = []
        lines.append(f"# ResearchGapAI Report: {topic_query}\n")
        lines.append(f"*Generated on: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*\n")
        lines.append(f"**Total Papers Analyzed:** {results.get('corpus_size', 0)} | "
                     f"**Silhouette Cohesion:** {results.get('silhouette_score', 0.0)} | "
                     f"**Candidate Gaps:** {results.get('candidate_gaps_count', 0)}\n")

        # Literature Review section if available
        if literature_review:
            lines.append("## Automated Literature Review\n")
            lines.append(literature_review.get("review_markdown", ""))
            lines.append("\n---\n")

        # Top Ranked Gaps
        lines.append("## Actionable Research Gap Dossiers\n")
        for idx, g in enumerate(results.get("ranked_gaps", [])):
            title = g.get("project_title", f"{g.get('axis_a')} in {g.get('axis_b')}")
            lines.append(f"### Gap #{idx+1}: {title}")
            lines.append(f"- **Discovered Method (Axis A):** {g.get('axis_a')}")
            lines.append(f"- **Discovered Domain (Axis B):** {g.get('axis_b')}")
            lines.append(f"- **Composite Opportunity Score:** {g.get('composite_score')}/5.0 (Novelty: {g.get('novelty_score')}, Feasibility: {g.get('feasibility_score')})")
            lines.append(f"- **Core Research Question:** {g.get('core_research_question')}")
            lines.append(f"- **Why It's a Gap:** {g.get('why_it_is_a_gap')}")
            lines.append(f"- **Suggested First Experiment:** {g.get('suggested_first_experiment')}")
            lines.append(f"- **Devil's Advocate Reality Check:** {g.get('counter_argument')}\n")

        # Research Questions
        if research_questions:
            lines.append("## Formal Research Questions & Experimental Hypotheses\n")
            for idx, rq in enumerate(research_questions):
                lines.append(f"### Hypothesis Protocol #{idx+1}")
                lines.append(f"- **Research Question:** {rq.get('primary_research_question')}")
                lines.append(f"- **Hypothesis ($H_1$):** {rq.get('primary_hypothesis_h1')}")
                lines.append(f"- **Null Hypothesis ($H_0$):** {rq.get('null_hypothesis_h0')}")
                vars_dict = rq.get("variables", {})
                lines.append(f"- **Variables:** Independent: {vars_dict.get('independent')} | Dependent: {vars_dict.get('dependent')}")
                if isinstance(vars_dict, dict):
                    ind_v = vars_dict.get('independent', 'N/A')
                    dep_v = vars_dict.get('dependent', 'N/A')
                else:
                    ind_v, dep_v = str(vars_dict), 'N/A'
                lines.append(f"- **Variables:** Independent: {ind_v} | Dependent: {dep_v}")
                lines.append(f"- **Expected Contribution:** {rq.get('expected_contributions')}\n")

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
            f"\\title{{Research Landscape, Gap Matrix, and Exploratory Hypotheses in {clean_topic}}}",
            "\\author{\\IEEEauthorblockN{ResearchGapAI Intelligence Suite}}",
            "\\maketitle",
            "\\begin{abstract}",
            f"This paper presents an automated, unsupervised landscape mapping and combinatorial gap analysis of scientific literature in {clean_topic}. "
            f"Based on a synthesized corpus of {results.get('corpus_size', 0)} publications, we discover structural methodological and domain dimensions, "
            "identify empirical research voids, and formulate evidence-backed research questions with adversarial failure-mode critique.",
            "\\end{abstract}",
            "\\section{Introduction}",
            f"Determining underexplored intersections in {clean_topic} requires combinatorial analysis across orthogonal research dimensions. "
            "Traditional review methodologies summarize individual documents in isolation; here we present a corpus-derived 2D density formulation.",
            "\\section{Identified Research Gaps}"
        ]

        for idx, g in enumerate(results.get("ranked_gaps", [])[:3]):
            title = _escape_latex(g.get("project_title", ""))
            a = _escape_latex(g.get("axis_a", ""))
            b = _escape_latex(g.get("axis_b", ""))
            why = _escape_latex(g.get("why_it_is_a_gap", ""))
            crit = _escape_latex(g.get("counter_argument", ""))

            tex_lines.extend([
                f"\\subsection{{Opportunity \\#{idx+1}: {title}}}",
                f"\\textbf{{Method Paradigm:}} {a} \\\\",
                f"\\textbf{{Application Setting:}} {b} \\\\",
                f"\\textbf{{Composite Opportunity:}} {g.get('composite_score', 0)}/5.0 \\\\",
                f"\\textbf{{Theoretical Gap:}} {why} \\\\",
                f"\\textbf{{Adversarial Reality Check:}} {crit}"
            ])

        tex_lines.extend([
            "\\section{Conclusion}",
            f"The combinatorial analysis reveals substantial untapped potential at the boundary of emerging methods and understudied application regimes in {clean_topic}.",
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
        """Export printable standalone HTML report."""
        clean_topic = topic_query.replace("<", "&lt;").replace(">", "&gt;")
        gaps_html = ""
        for idx, g in enumerate(results.get("ranked_gaps", [])):
            gaps_html += f"""
            <div style="border-left: 4px solid #f59e0b; padding-left: 16px; margin-bottom: 24px;">
                <h3>#{idx+1}: {g.get('project_title', '')}</h3>
                <p><b>Method:</b> {g.get('axis_a')} | <b>Domain:</b> {g.get('axis_b')} | <b>Opportunity Score:</b> {g.get('composite_score')}/5.0</p>
                <p><b>Research Question:</b> <i>{g.get('core_research_question', '')}</i></p>
                <p><b>Why It's a Gap:</b> {g.get('why_it_is_a_gap', '')}</p>
                <div style="background-color: #fee2e2; border-left: 4px solid #ef4444; padding: 10px; color: #991b1b; border-radius: 4px;">
                    <b>Devil's Advocate Critique:</b> {g.get('counter_argument', '')}
                </div>
            </div>
            """

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>ResearchGapAI Briefing: {clean_topic}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #1e293b; }}
        h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; }}
        h2 {{ color: #1e293b; margin-top: 32px; border-bottom: 1px solid #cbd5e1; padding-bottom: 8px; }}
        .metrics {{ display: flex; gap: 20px; background-color: #f8fafc; padding: 16px; border-radius: 8px; margin-bottom: 24px; border: 1px solid #e2e8f0; }}
        .metric-item {{ flex: 1; }}
        .metric-val {{ font-size: 1.5em; font-weight: bold; color: #0284c7; }}
    </style>
</head>
<body>
    <h1>🔬 ResearchGapAI Executive Briefing</h1>
    <p><b>Topic:</b> {clean_topic} &nbsp;|&nbsp; <b>Date:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>
    <div class="metrics">
        <div class="metric-item"><div class="metric-val">{results.get('corpus_size', 0)}</div>Corpus Size</div>
        <div class="metric-item"><div class="metric-val">{results.get('silhouette_score', 0.0)}</div>Silhouette Cohesion</div>
        <div class="metric-item"><div class="metric-val">{len(results.get('ranked_gaps', []))}</div>Ranked Opportunities</div>
    </div>
    <h2>Top Actionable Research Gap Dossiers</h2>
    {gaps_html}
</body>
</html>"""
        return html_content
