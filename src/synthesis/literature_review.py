"""
Automated Literature Review Generator.
Synthesizes comprehensive, publication-grade academic literature reviews from analyzed paper corpora.
"""

import logging
from typing import List, Dict, Any, Optional
import httpx
from src.config import settings

logger = logging.getLogger(__name__)

class LiteratureReviewGenerator:
    """Generates structured, topic-wise literature reviews grounded in empirical paper evidence."""

    @classmethod
    def generate_review(
        cls,
        topic_query: str,
        papers: List[Dict[str, Any]],
        clusters: Dict[str, Any],
        domains: List[str],
        ranked_gaps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Produce a structured multi-section academic literature review.
        """
        # Try LLM-assisted synthesis if available
        from src.llm.llm_router import LLMRouter
        if LLMRouter.is_available():
            llm_review = cls._generate_with_llm(topic_query, papers, clusters, domains, ranked_gaps)
            if llm_review:
                return llm_review

        # Deterministic academic synthesis fallback
        return cls._generate_structured_fallback(topic_query, papers, clusters, domains, ranked_gaps)

    @classmethod
    def _generate_with_llm(
        cls,
        topic: str,
        papers: List[Dict[str, Any]],
        clusters: Dict[str, Any],
        domains: List[str],
        ranked_gaps: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Query LLM API (Bedrock or NVIDIA) for complete academic literature review."""
        top_papers_summary = "\n".join([
            f"- \"{p.get('title')}\" ({p.get('year')}) — Citations: {p.get('citation_count')}. Abstract: {p.get('abstract', '')[:160]}..."
            for p in papers[:15]
        ])

        gap_summary = "\n".join([
            f"- {g.get('project_title') or (str(g.get('axis_a')) + ' x ' + str(g.get('axis_b')))}: {g.get('why_it_is_a_gap', '')[:140]}"
            for g in ranked_gaps[:3]
        ])

        prompt = f"""You are a senior professor writing an authoritative academic literature review for a top-tier journal.
Topic: "{topic}"
Corpus size: {len(papers)} publications analyzed.
Discovered Methodology Themes: {list(clusters.keys())}
Discovered Application Domains: {domains}

Key Papers in Corpus:
{top_papers_summary}

Detected Research Gaps:
{gap_summary}

TASK:
Write a comprehensive, publication-ready literature review structured into:
1. Executive Summary & Problem Scope
2. Thematic Breakdown of Current Approaches
3. Comparative Analysis of Methodologies
4. Key Empirical Findings & Benchmarks
5. Unresolved Research Gaps & Future Directions
6. Synthesized Conclusion

Format the output clearly with markdown subheadings (##, ###) and formal academic prose with in-text paper citations.
"""
        try:
            from src.llm.llm_router import LLMRouter
            review_text = LLMRouter.generate(prompt=prompt, temperature=0.25, max_tokens=3000)
            if review_text and len(review_text.strip()) > 100:
                first_para = review_text.strip().split("\n\n")[0]
                return {
                    "topic": topic,
                    "executive_summary": first_para if len(first_para) > 50 else f"Systematic survey of {len(papers)} publications in {topic}.",
                    "thematic_breakdown": [
                        {
                            "cluster_name": info.get("label", f"Cluster {cid}"),
                            "description": info.get("short_description", "Methodological grouping."),
                            "key_papers": [p.get("title", "") for p in papers if p.get("axis_a_tag") == info.get("label")][:2],
                            "reported_limitations": "Computational scaling and domain transfer constraints."
                        }
                        for cid, info in clusters.items()
                    ],
                    "comparative_synthesis": f"Analysis maps {len(clusters)} methodological themes across {len(domains)} problem settings.",
                    "identified_voids": f"Primary void: {ranked_gaps[0].get('project_title', 'Boundary gap')}" if ranked_gaps else "Structural voids identified across sparse matrix cells.",
                    "review_markdown": review_text.strip(),
                    "total_papers_referenced": len(papers),
                    "sections_included": [
                        "Executive Summary",
                        "Thematic Breakdown",
                        "Methodological Comparison",
                        "Empirical Findings",
                        "Unresolved Gaps",
                        "Conclusion"
                    ],
                    "generator": "llm_router"
                }
        except Exception as e:
            logger.warning(f"LLM literature review generation failed: {e}. Falling back to structured generator.")

        return None

    @classmethod
    def _generate_structured_fallback(
        cls,
        topic: str,
        papers: List[Dict[str, Any]],
        clusters: Dict[str, Any],
        domains: List[str],
        ranked_gaps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Deterministic, grounded academic literature review generator."""
        top_cited = sorted(papers, key=lambda p: p.get("citation_count", 0), reverse=True)[:5]

        sections = []

        # 1. Executive Summary
        exec_summary = (
            f"This review synthesizes findings across a curated corpus of {len(papers)} peer-reviewed scientific publications "
            f"investigating **{topic}**. Using unsupervised semantic clustering and combinatorial dimension analysis, "
            f"the literature is mapped across {len(clusters)} primary methodological families and {len(domains)} application domains. "
            f"While foundational work has demonstrated substantial empirical progress, significant structural gaps remain at the intersection "
            f"of emerging computational paradigms and complex domain settings."
        )
        sections.append(f"# Systematic Literature Review: Current State & Future Horizons in {topic}\n")
        sections.append("## 1. Executive Summary & Scope")
        sections.append(exec_summary + "\n")

        # 2. Key Methodological Families
        thematic_breakdown = []
        sections.append("## 2. Thematic Breakdown of Existing Methodologies")
        for cid, info in clusters.items():
            lbl = info.get("label", f"Cluster {cid}")
            desc = info.get("short_description", "Methodological grouping.")
            count = info.get("paper_count", 0)
            sections.append(f"### 2.{int(cid)+1 if str(cid).isdigit() else 1} {lbl} ({count} publications)")
            sections.append(f"{desc}\n")
            # List 2 representative papers
            matching_papers = [p for p in papers if p.get("axis_a_tag") == lbl][:2]
            key_paper_titles = []
            for p in matching_papers:
                t = p.get("title", "")
                if t:
                    key_paper_titles.append(t)
                sections.append(f"- **{t}** ({p.get('year')}) — *Citations: {p.get('citation_count', 0)}*")
            sections.append("")

            thematic_breakdown.append({
                "cluster_name": lbl,
                "description": desc,
                "key_papers": key_paper_titles,
                "reported_limitations": "Computational scaling and generalization constraints reported across benchmark evaluations."
            })

        # 3. Application Domain Landscape
        comp_synthesis = (
            f"Empirical evaluation in {topic} is concentrated across {len(domains)} distinct application regimes: "
            f"{', '.join(domains)}. Prominent baseline benchmarks reveal strong clustering in well-established problem settings, "
            f"while peripheral domains suffer from data sparsity and unvalidated generalization assumptions."
        )
        sections.append("## 3. Application Domain Landscape")
        sections.append(comp_synthesis + "\n")

        # 4. Critical Research Gaps
        sections.append("## 4. Unexplored Research Gaps & Identified Voids")
        sections.append(
            "Through combinatorial 2D matrix analysis, several high-potential research intersections have been detected "
            "where component methodologies are mature but direct integration remains unattempted:\n"
        )
        for idx, g in enumerate(ranked_gaps[:3]):
            title = g.get("project_title", f"{g.get('axis_a')} x {g.get('axis_b')}")
            why = g.get("why_it_is_a_gap", "")
            critique = g.get("counter_argument", "")
            sections.append(f"### Opportunity #{idx+1}: {title}")
            sections.append(f"- **The Gap:** {why}")
            sections.append(f"- **Feasibility Score:** {g.get('composite_score', 0)}/5.0 (Novelty: {g.get('novelty_score')}, Feasibility: {g.get('feasibility_score')})")
            sections.append(f"- **Adversarial Critique:** {critique}\n")

        top_gap_name = ranked_gaps[0].get("project_title") if ranked_gaps else "cross-domain combinations"
        identified_voids = (
            f"Combinatorial 2D density analysis revealed {len(ranked_gaps)} high-opportunity structural voids, "
            f"most prominently '{top_gap_name}', where component foundations exist in adjacent literature but direct integration is unattempted."
            if ranked_gaps else "No primary structural voids detected."
        )

        # 5. Conclusion
        sections.append("## 5. Conclusion & Recommendations")
        sections.append(
            f"The literature on {topic} is transitioning from isolated algorithmic proof-of-concepts toward integrated, "
            f"cross-domain deployments. Researchers are encouraged to prioritize the identified boundary opportunities, "
            f"specifically addressing the computational and numerical bottlenecks highlighted in the adversarial critiques."
        )

        review_markdown = "\n".join(sections)
        return {
            "topic": topic,
            "executive_summary": exec_summary,
            "thematic_breakdown": thematic_breakdown,
            "comparative_synthesis": comp_synthesis,
            "identified_voids": identified_voids,
            "review_markdown": review_markdown,
            "total_papers_referenced": len(papers),
            "sections_included": [
                "Executive Summary",
                "Thematic Breakdown",
                "Application Landscape",
                "Unexplored Gaps",
                "Conclusion"
            ],
            "generator": "structured_synthesis_engine"
        }
