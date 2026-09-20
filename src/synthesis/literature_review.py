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
        if settings.DYNAMIC_LLM_ENABLED and settings.NVIDIA_API_KEY:
            llm_review = cls._generate_with_llm(topic_query, papers, clusters, domains, ranked_gaps)
            if llm_review:
                return llm_review
        elif settings.LLM_REQUIRED:
            raise RuntimeError("NVIDIA_API_KEY is required because LLM_REQUIRED=true.")

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
        """Query NVIDIA AI API for complete academic literature review."""
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
            from src.llm.nvidia_client import NvidiaClient
            review_text = NvidiaClient.generate(prompt=prompt, temperature=settings.LLM_STRUCTURED_TEMPERATURE, max_tokens=3000)
            if review_text and len(review_text.strip()) > 100:
                return {
                    "topic": topic,
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
                    "generator": f"nvidia/{settings.NVIDIA_MODEL}"
                }
        except Exception as e:
            logger.warning(f"NVIDIA LLM literature review generation failed: {e}. Falling back to structured generator.")

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
        sections.append(f"# Systematic Literature Review: Current State & Future Horizons in {topic}\n")
        sections.append("## 1. Executive Summary & Scope")
        sections.append(
            f"This review synthesizes findings across a curated corpus of {len(papers)} peer-reviewed scientific publications "
            f"investigating **{topic}**. Using unsupervised semantic clustering and combinatorial dimension analysis, "
            f"the literature is mapped across {len(clusters)} primary methodological families and {len(domains)} application domains. "
            f"While foundational work has demonstrated substantial empirical progress, significant structural gaps remain at the intersection "
            f"of emerging computational paradigms and complex domain settings.\n"
        )

        # 2. Key Methodological Families
        sections.append("## 2. Thematic Breakdown of Existing Methodologies")
        for cid, info in clusters.items():
            lbl = info.get("label", f"Cluster {cid}")
            desc = info.get("short_description", "Methodological grouping.")
            count = info.get("paper_count", 0)
            sections.append(f"### 2.{int(cid)+1 if str(cid).isdigit() else 1} {lbl} ({count} publications)")
            sections.append(f"{desc}\n")
            # List 2 representative papers
            matching_papers = [p for p in papers if p.get("axis_a_tag") == lbl][:2]
            for p in matching_papers:
                sections.append(f"- **{p.get('title')}** ({p.get('year')}) — *Citations: {p.get('citation_count', 0)}*")
            sections.append("")

        # 3. Application Domain Landscape
        sections.append("## 3. Application Domain Landscape")
        sections.append(
            f"Empirical evaluation in {topic} is concentrated across {len(domains)} distinct application regimes: "
            f"{', '.join(domains)}. Prominent baseline benchmarks reveal strong clustering in well-established problem settings, "
            f"while peripheral domains suffer from data sparsity and unvalidated generalization assumptions.\n"
        )

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
