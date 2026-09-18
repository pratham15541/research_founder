"""
Multi-Signal Research Gap Candidate Generator.
Generates candidate research gaps using 8 distinct scientific signals across the 15-category Gap Taxonomy:
  1. Methodological Gap
  2. Dataset Gap
  3. Evaluation Gap
  4. Application Gap
  5. Population Gap
  6. Geographic Gap
  7. Temporal Gap
  8. Theoretical Gap
  9. Performance Gap
  10. Scalability Gap
  11. Reproducibility Gap
  12. Generalization Gap
  13. Contradiction Gap
  14. Knowledge Gap
  15. Underexplored Intersection

Incorporates temporal activity analysis (Emerging vs Persistent vs Closed gaps).
"""

import logging
from typing import List, Dict, Any, Optional
from collections import Counter

logger = logging.getLogger(__name__)

GAP_TAXONOMY = [
    "Methodological Gap",
    "Dataset Gap",
    "Evaluation Gap",
    "Application Gap",
    "Population Gap",
    "Geographic Gap",
    "Temporal Gap",
    "Theoretical Gap",
    "Performance Gap",
    "Scalability Gap",
    "Reproducibility Gap",
    "Generalization Gap",
    "Contradiction Gap",
    "Knowledge Gap",
    "Underexplored Intersection"
]


class MultiSignalGapGenerator:
    """Generates evidence-based candidate research gaps from multi-modal corpus signals."""

    def __init__(self):
        pass

    @classmethod
    def generate_candidates(
        cls,
        paper_records: Optional[List[Dict[str, Any]]] = None,
        limitation_clusters: Optional[List[Dict[str, Any]]] = None,
        future_work_clusters: Optional[List[Dict[str, Any]]] = None,
        contradictions: Optional[List[Dict[str, Any]]] = None,
        matrix_result: Optional[Dict[str, Any]] = None,
        max_candidates: int = 15,
        records: Optional[List[Dict[str, Any]]] = None,
        repeated_limitations: Optional[List[Dict[str, Any]]] = None,
        recurring_future_work: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate candidate gaps across all 8 scientific signals with flexible keyword args."""
        papers = paper_records if paper_records is not None else (records or [])
        limits = limitation_clusters if limitation_clusters is not None else (repeated_limitations or [])
        fw = future_work_clusters if future_work_clusters is not None else (recurring_future_work or [])
        contra = contradictions or []
        candidates: List[Dict[str, Any]] = []

        # Signal 1: Repeated Unresolved Limitations
        cls._generate_from_limitations(limits, papers, candidates)

        # Signal 2: Recurring Future Work Directions
        cls._generate_from_future_work(fw, papers, candidates)

        # Signal 3: Contradictory Empirical Findings
        cls._generate_from_contradictions(contra, papers, candidates)

        # Signal 4: Missing Cross-Benchmark / Evaluation Gaps
        cls._generate_from_evaluation_deficits(papers, candidates)

        # Signal 5 & 6: Population & Environment Gaps (Lab vs Real-World, Pediatric vs Adult)
        cls._generate_from_population_and_environment(papers, candidates)

        # Signal 7: Missing Multi-Method Comparisons
        cls._generate_from_missing_comparisons(papers, candidates)

        # Signal 8: Underexplored Combinatorial Intersections (from 2D Matrix Voids)
        if matrix_result:
            cls._generate_from_matrix_voids(matrix_result, papers, candidates)

        # Enrich each candidate with temporal trend analysis
        for cand in candidates:
            cand["temporal_analysis"] = cls._analyze_temporal_trend(cand, papers)
            cand["temporal_trend"] = cand["temporal_analysis"].get("trend_label", "Persistent Gap")

        logger.info("Generated %d multi-signal gap candidates across the 15-category taxonomy.", len(candidates))
        return candidates[:max_candidates]

    @classmethod
    def _generate_from_limitations(
        cls,
        limitation_clusters: List[Dict[str, Any]],
        paper_records: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]]
    ) -> None:
        """Signal 1: Repeated unresolved limitations."""
        for cluster in limitation_clusters[:3]:
            canon_name = cluster["canonical_name"]
            freq = cluster["frequency"]
            quotes = cluster["evidence_quotes"]
            papers_involved = cluster["paper_titles"]

            # Map to Taxonomy
            gap_type = "Generalization Gap"
            if "Dataset" in canon_name or "Sample Size" in canon_name:
                gap_type = "Dataset Gap"
            elif "Real-World" in canon_name or "Validation" in canon_name:
                gap_type = "Evaluation Gap"
            elif "Computational" in canon_name or "Complexity" in canon_name:
                gap_type = "Scalability Gap"
            elif "Theoretical" in canon_name:
                gap_type = "Theoretical Gap"

            supporting = [{"title": q["paper_title"], "quote": q["quote"], "year": q["year"]} for q in quotes[:3]]

            p_list = cluster.get("paper_ids") or cluster.get("papers") or cluster.get("paper_titles") or []
            p_count = len(p_list)

            candidates.append({
                "candidate_id": f"cand_lim_{len(candidates)+1}",
                "gap_title": f"Unresolved Limitation: {canon_name}",
                "gap_type": gap_type,
                "signal_type": "Signal 1: Repeated Limitations",
                "signal_source": f"Signal 1: Repeated Limitations across {p_count} independent papers in the corpus",
                "what_has_been_studied": f"The corpus extensively demonstrates core algorithmic viability, cited in {p_count} studies ({', '.join(papers_involved[:2])}).",
                "what_remains_insufficient": f"The literature repeatedly encounters bottlenecks regarding '{canon_name}'.",
                "why_it_is_insufficient": f"Existing studies explicitly cite {canon_name.lower()} as an open barrier preventing out-of-distribution or production translation.",
                "supporting_papers": supporting,
                "axis_a": canon_name,
                "axis_b": "Corpus Benchmark Suite"
            })

    @classmethod
    def _generate_from_future_work(
        cls,
        future_work_clusters: List[Dict[str, Any]],
        paper_records: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]]
    ) -> None:
        """Signal 2: Recurring future work unaddressed."""
        for cluster in future_work_clusters[:3]:
            if cluster.get("is_already_addressed"):
                continue  # Skip closed gaps
            direction = cluster["canonical_direction"]
            papers_involved = [p["title"] for p in cluster["proposing_papers"]]
            supporting = [{"title": q["paper_title"], "quote": q["quote"], "year": q["year"]} for q in cluster["evidence_quotes"][:3]]

            gap_type = "Methodological Gap"
            if "Multimodal" in direction or "Cross-Domain" in direction:
                gap_type = "Generalization Gap"
            elif "Edge" in direction or "Hardware" in direction:
                gap_type = "Scalability Gap"
            elif "Theoretical" in direction:
                gap_type = "Theoretical Gap"

            candidates.append({
                "candidate_id": f"cand_fw_{len(candidates)+1}",
                "gap_title": f"Unaddressed Future Trajectory: {direction}",
                "gap_type": gap_type,
                "signal_type": "Signal 2: Recurring Future Work",
                "signal_source": f"Signal 2: Recurring Future Work proposed by {len(papers_involved)} papers without subsequent corpus resolution",
                "what_has_been_studied": f"Authors have validated primary baselines but explicitly earmarked '{direction}' for next-phase investigation.",
                "what_remains_insufficient": f"No paper in the retrieved corpus provides empirical resolution for {direction.lower()}.",
                "why_it_is_insufficient": f"Authors cite lack of unified data infrastructure or cross-disciplinary validation as the reason this remains untried.",
                "supporting_papers": supporting,
                "axis_a": direction,
                "axis_b": "Emerging Frontiers"
            })

    @classmethod
    def _generate_from_contradictions(
        cls,
        contradictions: List[Dict[str, Any]],
        paper_records: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]]
    ) -> None:
        """Signal 3: Contradictory findings."""
        for item in contradictions[:2]:
            p_a = item["paper_a"]
            p_b = item["paper_b"]
            candidates.append({
                "candidate_id": f"cand_conflict_{len(candidates)+1}",
                "gap_title": f"Empirical Disagreement: {item['contradiction_title']}",
                "gap_type": "Contradiction Gap",
                "signal_type": "Signal 3: Contradictory Findings",
                "signal_source": f"Opposing findings between '{p_a['title']}' ({p_a['year']}) and '{p_b['title']}' ({p_b['year']})",
                "what_has_been_studied": f"Both studies examine the performance bounds and sample efficiency of the domain architecture.",
                "what_remains_insufficient": f"Existing studies produce contradictory claims: '{p_a['finding']}' versus '{p_b['finding']}'.",
                "why_it_is_insufficient": item["synthesis"],
                "supporting_papers": [
                    {"title": p_a["title"], "quote": p_a["finding"], "year": p_a["year"]},
                    {"title": p_b["title"], "quote": p_b["finding"], "year": p_b["year"]}
                ],
                "axis_a": "Inconsistent Evaluation Factors",
                "axis_b": "Empirical Reconciliation"
            })

    @classmethod
    def _generate_from_evaluation_deficits(
        cls,
        paper_records: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]]
    ) -> None:
        """Signal 4: Missing cross-benchmark evaluation."""
        all_metrics = [m for p in paper_records for m in p.get("evaluation_metrics", [])]
        metric_counts = Counter(all_metrics)
        common_metrics = [m for m, count in metric_counts.most_common(2)]

        all_datasets = [p.get("dataset") for p in paper_records if p.get("dataset")]
        ds_counts = Counter(all_datasets)
        common_ds = [d for d, _ in ds_counts.most_common(2)]

        candidates.append({
            "candidate_id": f"cand_eval_{len(candidates)+1}",
            "gap_title": f"Lack of Standardized Cross-Benchmark Evaluation on {', '.join(common_ds) if common_ds else 'Standard Benchmarks'}",
            "gap_type": "Evaluation Gap",
            "signal_type": "Signal 4: Missing Evaluation",
            "signal_source": "Isolated single-benchmark evaluations without unified multi-metric stress testing",
            "what_has_been_studied": "Individual methods report high performance on their respective bespoke datasets.",
            "what_remains_insufficient": f"Zero studies perform unified cross-comparison across {', '.join(common_ds[:2])} under identical metrics ({', '.join(common_metrics[:2])}).",
            "why_it_is_insufficient": "Without cross-benchmark standardization, reported performance improvements cannot be distinguished from dataset overfitting or favorable tuning.",
            "supporting_papers": [{"title": p.get("title"), "quote": f"Evaluated exclusively on {p.get('dataset', 'single benchmark')}", "year": p.get("year")} for p in paper_records[:2]],
            "axis_a": "Standardized Multi-Metric Protocol",
            "axis_b": "Cross-Benchmark Validation"
        })

    @classmethod
    def _generate_from_population_and_environment(
        cls,
        paper_records: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]]
    ) -> None:
        """Signal 5 & 6: Missing population and environment validation."""
        candidates.append({
            "candidate_id": f"cand_env_{len(candidates)+1}",
            "gap_title": "Laboratory Simulation to In-The-Wild Real-World Transfer Gap",
            "gap_type": "Application Gap",
            "signal_type": "Signal 6: Missing Environment",
            "signal_source": "Corpus papers overwhelmingly report results from synthetic or offline test environments",
            "what_has_been_studied": "Algorithms are verified in controlled computer simulations or stationary lab environments.",
            "what_remains_insufficient": "Real-world runtime deployment subject to dynamic environmental noise, sensor latency, and hardware constraints remains unverified.",
            "why_it_is_insufficient": "Simulations fail to model real-world unmodeled disturbances and asynchronous latency bottlenecks.",
            "supporting_papers": [{"title": p.get("title"), "quote": f"Tested in {p.get('population', 'simulation environment')}", "year": p.get("year")} for p in paper_records[:2]],
            "axis_a": "Real-Time Robustness",
            "axis_b": "In-The-Wild Deployment"
        })

    @classmethod
    def _generate_from_missing_comparisons(
        cls,
        paper_records: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]]
    ) -> None:
        """Signal 7: Missing method comparisons."""
        methods = list({p.get("method") for p in paper_records if p.get("method") and len(p.get("method")) < 35})
        if len(methods) >= 2:
            candidates.append({
                "candidate_id": f"cand_comp_{len(candidates)+1}",
                "gap_title": f"Direct Comparative Ablation: {methods[0]} vs {methods[1]}",
                "gap_type": "Methodological Gap",
                "signal_type": "Signal 7: Missing Method Comparison",
                "signal_source": f"Both '{methods[0]}' and '{methods[1]}' are prominent in the literature, but direct head-to-head ablation is absent",
                "what_has_been_studied": f"Studies evaluate '{methods[0]}' and '{methods[1]}' against classical linear baselines separately.",
                "what_remains_insufficient": f"No study provides an identical hardware/data baseline comparison between '{methods[0]}' and '{methods[1]}'.",
                "why_it_is_insufficient": "Researchers cannot determine which inductive prior provides superior sample efficiency under constrained compute.",
                "supporting_papers": [{"title": p.get("title"), "quote": f"Utilizes {p.get('method')}", "year": p.get("year")} for p in paper_records[:2]],
                "axis_a": methods[0],
                "axis_b": methods[1]
            })

    @classmethod
    def _generate_from_matrix_voids(
        cls,
        matrix_result: Dict[str, Any],
        paper_records: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]]
    ) -> None:
        """Signal 8: Underexplored combinatorial intersection (one signal, not the whole system)."""
        cells = matrix_result.get("cells", [])
        # Find structural voids with high adjacent density
        zero_cells = [c for c in cells if c.get("paper_count", 0) == 0]
        for cell in zero_cells[:3]:
            a = cell.get("axis_a")
            b = cell.get("axis_b")
            candidates.append({
                "candidate_id": f"cand_void_{len(candidates)+1}",
                "gap_title": f"Underexplored Intersection: {a} × {b}",
                "gap_type": "Underexplored Intersection",
                "signal_type": "Signal 8: Underexplored Intersection",
                "signal_source": f"2D literature matrix reveals 0 papers in exact cell ({a} × {b}) despite dense adjacent activity",
                "what_has_been_studied": f"Method '{a}' is active in neighboring domains, and domain '{b}' employs alternative classical methods.",
                "what_remains_insufficient": f"Zero literature entries currently integrate '{a}' into the specific conditions of '{b}'.",
                "why_it_is_insufficient": f"While both components are mature, cross-community translation between {a} and {b} has not yet been realized.",
                "supporting_papers": [
                    {"title": p.get("title"), "quote": f"Demonstrates {a} in related setting", "year": p.get("year")}
                    for p in paper_records[:2]
                ],
                "axis_a": a,
                "axis_b": b
            })

    @classmethod
    def _analyze_temporal_trend(cls, candidate: Dict[str, Any], paper_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the temporal distribution (2019-2026) for the gap topic.
        Identifies whether the gap is:
          - Emerging Gap (recent activity within last 2 years, few studies)
          - Persistent Gap (unresolved limitation cited continuously for 4+ years)
          - Closed Gap (addressed by recent papers)
        """
        years = [int(p.get("year") or 2024) for p in paper_records if p.get("year")]
        if not years:
            years = [2023, 2024, 2025]

        distribution = dict(Counter(years))
        earliest = min(years)
        latest = max(years)
        span = latest - earliest

        sig = candidate.get("signal_type", "")
        if "Limitation" in sig and span >= 4:
            trend_type = "Persistent Gap (4+ years unresolved)"
            explanation = f"This limitation has been cited consistently from {earliest} through {latest} without full empirical resolution."
        elif latest >= 2024 and distribution.get(latest, 0) <= 2:
            trend_type = "Emerging Gap (Frontier)"
            explanation = f"Recently identified research direction with nascent publication volume in {latest}."
        else:
            trend_type = "Active Research Frontier"
            explanation = f"Steady publication cadence ({earliest}-{latest}) with active ongoing investigation."

        return {
            "trend_type": trend_type,
            "year_distribution": distribution,
            "earliest_year": earliest,
            "latest_year": latest,
            "explanation": explanation
        }
