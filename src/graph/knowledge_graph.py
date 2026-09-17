"""
Knowledge Graph generator for paper-to-paper relationships.
Builds NetworkX citation and semantic similarity graphs with Pyvis interactive HTML export.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)

class KnowledgeGraphBuilder:
    """Constructs interactive citation and semantic similarity networks for the research corpus."""

    @classmethod
    def build_graph(
        cls,
        papers: List[Dict[str, Any]],
        embeddings: Optional[np.ndarray] = None,
        similarity_threshold: float = 0.70,
        max_edges_per_node: int = 4
    ) -> nx.Graph:
        """
        Build an undirected NetworkX graph where:
        - Nodes = Papers (with cluster, year, citation metadata)
        - Edges = High semantic cosine similarity (> threshold)
        """
        G = nx.Graph()

        # Add nodes
        for idx, p in enumerate(papers):
            pid = p.get("id") or f"paper_{idx}"
            title = p.get("title", "Untitled")
            year = p.get("year", 2024)
            citations = p.get("citation_count", 0)
            cluster = p.get("axis_a_tag", "Uncategorized")
            is_uploaded = p.get("is_uploaded", False)

            # Node size proportional to citation count
            node_size = max(10, min(35, 10 + int(np.log1p(citations) * 3)))

            G.add_node(
                pid,
                title=f"{title} ({year})",
                label=title[:30] + "..." if len(title) > 30 else title,
                year=year,
                citations=citations,
                cluster=cluster,
                is_uploaded=is_uploaded,
                size=node_size,
                url=p.get("source_url", "")
            )

        # Add semantic similarity edges if embeddings provided
        if embeddings is not None and embeddings.shape[0] == len(papers):
            n = len(papers)
            # Normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            norm_embs = embeddings / norms
            sim_matrix = np.dot(norm_embs, norm_embs.T)

            for i in range(n):
                pid_i = papers[i].get("id") or f"paper_{i}"
                sims = sim_matrix[i]
                # Find top similar neighbors excluding self
                top_neighbors = np.argsort(sims)[::-1]
                added = 0
                for j in top_neighbors:
                    if i == j:
                        continue
                    if sims[j] >= similarity_threshold and added < max_edges_per_node:
                        pid_j = papers[j].get("id") or f"paper_{j}"
                        G.add_edge(pid_i, pid_j, weight=round(float(sims[j]), 2))
                        added += 1

        logger.info(f"Knowledge Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    @classmethod
    def export_pyvis_html(
        cls,
        graph: nx.Graph,
        highlight_pids: Optional[List[str]] = None,
        height: str = "500px",
        width: str = "100%"
    ) -> str:
        """Export graph to standalone interactive HTML string using Pyvis."""
        try:
            from pyvis.network import Network
            net = Network(height=height, width=width, bgcolor="#0f172a", font_color="#e2e8f0")
            net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=95)

            highlight_set = set(highlight_pids or [])

            for node_id, attrs in graph.nodes(data=True):
                is_highlighted = node_id in highlight_set
                color = "#38bdf8"  # Default light blue
                if attrs.get("is_uploaded"):
                    color = "#a855f7"  # Purple for uploaded
                if is_highlighted:
                    color = "#fbbf24"  # Gold for candidate gap papers

                border_width = 3 if is_highlighted else 1

                net.add_node(
                    node_id,
                    label=attrs.get("label", node_id),
                    title=f"<b>{attrs.get('title')}</b><br>Citations: {attrs.get('citations')}<br>Cluster: {attrs.get('cluster')}",
                    color=color,
                    size=attrs.get("size", 15) * (1.5 if is_highlighted else 1.0),
                    borderWidth=border_width
                )

            for u, v, data in graph.edges(data=True):
                weight = data.get("weight", 0.5)
                net.add_edge(u, v, value=weight, color="#475569")

            return net.generate_html()
        except Exception as e:
            logger.warning(f"Pyvis HTML export failed ({e}). Returning fallback SVG/HTML.")
            return f"<div style='color:white;padding:20px;'>Graph generated ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges).</div>"

