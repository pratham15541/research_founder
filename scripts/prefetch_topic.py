"""
Topic Pre-fetching & Caching Script.
Pre-fetches, embeds, and caches a complete research topic corpus into PostgreSQL/SQLite
so judged hackathon demos run with ZERO live API dependencies in under 2 seconds.
"""

import sys
import asyncio
import argparse
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.workflow import ResearchWorkflowRunner
from src.storage.db import get_db_session, init_db

async def prefetch_topic(topic: str, corpus_size: int = 80):
    print(f"==================================================")
    print(f"Pre-fetching Research Corpus for: '{topic}'")
    print(f"Target Corpus Size: {corpus_size}")
    print(f"==================================================")

    await init_db()
    runner = ResearchWorkflowRunner()

    async for session in get_db_session():
        results = await runner.run_pipeline(
            session=session,
            topic_query=topic,
            target_corpus_size=corpus_size
        )
        print(f"\n[SUCCESS] Successfully pre-cached topic '{topic}'!")
        print(f"- Total Papers: {results['corpus_size']}")
        print(f"- Silhouette Score: {results['silhouette_score']} ({results['cluster_method']})")
        print(f"- Discovered Clusters: {len(results['discovered_clusters'])}")
        print(f"- Candidate Gaps: {results['candidate_gaps_count']}")
        print(f"- Top Gaps Ranked: {len(results['ranked_gaps'])}")
        print(f"- Graph Nodes: {results['graph_summary']['nodes']}, Edges: {results['graph_summary']['edges']}")
        print(f"\nAll papers and vectors are now persisted in the compounding database.")
        print(f"Subsequent queries for this topic will hit local cache with 0 API calls.")

def main():
    parser = argparse.ArgumentParser(description="Pre-fetch topic for offline demo.")
    parser.add_argument("--topic", type=str, default="Physics-Informed Neural Networks", help="Research topic")
    parser.add_argument("--size", type=int, default=60, help="Number of papers to pull")
    args = parser.parse_args()

    asyncio.run(prefetch_topic(args.topic, args.size))

if __name__ == "__main__":
    main()

