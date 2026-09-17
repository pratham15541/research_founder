"""
Streamlit Interactive Dashboard for Research Landscape, Gap Matrix & Knowledge Graph.
Synchronizes 2D combinatorial matrix heatmap with interactive knowledge graph and evidence dossiers.
Uses 100% native Streamlit components to ensure responsive, error-free rendering on all devices.
"""

import sys
from pathlib import Path
import asyncio
import streamlit as st
import plotly.express as px
import pandas as pd
import streamlit.components.v1 as components
import httpx

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.workflow import ResearchWorkflowRunner
from src.storage.db import get_db_session, init_db
from src.config import settings
from src.rag.chat import ResearchChatSession
from src.rag.rag_engine import RAGEngine
from src.reports.exporter import ReportExporter
from src.evaluation.human_eval import HumanEvaluationManager

st.set_page_config(
    page_title="ResearchGapAI — Intelligence & Gap Discovery Suite",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Title & Positioning
st.title("🔬 ResearchGapAI — Intelligence & Gap Discovery Suite")
st.caption("AI-Powered Discovery of Unexplored Research Intersections with Grounded Citations, RAG Assistant, and Adversarial Self-Critique")

# Collapsible Human Guide
with st.expander("📖 Guide: How to Read the Landscape, Matrix, and Gaps", expanded=False):
    g_col1, g_col2, g_col3 = st.columns(3)
    with g_col1:
        st.markdown("""
        ### 📊 1. The 2D Matrix
        - **Rows (Axis A):** Methodological paradigms dynamically discovered via unsupervised clustering.
        - **Columns (Axis B):** Application domains dynamically extracted from the corpus.
        - **Numbers & Colors:** Count of published papers in that exact combination.
        - 🟦 **Dark Blue:** Heavily studied areas (saturated literature).
        - 🟨 **Light/Zero Cells Bordered by Dense Cells:** Prime **Candidate Research Gaps**!
        """)
    with g_col2:
        st.markdown("""
        ### 🌐 2. The Knowledge Graph
        - **Each Bubble (Node):** A real academic paper in the analyzed topic.
        - **Bubble Size:** Citation impact (larger bubbles = foundational papers).
        - **Bubble Color:** 🔵 Retrieved online | 🟣 Uploaded by you.
        - **Connecting Lines:** Semantic similarity (>0.65) between abstracts.
        - 🏝️ **Empty Spaces / Voids:** Unconnected subfields that don't talk to each other yet.
        """)
    with g_col3:
        st.markdown("""
        ### 🏆 3. Actionable Gap Dossiers & RAG
        - **Real Proposals:** Concrete paper titles, hypotheses, and kickoff plans.
        - **Feasibility:** Empirically calculated from component maturity in adjacent literature.
        - **Devil's Advocate:** Rigorous reality check pointing out potential failure modes.
        - **RAG Chat Assistant:** Ask questions directly to an AI grounded in your indexed papers.
        """)

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Configuration")
    topic_query = st.text_input(
        "Research Topic / Query",
        value="Physics-Informed Neural Networks",
        help="Enter an academic topic to discover its dimensions and unexplored intersections."
    )

    target_corpus = st.slider("Corpus Target Size", min_value=30, max_value=200, value=60, step=10)

    clustering_choice = st.selectbox(
        "Clustering Algorithm (Axis A)",
        options=[
            "Adaptive Auto (Recommended)",
            "Agglomerative (Ward Hierarchical)",
            "K-Means (Silhouette-Optimized)",
            "HDBSCAN (Density-Based)"
        ],
        index=0,
        help="Adaptive Auto dynamically tests HDBSCAN, Ward Agglomerative, and Silhouette-Optimized KMeans, selecting the taxonomy with highest cohesion and zero unassigned noise."
    )
    algo_map = {
        "Adaptive Auto (Recommended)": "auto",
        "Agglomerative (Ward Hierarchical)": "agglomerative",
        "K-Means (Silhouette-Optimized)": "kmeans",
        "HDBSCAN (Density-Based)": "hdbscan"
    }
    selected_algo = algo_map[clustering_choice]

    st.subheader("📄 Hybrid Upload")
    uploaded_files = st.file_uploader(
        "Upload Custom Academic PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload advisor-suggested papers to merge into the compounding corpus."
    )

    run_btn = st.button("🚀 Analyze Research Landscape", type="primary")
    st.divider()
    st.markdown("""
    **Platform Capabilities:**
    - **Dynamic Dual-Axis Discovery:** Discovers methods AND domains unsupervised from literature.
    - **Zero Outlier Noise:** All papers reassigned to nearest coherent cluster.
    - **FAISS Vector Indexing:** Semantic chunk retrieval for RAG assistant.
    - **Automated Literature Review:** Executive overview and thematic synthesis.
    - **Hypothesis Generator:** Formal $H_1/H_0$ and 3-phase experimental execution.
    - **Multi-Metric Evaluation:** Silhouette, Topic Coherence, Topic Diversity, Retrieval P@K, and Human Evaluation.
    - **Multi-Format Export:** Instant download in Markdown, IEEE LaTeX (.tex), and HTML.
    """)

# Backend Connectivity
BACKEND_URL = sys.modules.get("os", __import__("os")).environ.get("BACKEND_API_URL", "http://127.0.0.1:8000")

def check_backend_status() -> dict:
    try:
        with httpx.Client(timeout=1.5) as client:
            res = client.get(f"{BACKEND_URL}/health")
            if res.status_code == 200:
                return {"connected": True, "data": res.json()}
    except Exception:
        pass
    return {"connected": False, "data": None}

backend_status = check_backend_status()
with st.sidebar:
    if backend_status["connected"]:
        st.success(f"🟢 Backend Connected: {BACKEND_URL}")
        pool_size = backend_status["data"].get("proxy_pool_size", 0)
        st.caption(f"🛡️ Proxy Pool: {pool_size} rotating IPs (iplocate)")
    else:
        st.info("🟡 Standalone Mode (Direct In-Process Engine)")

# Session State Initialization
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "workflow_runner" not in st.session_state:
    st.session_state.workflow_runner = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "human_eval_records" not in st.session_state:
    st.session_state.human_eval_records = []

async def execute_in_process_pipeline(query: str, pdf_paths: list, corpus_size: int, algorithm: str = "auto"):
    await init_db()
    runner = ResearchWorkflowRunner()
    async for session in get_db_session():
        results = await runner.run_pipeline(
            session=session,
            topic_query=query,
            uploaded_pdf_paths=pdf_paths,
            target_corpus_size=corpus_size,
            clustering_algorithm=algorithm
        )
        return runner, results

def execute_backend_api_pipeline(query: str, saved_paths: list, corpus_size: int, algorithm: str = "auto"):
    for p in saved_paths:
        with open(p, "rb") as f:
            files = {"file": (p.name, f, "application/pdf")}
            upload_resp = httpx.post(f"{BACKEND_URL}/api/upload", files=files, timeout=30.0)
            upload_resp.raise_for_status()

    payload = {
        "topic_query": query,
        "target_corpus_size": corpus_size,
        "clustering_algorithm": algorithm
    }
    timeout = httpx.Timeout(
        connect=10.0,
        read=float(settings.ANALYSIS_TIMEOUT_SECONDS),
        write=60.0,
        pool=10.0,
    )
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(f"{BACKEND_URL}/api/analyze", json=payload)
            resp.raise_for_status()
            return resp.json()
    except httpx.TimeoutException as exc:
        raise RuntimeError(
            f"Backend analysis exceeded {settings.ANALYSIS_TIMEOUT_SECONDS} seconds. "
            "The pipeline may still be running; reduce the corpus size or raise ANALYSIS_TIMEOUT_SECONDS."
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Backend returned error {exc.response.status_code}: {exc.response.text}"
        ) from exc

# Pipeline Execution Trigger
if run_btn and topic_query:
    with st.spinner("Executing dynamic hybrid retrieval, dual-axis unsupervised discovery, RAG indexing & synthesis..."):
        saved_paths = []
        if uploaded_files:
            settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            for uf in uploaded_files:
                save_path = settings.UPLOAD_DIR / uf.name
                with open(save_path, "wb") as f:
                    f.write(uf.getbuffer())
                saved_paths.append(save_path)

        try:
            if backend_status["connected"]:
                results = execute_backend_api_pipeline(topic_query, saved_paths, target_corpus, selected_algo)
                st.session_state.workflow_runner = None
            else:
                runner, results = asyncio.run(execute_in_process_pipeline(topic_query, saved_paths, target_corpus, selected_algo))
                st.session_state.workflow_runner = runner

            st.session_state.analysis_results = results
            st.session_state.chat_history = []
            st.success(f"Analysis complete! Mapped {results['corpus_size']} papers across dynamic dimensions.")
        except Exception as e:
            st.error(f"Error during analysis: {e}")

# Render Results
if st.session_state.analysis_results:
    res = st.session_state.analysis_results

    # 1. Macro Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Papers Mapped", res.get("corpus_size", 0))
    with col2:
        st.metric("Silhouette Cohesion", f"{res.get('silhouette_score', 0.0)}", f"Method: {res.get('cluster_method', 'N/A')}")
    with col3:
        st.metric("Discovered Methods (Axis A)", len(res.get("discovered_clusters", {})))
    with col4:
        st.metric("Discovered Domains (Axis B)", len(res.get("matrix", {}).get("axis_b_labels", [])))
    with col5:
        st.metric("Actionable Research Gaps", len(res.get("ranked_gaps", [])))

    st.divider()

    # 2. Main Navigation Tabs
    tab_dossiers, tab_matrix, tab_graph, tab_chat, tab_review, tab_questions, tab_eval, tab_export, tab_corpus = st.tabs([
        "🏆 Top-Ranked Gaps",
        "📊 2D Matrix Heatmap",
        "🌐 Knowledge Graph",
        "💬 Research Chat (RAG)",
        "📚 Literature Review",
        "🔬 Hypotheses & Questions",
        "📈 Evaluation Scorecard",
        "📄 Export Reports",
        "🗄️ Corpus Explorer"
    ])

    # Tab 1: Top-Ranked Gap Dossiers (100% Native Streamlit Components)
    with tab_dossiers:
        st.subheader("🏆 Actionable Research Gap Dossiers")
        st.markdown(
            "These proposals represent **unexplored intersections** in the scientific literature. "
            "Both dimensions are discovered unsupervised from the papers. "
            "Each proposal is vetted by component feasibility math and checked with a **skeptical reality check**."
        )

        ranked_gaps = res.get("ranked_gaps", [])
        if not ranked_gaps:
            st.warning("No high-potential candidate gaps met the structural adjacency threshold. Try increasing the corpus target size.")

        for rank_idx, dossier in enumerate(ranked_gaps):
            proj_title = dossier.get("project_title") or f"{dossier['axis_a']} in {dossier['axis_b']}"
            core_q = dossier.get("core_research_question") or f"How can {dossier['axis_a']} be integrated into {dossier['axis_b']}?"
            why_gap = dossier.get("why_it_is_a_gap") or dossier.get("feasibility_rationale", "")
            exp_steps = dossier.get("suggested_first_experiment") or "Benchmark against standard open-source datasets in this domain."
            impact_desc = dossier.get("practical_impact") or "Bridges two isolated research communities with high potential for novel publication."

            with st.container(border=True):
                # Header row with badges
                st.caption(f":orange[**RESEARCH GAP #{rank_idx + 1}**] &nbsp; | &nbsp; 🔬 **Discovered Method:** `{dossier['axis_a']}` &nbsp; | &nbsp; 🎯 **Discovered Domain:** `{dossier['axis_b']}`")
                st.subheader(f"📌 {proj_title}")

                # Metrics row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Opportunity Score", f"{dossier['composite_score']}/5.0")
                m2.metric("Novelty Score", f"{dossier['novelty_score']}/5.0")
                m3.metric("Feasibility Score", f"{dossier['feasibility_score']}/5.0")
                m4.metric("Impact Potential", f"{dossier['impact_score']}/5.0")

                st.markdown("##### ❓ Core Research Hypothesis / Question")
                st.info(f"*{core_q}*")

                st.markdown("##### 🔍 Why This is an Unexplored Gap")
                st.write(why_gap)

                st.markdown("##### 🧪 Suggested First Experiment (Kickoff Plan)")
                st.markdown(f"**Step-by-step action:** {exp_steps}")

                st.error(f"🚨 **Devil's Advocate Reality Check (Potential Failure Modes):**\n\n{dossier['counter_argument']}")

                if impact_desc:
                    st.caption(f"🌍 **Broader Impact:** {impact_desc}")

                with st.expander(f"Inspect Decision Tree & Supporting Citations for Project #{rank_idx + 1}"):
                    st.markdown("**Transparent Decision Conditions (Why the Algorithm Picked This):**")
                    for cond in dossier.get("grounding_conditions", []):
                        st.markdown(f"- {cond}")

                    st.markdown("**Grounded Neighbor Citations (Proof that both components work in adjacent contexts):**")
                    evidence_list = dossier.get("supporting_evidence", [])
                    if evidence_list:
                        for ev in evidence_list:
                            link_url = ev.get("source_url") or f"https://doi.org/{ev.get('paper_id', '')}"
                            st.markdown(f"- **[{ev.get('title')}]({link_url})** ({ev.get('year')}) — *{ev.get('role')}*")
                    else:
                        st.write("Neighbor evidence compiled from corpus matrix.")

    # Tab 2: 2D Matrix Heatmap
    with tab_matrix:
        st.subheader("2D Density Matrix (Discovered Methodologies × Discovered Problem Regimes)")
        st.markdown("""
        **How to read this matrix:**
        - **Rows (Axis A):** Discovered algorithmic methodologies clustered from the literature.
        - **Columns (Axis B):** Discovered application domains and problem settings.
        - **Numbers & Color Intensity:** Count of papers published in that exact intersection.
        - 🟦 **Dark Blue / High Numbers:** Heavily studied areas (saturated literature).
        - 🟨 **Light or Zero with Dark Neighbors:** Prime **Combinatorial Research Gaps**!
        """)

        matrix_data = res["matrix"]
        density_table = matrix_data["density_table"]
        df_matrix = pd.DataFrame(density_table).T  # Axis A as rows, Axis B as columns

        fig = px.imshow(
            df_matrix,
            labels=dict(x="Axis B: Discovered Domain", y="Axis A: Discovered Methodology", color="Paper Count"),
            x=df_matrix.columns,
            y=df_matrix.index,
            color_continuous_scale="Blues",
            text_auto=True,
            aspect="auto"
        )
        fig.update_layout(
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font=dict(color="#f8fafc", size=13),
            height=460
        )
        st.plotly_chart(fig)

        # Interactive Cell Inspector
        st.markdown("#### 🔍 Matrix Cell Inspector")
        inspect_col1, inspect_col2 = st.columns(2)
        with inspect_col1:
            sel_axis_a = st.selectbox("Select Axis A (Discovered Method)", matrix_data["axis_a_labels"])
        with inspect_col2:
            sel_axis_b = st.selectbox("Select Axis B (Discovered Domain)", matrix_data["axis_b_labels"])

        target_cell = next(
            (c for c in matrix_data["cells"] if c["axis_a"] == sel_axis_a and c["axis_b"] == sel_axis_b),
            None
        )
        if target_cell:
            sparsity_badge = "⚠️ Sparse Gap Opportunity (Low literature count)" if target_cell["is_sparse"] else "✅ Well-Studied / Saturated"
            st.info(
                f"**Cell**: '{sel_axis_a}' × '{sel_axis_b}' | "
                f"**Paper Count**: {target_cell['paper_count']} papers | "
                f"**Status**: {sparsity_badge}"
            )
            if target_cell["papers"]:
                st.markdown("**Published Papers in this exact cell:**")
                for p in target_cell["papers"][:5]:
                    url = p.get("source_url") or f"https://doi.org/{p.get('doi', '')}"
                    st.markdown(f"- **[{p.get('title')}]({url})** ({p.get('year')}) — Citations: {p.get('citation_count')}")
            else:
                st.write("Zero papers exist in this intersection. This represents an empirical void in the retrieved corpus.")

    # Tab 3: Synchronized Knowledge Graph
    with tab_graph:
        st.subheader("🌐 Citation & Semantic Similarity Network")
        st.markdown("""
        **What does this graph show?**
        - **Each Bubble (Node):** A real academic paper in the analyzed topic.
        - **Bubble Size:** Relative citation count (larger bubbles = foundational papers).
        - **Bubble Color:** 🔵 Retrieved online | 🟣 Uploaded by you.
        - **Connecting Lines (Edges):** Semantic cosine similarity (>0.65) between abstracts.
        - 🏝️ **Disconnected Islands / Empty Whitespace:** Visual proof of gaps! When two clusters have no connecting bridges, it proves that the scientific communities using those techniques are currently working in isolation.
        """)
        components.html(res["graph_html"], height=580, scrolling=False)

    # Tab 4: Research Chat Assistant (RAG)
    with tab_chat:
        st.subheader("💬 Context-Aware Research Chat Assistant")
        st.caption("Chat with an AI grounded in your analyzed papers and section chunks. Backed by FAISS vector index.")

        # Render conversation history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("citations"):
                    with st.expander("📚 Grounded Citations & Retrieved Chunks", expanded=False):
                        for c in msg["citations"]:
                            c_title = c.get("paper_title") or c.get("title", "Untitled Publication")
                            c_sec = c.get("section_type") or c.get("section", "excerpt")
                            c_snip = c.get("snippet") or c.get("text", "")
                            st.markdown(f"- **{c_title}** `[{c_sec}]`")
                            if c_snip:
                                st.caption(f'"{c_snip}"')

        # Chat input box
        user_query_input = st.chat_input("Ask a research question (e.g., 'What are the main convergence issues reported in Navier-Stokes PINNs?')")
        if user_query_input:
            # Append user turn
            st.session_state.chat_history.append({"role": "user", "content": user_query_input})
            with st.chat_message("user"):
                st.markdown(user_query_input)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving grounded chunks from FAISS vector store..."):
                    answer_text = ""
                    citations_list = []

                    # If backend connected, call /api/chat
                    if backend_status["connected"]:
                        try:
                            with httpx.Client(timeout=30.0) as client:
                                chat_res = client.post(f"{BACKEND_URL}/api/chat", json={"message": user_query_input, "top_k": 4})
                                if chat_res.status_code == 200:
                                    chat_data = chat_res.json()
                                    answer_text = chat_data.get("answer", "")
                                    citations_list = chat_data.get("citations", [])
                                else:
                                    answer_text = f"Backend error: {chat_res.text}"
                        except Exception as ce:
                            answer_text = f"Failed to reach backend chat endpoint: {ce}"
                    else:
                        # Direct in-process RAG
                        runner = st.session_state.workflow_runner
                        if runner and runner.vector_index.count() > 0:
                            rag_res = RAGEngine.query(
                                user_query=user_query_input,
                                faiss_index=runner.vector_index,
                                embedder=runner.embedder,
                                top_k=4
                            )
                            answer_text = rag_res.get("answer", "")
                            citations_list = rag_res.get("citations", [])
                        else:
                            answer_text = "Vector index is currently empty. Run an analysis first."

                    st.markdown(answer_text)
                    if citations_list:
                        with st.expander("📚 Grounded Citations & Retrieved Chunks", expanded=False):
                            for c in citations_list:
                                c_title = c.get("paper_title") or c.get("title", "Untitled Publication")
                                c_sec = c.get("section_type") or c.get("section", "excerpt")
                                c_snip = c.get("snippet") or c.get("text", "")
                                st.markdown(f"- **{c_title}** `[{c_sec}]`")
                                if c_snip:
                                    st.caption(f'"{c_snip}"')

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer_text,
                        "citations": citations_list
                    })

    # Tab 5: Automated Literature Review
    with tab_review:
        st.subheader("📚 Automated Literature Review Synthesis")
        st.caption("Synthesized cross-paper survey structuring methodological themes, domain applications, and open voids.")

        lit_review = res.get("literature_review", {})
        if lit_review:
            # Executive Summary
            exec_sum = lit_review.get("executive_summary")
            if exec_sum:
                st.markdown("### 📋 Executive Summary")
                st.write(exec_sum)
                st.divider()

            # Thematic Breakdown
            thematic_themes = lit_review.get("thematic_breakdown", [])
            if thematic_themes:
                st.markdown("### 🧩 Thematic Breakdown Across Discovered Clusters")
                for theme in thematic_themes:
                    with st.container(border=True):
                        st.subheader(f"Theme: {theme.get('cluster_name')}")
                        st.markdown(f"**Description & Patterns:** {theme.get('description')}")
                        st.markdown(f"**Key Publications:** {', '.join(theme.get('key_papers', []))}")
                        st.info(f"**Reported Limitations:** {theme.get('reported_limitations')}")
                st.divider()

            # Cross-Cutting Synthesis & Voids
            comp_syn = lit_review.get("comparative_synthesis")
            if comp_syn:
                st.markdown("### 🔬 Comparative Synthesis & Open Voids")
                st.write(comp_syn)
            voids = lit_review.get("identified_voids")
            if voids:
                st.warning(f"**Identified Research Voids:** {voids}")

            # If structured breakdown was empty, render the markdown review
            if not exec_sum and not thematic_themes and lit_review.get("review_markdown"):
                st.markdown(lit_review.get("review_markdown"))
            else:
                with st.expander("📄 View & Copy Full Markdown Review"):
                    st.code(lit_review.get("review_markdown", ""), language="markdown")
        else:
            st.info("Literature review synthesis is being generated or was not returned.")

    # Tab 6: Formal Hypotheses & Research Questions
    with tab_questions:
        st.subheader("🔬 Evidence-Grounded Research Questions & Hypotheses")
        st.markdown("Formal hypothesis formulations with testable variables and 3-phase experimental execution plans.")

        rq_list = res.get("research_questions", [])
        if rq_list:
            for idx, rq in enumerate(rq_list):
                with st.container(border=True):
                    st.caption(f":blue[**RESEARCH QUESTION #{idx + 1}**]")
                    st.subheader(f"❓ {rq.get('primary_research_question')}")

                    h_col1, h_col2 = st.columns(2)
                    with h_col1:
                        st.success(f"**Alternative Hypothesis ($H_1$):**\n\n{rq.get('primary_hypothesis_h1')}")
                    with h_col2:
                        st.info(f"**Null Hypothesis ($H_0$):**\n\n{rq.get('null_hypothesis_h0')}")

                    st.markdown("##### 📊 Variables")
                    v_dict = rq.get("variables", {})
                    if isinstance(v_dict, dict):
                        indep = v_dict.get("independent", "N/A")
                        dep = v_dict.get("dependent", "N/A")
                        ctrl = v_dict.get("controlled", v_dict.get("control", "N/A"))
                    else:
                        indep, dep, ctrl = str(v_dict), "N/A", "N/A"
                    st.markdown(f"- **Independent Variables:** {indep}")
                    st.markdown(f"- **Dependent Variables:** {dep}")
                    st.markdown(f"- **Control Variables:** {ctrl}")

                    st.markdown("##### 🧪 Three-Phase Experimental Execution")
                    exp_phases = rq.get("experimental_phases", [])
                    p1, p2, p3 = st.columns(3)
                    if isinstance(exp_phases, list):
                        phase1 = exp_phases[0] if len(exp_phases) > 0 else "N/A"
                        phase2 = exp_phases[1] if len(exp_phases) > 1 else "N/A"
                        phase3 = exp_phases[2] if len(exp_phases) > 2 else "N/A"
                        with p1:
                            st.markdown(f"**Phase 1**\n\n{phase1}")
                        with p2:
                            st.markdown(f"**Phase 2**\n\n{phase2}")
                        with p3:
                            st.markdown(f"**Phase 3**\n\n{phase3}")
                    elif isinstance(exp_phases, dict):
                        with p1:
                            st.markdown(f"**Phase 1: Baselines**\n\n{exp_phases.get('phase_1_baseline_setup', 'N/A')}")
                        with p2:
                            st.markdown(f"**Phase 2: Hybridization**\n\n{exp_phases.get('phase_2_hybridization', 'N/A')}")
                        with p3:
                            st.markdown(f"**Phase 3: Validation**\n\n{exp_phases.get('phase_3_stress_testing', 'N/A')}")
                    else:
                        st.markdown(str(exp_phases))

                    contrib = rq.get("expected_contributions") or rq.get("expected_contribution", "")
                    if contrib:
                        st.caption(f"🌟 **Expected Scientific Contribution:** {contrib}")
        else:
            st.info("No research question protocols generated yet.")

    # Tab 7: Multi-Metric Evaluation Framework
    with tab_eval:
        st.subheader("📈 Multi-Metric Evaluation Framework")
        st.markdown("Quantitative intrinsic clustering metrics, topic coherence, retrieval metrics, and expert scorecard.")

        eval_data = res.get("evaluation_metrics", {})
        retrieval_data = eval_data.get("retrieval", {})

        e_col1, e_col2, e_col3 = st.columns(3)
        with e_col1:
            st.metric("Silhouette Cohesion Score", f"{eval_data.get('silhouette_score', 0.0)}", "Intrinsic Cluster Quality")
        with e_col2:
            st.metric("Topic Coherence (C_v proxy)", f"{eval_data.get('topic_coherence', 0.0)}", "Term Semantic Relatedness")
        with e_col3:
            st.metric("Topic Diversity", f"{eval_data.get('topic_diversity', 0.0)}", "Ratio of Unique Topic Terms")

        st.markdown("#### 🎯 Extrinsic Retrieval Performance (Corpus Grounding)")
        r_col1, r_col2, r_col3 = st.columns(3)
        with r_col1:
            st.metric("Precision@K", f"{retrieval_data.get('precision_at_k', 0.0)}")
        with r_col2:
            st.metric("Recall@K", f"{retrieval_data.get('recall_at_k', 0.0)}")
        with r_col3:
            f1_val = retrieval_data.get("f1_score", retrieval_data.get("f1_at_k", 0.0))
            st.metric("F1 Score", f"{f1_val}")

        st.divider()

        # Human Evaluation Section
        st.markdown("#### 🧑‍🔬 Human Evaluation Scorecard (Expert Feedback)")
        with st.form("human_eval_form"):
            gap_choices = [f"#{idx+1}: {g.get('project_title', g['axis_a'] + ' × ' + g['axis_b'])}" for idx, g in enumerate(res.get("ranked_gaps", []))]
            selected_gap_label = st.selectbox("Select Research Gap Dossier to Rate:", gap_choices) if gap_choices else None

            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                rel_score = st.slider("Relevance to Topic (1=Irrelevant, 5=Highly Relevant)", 1, 5, 5)
            with f_col2:
                nov_score = st.slider("Scientific Novelty (1=Trivial, 5=Groundbreaking)", 1, 5, 4)
            with f_col3:
                exp_score = st.slider("Technical Explainability (1=Vague, 5=Rigorous)", 1, 5, 4)

            hallucination_flag = st.checkbox("🚩 Evidence Hallucination Detected (Invalid or mismatched citations)")
            expert_note = st.text_area("Expert Comments / Critiques:", placeholder="Note any missing baseline architectures or domain subtleties...")
            submit_eval = st.form_submit_button("Submit Expert Evaluation Record")

            if submit_eval and selected_gap_label:
                gap_idx = int(selected_gap_label.split(":")[0].replace("#", "")) - 1
                gap_rec = res["ranked_gaps"][gap_idx]
                eval_record = HumanEvaluationManager.create_evaluation_record(
                    gap_id=f"gap_{gap_idx}",
                    project_title=gap_rec.get("project_title", selected_gap_label),
                    relevance_score=rel_score,
                    novelty_score=nov_score,
                    explainability_score=exp_score,
                    hallucination_detected=hallucination_flag,
                    expert_comments=expert_note
                )
                st.session_state.human_eval_records.append(eval_record)

                if backend_status["connected"]:
                    try:
                        with httpx.Client(timeout=5.0) as client:
                            client.post(f"{BACKEND_URL}/api/evaluation/human", json=eval_record)
                    except Exception:
                        pass
                st.success("Evaluation record recorded successfully!")

        # Aggregate Human Scorecard
        if st.session_state.human_eval_records:
            st.markdown("##### 🏆 Current Aggregate Expert Scorecard")
            agg = HumanEvaluationManager.aggregate_human_scores(st.session_state.human_eval_records)
            a_col1, a_col2, a_col3, a_col4 = st.columns(4)
            with a_col1:
                st.metric("Total Expert Reviews", agg["total_ratings"])
            with a_col2:
                st.metric("Mean Relevance", f"{agg['mean_relevance']}/5.0")
            with a_col3:
                st.metric("Mean Novelty", f"{agg['mean_novelty']}/5.0")
            with a_col4:
                st.metric("Hallucination Rate", f"{agg['hallucination_rate_percent']}%")

    # Tab 8: Multi-Format Report Exporter
    with tab_export:
        st.subheader("📄 Multi-Format Report Exporter")
        st.markdown("Download publication-ready briefings and drafts in academic formats.")

        topic_str = res.get("topic_query", "Research_Topic")
        topic_clean = topic_str.replace(" ", "_")
        lit_rev_obj = res.get("literature_review")
        rq_objs = res.get("research_questions")

        # Generate formats
        md_text = ReportExporter.export_markdown(topic_str, res, lit_rev_obj, rq_objs)
        tex_text = ReportExporter.export_latex(topic_str, res, lit_rev_obj, rq_objs)
        html_text = ReportExporter.export_html(topic_str, res, lit_rev_obj)

        down_col1, down_col2, down_col3 = st.columns(3)
        with down_col1:
            st.download_button(
                label="📥 Download Markdown Dossier (.md)",
                data=md_text,
                file_name=f"ResearchGapAI_{topic_clean}.md",
                mime="text/markdown"
            )
            st.caption("Standard Markdown format with headers, tables, and clickable references.")
        with down_col2:
            st.download_button(
                label="📥 Download IEEE Conference Paper (.tex)",
                data=tex_text,
                file_name=f"ResearchGapAI_{topic_clean}.tex",
                mime="application/x-latex"
            )
            st.caption("Ready-to-compile IEEEtran 2-column LaTeX document with abstract & sections.")
        with down_col3:
            st.download_button(
                label="📥 Download Standalone Printable (.html)",
                data=html_text,
                file_name=f"ResearchGapAI_{topic_clean}.html",
                mime="text/html"
            )
            st.caption("Self-contained HTML briefing with embedded responsive styles.")

    # Tab 9: Ingested Corpus Explorer
    with tab_corpus:
        st.subheader("🗄️ Ingested Corpus Explorer")
        st.markdown("Browse all deduplicated academic papers fetched and indexed for this topic.")
        df_corpus = pd.DataFrame(res.get("papers", []))
        if not df_corpus.empty:
            cols_to_show = ["title", "year", "citation_count", "source", "axis_a_tag", "axis_b_tag", "is_uploaded"]
            existing_cols = [c for c in cols_to_show if c in df_corpus.columns]
            st.dataframe(df_corpus[existing_cols])
        else:
            st.info("No papers currently loaded.")
