"""
pipeline/nodes/memory.py — Vector memory nodes (Milvus retrieval and write-back).

Nodes:
  retrieve_context_node  — Node 1: query Milvus for historical correction rules.
  update_memory_node     — Node 3.5 (replan path): persist critic rejection feedback.

Both use hybrid multi-tenant routing:
  - Physical partitions (ProjectA, ProjectB, Global_Rules) for coarse isolation.
  - A 'style' scalar field for fine-grained filtering within a partition.

Standalone usage (dummy state, no real Milvus required — will log a graceful warning):
    python -m pipeline.nodes.memory
"""

from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

from pipeline.shared import (
    MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION,
    PipelineState, logger, log_input, log_output,
)


def retrieve_context_node(state: PipelineState) -> dict:
    """Node 1: Query Milvus for historical motion correction rules.

    Implements hybrid multi-tenant retrieval:
      - Searches across the active project partition AND Global_Rules simultaneously
        so universal physics/kinematics rules are always considered.
      - Filters results by animation style (or the universal 'physics' tag) to keep
        retrieved context relevant to the current task.

    Gracefully degrades if Milvus is offline — the pipeline continues without context.
    """
    log_input("retrieve_context", ["prompt"], state)
    logger.info("[retrieve_context] Connecting to Milvus at %s:%s ...", MILVUS_HOST, MILVUS_PORT)
    historical_context = []

    # Placeholders — swap for dynamic values when multi-project support is added.
    active_project  = "ProjectA"
    requested_style = "combat"

    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        collection = Collection(MILVUS_COLLECTION)
        collection.load()
        logger.info("[retrieve_context] Milvus connected. Collection '%s' loaded.", MILVUS_COLLECTION)

        query_model     = SentenceTransformer("all-MiniLM-L6-v2")
        query_embedding = query_model.encode(state["prompt"]).tolist()
        logger.debug("[retrieve_context] Query embedding generated (%d dims).", len(query_embedding))

        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
            data            = [query_embedding],
            anns_field      = "embedding",
            param           = search_params,
            limit           = 5,
            expr            = f"style == '{requested_style}' or style == 'physics'",
            partition_names = [active_project, "Global_Rules"],
            output_fields   = ["text"],
        )

        for hits in results:
            for hit in hits:
                text = hit.entity.get("text")
                if text:
                    historical_context.append(text)

        logger.info(
            "[retrieve_context] Retrieved %d correction rule(s) "
            "(partitions: %s + Global_Rules, style filter: '%s' | 'physics').",
            len(historical_context), active_project, requested_style,
        )

    except Exception as e:
        logger.warning(
            "[retrieve_context] Vector DB not available — continuing without context. Error: %s", e
        )

    result = {"historical_context": historical_context}
    log_output("retrieve_context", result)
    return result


def update_memory_node(state: PipelineState) -> dict:
    """Node 3.5 (replan path): Vectorize critic rejection feedback and persist to Milvus.

    Only activates on genuine rejections — human approvals and no-data passes are
    skipped so the collection accumulates only actionable correction rules.

    Hybrid multi-tenant routing:
      current_project : physical partition target (ProjectA / ProjectB / Global_Rules)
      motion_style    : scalar metadata tag for style-filtered retrieval at query time.
    """
    log_input("update_memory", ["critic_feedback", "critic_score", "iteration_count"], state)
    feedback = state.get("critic_feedback", "")

    if not feedback or "approved" in feedback.lower():
        logger.info("[update_memory] No rejection feedback to store — skipping memory update.")
        log_output("update_memory", {})
        return state

    logger.info("[update_memory] Rejection detected — vectorizing feedback for long-term memory...")
    logger.debug("[update_memory] Feedback: %s", feedback)

    # Placeholders — swap for dynamic values when multi-project support is added.
    current_project = "ProjectA"
    motion_style    = "combat"

    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding = model.encode(feedback).tolist()
        logger.debug("[update_memory] Embedding generated (%d dims).", len(embedding))

        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        collection = Collection(MILVUS_COLLECTION)
        collection.load()
        collection.insert([[embedding], [feedback], [motion_style]], partition_name=current_project)
        collection.flush()
        logger.info(
            "[update_memory] Correction rule saved — partition='%s', style='%s'.",
            current_project, motion_style,
        )

    except Exception as e:
        logger.warning(
            "[update_memory] Could not write to Milvus — pipeline will continue. Error: %s", e
        )

    log_output("update_memory", {
        "stored_feedback": feedback,
        "partition": current_project,
        "style": motion_style,
    })
    return state


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    dummy_state: PipelineState = {
        "prompt": "A character performs a heavy broadsword swing.",
        "historical_context": [],
        "laban_plan": {},
        "motion_tensor_path": "",
        "critic_score": 0.0,
        "critic_feedback": "",
        "iteration_count": 0,
        "final_asset_path": "",
    }

    print("=== retrieve_context_node (expect graceful Milvus warning) ===")
    out = retrieve_context_node(dummy_state)
    print("  historical_context:", out["historical_context"])

    print("\n=== update_memory_node — no rejection (should skip) ===")
    dummy_state["critic_feedback"] = "Human approved."
    update_memory_node(dummy_state)

    print("\n=== update_memory_node — with rejection feedback ===")
    dummy_state["critic_feedback"] = "Kinematic FAIL: foot sliding detected."
    update_memory_node(dummy_state)
