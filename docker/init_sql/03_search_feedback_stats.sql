-- Search feedback aggregation table.
--
-- Populated by the airflow DAG `search_feedback_rerank` (hourly). Read
-- query-time by ml_gateway's /predict/search to boost/demote search
-- results based on accumulated user thumbs-up/down feedback.
--
-- One row per document with aggregate counts over ALL feedback ever
-- submitted. No per-query rolling window (v1 simplicity); later we can
-- add decay weighting or per-(query_embedding, doc) stats for more
-- targeted reranking.
--
-- The aggregation job TRUNCATEs + reinserts, so deleting this table
-- entirely is safe — the next DAG run rebuilds it. Stats are fully
-- derivable from search_feedback + query_sessions.

CREATE TABLE IF NOT EXISTS document_feedback_stats (
    document_id         UUID PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    thumbs_up           INTEGER NOT NULL DEFAULT 0,
    thumbs_down         INTEGER NOT NULL DEFAULT 0,
    clicks              INTEGER NOT NULL DEFAULT 0,
    total_impressions   INTEGER NOT NULL DEFAULT 0,  -- times doc appeared in a result set
    up_rate             DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    down_rate           DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    computed_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feedback_stats_computed_at ON document_feedback_stats(computed_at);
