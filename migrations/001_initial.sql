-- Bids, parsed documents, chunks, ingestion jobs, and agent traces.

CREATE TABLE IF NOT EXISTS bids (
    id uuid PRIMARY KEY,
    folder_name text NOT NULL UNIQUE,
    path text NOT NULL,
    status text NOT NULL DEFAULT 'pending',
    first_seen_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS source_files (
    id uuid PRIMARY KEY,
    bid_id uuid NOT NULL REFERENCES bids (id) ON DELETE CASCADE,
    relative_path text NOT NULL,
    sha256 text NOT NULL DEFAULT '',
    doc_type text NOT NULL DEFAULT 'rfp',
    addendum_number integer,
    parse_status text NOT NULL DEFAULT 'pending',
    error text,
    parsed_at timestamptz,
    CONSTRAINT uq_source_file_path UNIQUE (bid_id, relative_path)
);

CREATE TABLE IF NOT EXISTS sections (
    id uuid PRIMARY KEY,
    document_id uuid NOT NULL REFERENCES source_files (id) ON DELETE CASCADE,
    heading_path text[] NOT NULL DEFAULT '{}',
    level integer NOT NULL DEFAULT 0,
    ordinal integer NOT NULL DEFAULT 0,
    page_start integer,
    page_end integer,
    body_text text NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS tables (
    id uuid PRIMARY KEY,
    section_id uuid NOT NULL REFERENCES sections (id) ON DELETE CASCADE,
    page_number integer,
    markdown text NOT NULL DEFAULT '',
    rows_json jsonb NOT NULL DEFAULT '[]'::jsonb,
    caption text
);

CREATE TABLE IF NOT EXISTS chunks (
    id uuid PRIMARY KEY,
    section_id uuid REFERENCES sections (id) ON DELETE SET NULL,
    source_file_id uuid NOT NULL REFERENCES source_files (id) ON DELETE CASCADE,
    bid_id uuid NOT NULL REFERENCES bids (id) ON DELETE CASCADE,
    bid_folder text NOT NULL,
    ordinal integer NOT NULL,
    text text NOT NULL,
    page_number integer,
    token_count integer NOT NULL DEFAULT 0,
    heading_path text[] NOT NULL DEFAULT '{}',
    doc_type text NOT NULL DEFAULT 'rfp',
    addendum_number integer,
    file_name text NOT NULL,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(text, ''))) STORED,
    embedded_at timestamptz,
    qdrant_point_id text,
    CONSTRAINT uq_chunk_ordinal UNIQUE (source_file_id, ordinal)
);

CREATE INDEX IF NOT EXISTS ix_chunks_tsv ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS ix_chunks_bid_folder ON chunks (bid_folder);
CREATE INDEX IF NOT EXISTS ix_chunks_embedded ON chunks (embedded_at);

CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id uuid PRIMARY KEY,
    bid_id uuid NOT NULL REFERENCES bids (id) ON DELETE CASCADE,
    status text NOT NULL DEFAULT 'pending',
    attempt integer NOT NULL DEFAULT 0,
    next_run_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    finished_at timestamptz,
    error text
);

CREATE TABLE IF NOT EXISTS agent_runs (
    id uuid PRIMARY KEY,
    bid_folder text,
    mode text NOT NULL,
    status text NOT NULL DEFAULT 'running',
    question text,
    plan jsonb,
    started_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    error text
);

CREATE TABLE IF NOT EXISTS agent_steps (
    id uuid PRIMARY KEY,
    run_id uuid NOT NULL REFERENCES agent_runs (id) ON DELETE CASCADE,
    agent text NOT NULL,
    input_json jsonb,
    output_json jsonb,
    tokens integer NOT NULL DEFAULT 0,
    latency_ms integer NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS extracted_fields (
    id uuid PRIMARY KEY,
    run_id uuid NOT NULL REFERENCES agent_runs (id) ON DELETE CASCADE,
    bid_folder text NOT NULL,
    field_name text NOT NULL,
    value text,
    sources jsonb NOT NULL DEFAULT '[]'::jsonb,
    confidence double precision NOT NULL DEFAULT 0,
    notes text NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS addendum_changes (
    id uuid PRIMARY KEY,
    run_id uuid NOT NULL REFERENCES agent_runs (id) ON DELETE CASCADE,
    bid_folder text NOT NULL,
    field_name text NOT NULL,
    previous_value text,
    new_value text,
    addendum_number integer,
    file_name text,
    page_number integer
);

DROP TABLE IF EXISTS alembic_version;
