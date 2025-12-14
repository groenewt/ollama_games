"""DuckDB schema definitions for persistent storage."""

ROLES_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS roles (
    role_id VARCHAR PRIMARY KEY,
    name VARCHAR UNIQUE NOT NULL,
    description VARCHAR,

    -- Model configuration
    endpoint VARCHAR NOT NULL,
    model VARCHAR NOT NULL,
    system_prompt TEXT,

    -- Game-specific instructions (JSON object: game_id -> strategy text, empty = all games)
    game_instructions JSON DEFAULT '{}',

    -- Default parameters
    temperature DOUBLE DEFAULT 0.7,
    top_p DOUBLE DEFAULT 0.9,
    top_k INTEGER DEFAULT 40,
    repeat_penalty DOUBLE DEFAULT 1.1,

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_roles_name ON roles(name);
CREATE INDEX IF NOT EXISTS idx_roles_active ON roles(is_active);
"""

# Optional: Session tracking for role usage analytics
ROLE_SESSIONS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS role_sessions (
    session_id VARCHAR PRIMARY KEY,
    role_id VARCHAR,
    game_type VARCHAR NOT NULL,
    player_position INTEGER NOT NULL,
    num_rounds INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (role_id) REFERENCES roles(role_id)
);
"""

# ============================================================
# NEW: Enhanced schemas for metrics, data, and strategy capture
# ============================================================

# Sessions table with token totals and performance metrics
SESSIONS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id VARCHAR PRIMARY KEY,
    game_type VARCHAR NOT NULL,
    game_name VARCHAR,
    num_rounds INTEGER NOT NULL,
    num_players INTEGER NOT NULL DEFAULT 2,
    runtime_mode VARCHAR,

    -- Token totals (aggregated from all interactions)
    total_prompt_tokens INTEGER DEFAULT 0,
    total_completion_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    estimated_cost_usd DOUBLE DEFAULT 0.0,

    -- Performance metrics
    total_response_time_ms DOUBLE DEFAULT 0.0,
    avg_response_time_ms DOUBLE DEFAULT 0.0,
    min_response_time_ms DOUBLE,
    max_response_time_ms DOUBLE,

    -- Success metrics
    total_requests INTEGER DEFAULT 0,
    successful_requests INTEGER DEFAULT 0,
    failed_requests INTEGER DEFAULT 0,
    parse_success_rate DOUBLE,

    -- Custom game configuration
    uses_custom_payoffs BOOLEAN DEFAULT FALSE,
    payoff_matrix JSON,
    game_actions JSON,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    status VARCHAR DEFAULT 'pending'
);

CREATE INDEX IF NOT EXISTS idx_sessions_game_type ON sessions(game_type);
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
"""

# Full prompt/response storage for comprehensive analysis
INTERACTIONS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS interactions (
    interaction_id VARCHAR PRIMARY KEY,
    session_id VARCHAR NOT NULL,
    round_number INTEGER NOT NULL,
    player_id INTEGER NOT NULL,

    -- Model info
    model VARCHAR NOT NULL,
    endpoint VARCHAR NOT NULL,

    -- Prompt and response (compressed in Parquet, stored as TEXT here for queries)
    prompt TEXT NOT NULL,
    raw_response TEXT NOT NULL,
    parsed_action VARCHAR,

    -- Parsing status
    was_parsed BOOLEAN DEFAULT FALSE,
    was_normalized BOOLEAN DEFAULT FALSE,

    -- Token metrics
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,

    -- Timing
    response_time_ms DOUBLE,

    -- Decision reasoning capture (experimental)
    reasoning_trace TEXT,
    alternatives_considered JSON,
    confidence_score DOUBLE,

    -- Inference parameters
    temperature DOUBLE,
    top_p DOUBLE,
    top_k INTEGER,
    repeat_penalty DOUBLE,

    -- Timestamps
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_interactions_session ON interactions(session_id);
CREATE INDEX IF NOT EXISTS idx_interactions_model ON interactions(model);
CREATE INDEX IF NOT EXISTS idx_interactions_round ON interactions(session_id, round_number);
CREATE INDEX IF NOT EXISTS idx_interactions_player ON interactions(session_id, player_id);
"""

# Per-request token metrics for detailed cost tracking
TOKEN_METRICS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS token_metrics (
    metric_id VARCHAR PRIMARY KEY,
    session_id VARCHAR NOT NULL,
    round_number INTEGER NOT NULL,
    player_id INTEGER NOT NULL,
    model VARCHAR NOT NULL,

    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    estimated_cost_usd DOUBLE DEFAULT 0.0,

    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_token_metrics_session ON token_metrics(session_id);
CREATE INDEX IF NOT EXISTS idx_token_metrics_model ON token_metrics(model);
CREATE INDEX IF NOT EXISTS idx_token_metrics_timestamp ON token_metrics(timestamp);
"""

# Model cost configuration (for estimating API costs)
MODEL_COSTS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS model_costs (
    model VARCHAR PRIMARY KEY,
    provider VARCHAR NOT NULL,
    cost_per_prompt_token DOUBLE DEFAULT 0.0,
    cost_per_completion_token DOUBLE DEFAULT 0.0,
    is_local BOOLEAN DEFAULT TRUE,
    notes VARCHAR,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Default values for common local models (free)
INSERT OR IGNORE INTO model_costs (model, provider, cost_per_prompt_token, cost_per_completion_token, is_local)
VALUES
    ('llama3.2', 'ollama', 0.0, 0.0, TRUE),
    ('llama3', 'ollama', 0.0, 0.0, TRUE),
    ('qwen3', 'ollama', 0.0, 0.0, TRUE),
    ('tinyllama', 'ollama', 0.0, 0.0, TRUE),
    ('granite', 'ollama', 0.0, 0.0, TRUE);
"""

# Strategy detection results
STRATEGY_RESULTS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS strategy_results (
    result_id VARCHAR PRIMARY KEY,
    session_id VARCHAR NOT NULL,
    player_id INTEGER NOT NULL,

    -- Detected strategy
    strategy_type VARCHAR NOT NULL,
    confidence DOUBLE NOT NULL,
    detection_method VARCHAR,

    -- Strategy details
    cooperation_rate DOUBLE,
    defection_rate DOUBLE,
    memory_effect_lag1 DOUBLE,
    memory_effect_lag2 DOUBLE,
    memory_effect_lag3 DOUBLE,

    -- Equilibrium comparison
    nash_distance DOUBLE,
    pareto_optimal_rate DOUBLE,

    -- Timestamps
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_strategy_session ON strategy_results(session_id);
CREATE INDEX IF NOT EXISTS idx_strategy_type ON strategy_results(strategy_type);
"""

# All schema DDLs for easy initialization
ALL_SCHEMAS = [
    ROLES_TABLE_DDL,
    ROLE_SESSIONS_TABLE_DDL,
    SESSIONS_TABLE_DDL,
    INTERACTIONS_TABLE_DDL,
    TOKEN_METRICS_TABLE_DDL,
    MODEL_COSTS_TABLE_DDL,
    STRATEGY_RESULTS_TABLE_DDL,
]
