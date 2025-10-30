"""Initial schema

Revision ID: 001
Revises:
Create Date: 2025-01-15 00:00:00.000000

"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create initial database schema."""

    # Create organizations table
    op.create_table(
        "organizations",
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("api_key", sa.String(255), nullable=False),
        sa.Column("domain", sa.String(100), nullable=True),
        sa.Column("tier", sa.String(50), nullable=True),
        sa.Column("monthly_trace_limit", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.PrimaryKeyConstraint("organization_id"),
    )
    op.create_index(
        "ix_organizations_api_key", "organizations", ["api_key"], unique=True
    )

    # Create traces table
    op.create_table(
        "traces",
        sa.Column("trace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("application_id", sa.String(255), nullable=True),
        sa.Column("application_type", sa.String(100), nullable=False),
        sa.Column("session_id", sa.String(255), nullable=True),
        sa.Column("model", sa.String(255), nullable=False),
        sa.Column("input_text", sa.Text(), nullable=False),
        sa.Column("output_text", sa.Text(), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("rl_episode", postgresql.JSONB(), nullable=True),
        sa.Column("policy_info", postgresql.JSONB(), nullable=True),
        sa.Column("cumulative_reward", sa.Float(), nullable=True),
        sa.Column("episode_length", sa.Integer(), nullable=True),
        sa.Column("retrieved_contexts", postgresql.JSONB(), nullable=True),
        sa.Column("domain", sa.String(100), nullable=True),
        sa.Column("specialty", sa.String(100), nullable=True),
        sa.Column("patient_context", sa.Text(), nullable=True),
        sa.Column("market_context", sa.Text(), nullable=True),
        sa.Column("environment_context", sa.Text(), nullable=True),
        sa.Column("additional_metadata", postgresql.JSONB(), nullable=True),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["organization_id"],
            ["organizations.organization_id"],
        ),
        sa.PrimaryKeyConstraint("trace_id"),
    )
    op.create_index("ix_traces_created_at", "traces", ["created_at"])
    op.create_index("ix_traces_domain", "traces", ["domain"])
    op.create_index("ix_traces_session_id", "traces", ["session_id"])
    op.create_index(
        "idx_org_domain_created", "traces", ["organization_id", "domain", "created_at"]
    )
    op.create_index("idx_model_domain", "traces", ["model", "domain"])

    # Create evaluation_results table
    op.create_table(
        "evaluation_results",
        sa.Column("evaluation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("trace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("overall_passed", sa.Boolean(), nullable=False),
        sa.Column(
            "overall_severity",
            sa.Enum("LOW", "MEDIUM", "HIGH", "CRITICAL", name="severity"),
            nullable=True,
        ),
        sa.Column(
            "status",
            sa.Enum(
                "PENDING", "RUNNING", "COMPLETED", "FAILED", name="evaluationstatus"
            ),
            nullable=True,
        ),
        sa.Column("total_execution_time_ms", sa.Float(), nullable=False),
        sa.Column("cost", sa.Float(), nullable=True),
        sa.Column("tier1_results", postgresql.JSONB(), nullable=True),
        sa.Column("tier2_results", postgresql.JSONB(), nullable=True),
        sa.Column("tier3_results", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["trace_id"],
            ["traces.trace_id"],
        ),
        sa.PrimaryKeyConstraint("evaluation_id"),
    )
    op.create_index(
        "ix_evaluation_results_created_at", "evaluation_results", ["created_at"]
    )
    op.create_index("ix_evaluation_results_status", "evaluation_results", ["status"])
    op.create_index(
        "idx_trace_created", "evaluation_results", ["trace_id", "created_at"]
    )

    # Create evaluation_issues table
    op.create_table(
        "evaluation_issues",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("evaluation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("issue_type", sa.String(255), nullable=False),
        sa.Column(
            "severity",
            sa.Enum("LOW", "MEDIUM", "HIGH", "CRITICAL", name="severity"),
            nullable=False,
        ),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("details", postgresql.JSONB(), nullable=True),
        sa.Column("module_name", sa.String(255), nullable=True),
        sa.Column("tier_level", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["evaluation_id"],
            ["evaluation_results.evaluation_id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_evaluation_issues_issue_type", "evaluation_issues", ["issue_type"]
    )
    op.create_index("ix_evaluation_issues_severity", "evaluation_issues", ["severity"])
    op.create_index(
        "idx_eval_severity", "evaluation_issues", ["evaluation_id", "severity"]
    )

    # Create expert_labels table
    op.create_table(
        "expert_labels",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("trace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("expert_id", sa.String(255), nullable=False),
        sa.Column("is_safe", sa.Boolean(), nullable=False),
        sa.Column(
            "severity",
            sa.Enum("LOW", "MEDIUM", "HIGH", "CRITICAL", name="severity"),
            nullable=True,
        ),
        sa.Column("issues_identified", postgresql.JSONB(), nullable=True),
        sa.Column("feedback", sa.Text(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("labeled_at", sa.DateTime(), nullable=False),
        sa.Column("labeling_time_seconds", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(
            ["trace_id"],
            ["traces.trace_id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_trace_expert", "expert_labels", ["trace_id", "expert_id"])

    # Create alerts table
    op.create_table(
        "alerts",
        sa.Column("alert_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("trace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("evaluation_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("rule_name", sa.String(255), nullable=False),
        sa.Column("severity", sa.String(50), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("details", postgresql.JSONB(), nullable=True),
        sa.Column("channels", postgresql.JSONB(), nullable=False),
        sa.Column("sent_at", sa.DateTime(), nullable=False),
        sa.Column("acknowledged", sa.Boolean(), nullable=True),
        sa.Column("acknowledged_at", sa.DateTime(), nullable=True),
        sa.Column("acknowledged_by", sa.String(255), nullable=True),
        sa.ForeignKeyConstraint(
            ["evaluation_id"],
            ["evaluation_results.evaluation_id"],
        ),
        sa.ForeignKeyConstraint(
            ["organization_id"],
            ["organizations.organization_id"],
        ),
        sa.ForeignKeyConstraint(
            ["trace_id"],
            ["traces.trace_id"],
        ),
        sa.PrimaryKeyConstraint("alert_id"),
    )
    op.create_index("idx_org_sent", "alerts", ["organization_id", "sent_at"])
    op.create_index("idx_severity_ack", "alerts", ["severity", "acknowledged"])

    # Create model_versions table
    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("domain", sa.String(100), nullable=False),
        sa.Column("tier", sa.Integer(), nullable=False),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("version", sa.String(50), nullable=False),
        sa.Column("model_type", sa.String(100), nullable=False),
        sa.Column("model_path", sa.String(500), nullable=False),
        sa.Column("metrics", postgresql.JSONB(), nullable=True),
        sa.Column("training_samples", sa.Integer(), nullable=True),
        sa.Column("is_production", sa.Boolean(), nullable=True),
        sa.Column("deployed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_versions_domain", "model_versions", ["domain"])
    op.create_index(
        "idx_domain_tier_prod", "model_versions", ["domain", "tier", "is_production"]
    )
    op.create_index("idx_model_version", "model_versions", ["model_name", "version"])

    # Create usage_metrics table
    op.create_table(
        "usage_metrics",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("date", sa.DateTime(), nullable=False),
        sa.Column("hour", sa.Integer(), nullable=True),
        sa.Column("traces_count", sa.Integer(), nullable=True),
        sa.Column("tier1_count", sa.Integer(), nullable=True),
        sa.Column("tier2_count", sa.Integer(), nullable=True),
        sa.Column("tier3_count", sa.Integer(), nullable=True),
        sa.Column("tier4_count", sa.Integer(), nullable=True),
        sa.Column("tier1_cost", sa.Float(), nullable=True),
        sa.Column("tier2_cost", sa.Float(), nullable=True),
        sa.Column("tier3_cost", sa.Float(), nullable=True),
        sa.Column("tier4_cost", sa.Float(), nullable=True),
        sa.Column("total_cost", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["organization_id"],
            ["organizations.organization_id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_usage_metrics_date", "usage_metrics", ["date"])
    op.create_index("idx_org_date", "usage_metrics", ["organization_id", "date"])


def downgrade() -> None:
    """Drop all tables."""
    op.drop_table("usage_metrics")
    op.drop_table("model_versions")
    op.drop_table("alerts")
    op.drop_table("expert_labels")
    op.drop_table("evaluation_issues")
    op.drop_table("evaluation_results")
    op.drop_table("traces")
    op.drop_table("organizations")
