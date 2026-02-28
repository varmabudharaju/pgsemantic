"""Tests for database introspection -- column scoring and schema parsing."""

from pgsemantic.db.introspect import (
    LOW_VALUE_COLUMN_NAMES,
    SEMANTIC_COLUMN_NAMES,
    ColumnCandidate,
    score_column,
    score_to_stars,
)


class TestScoreColumn:
    def test_long_text_high_score(self) -> None:
        candidate = ColumnCandidate(
            table_name="products",
            table_schema="public",
            column_name="description",
            data_type="text",
            avg_length=847.0,
            sampled_rows=50,
        )
        score = score_column(candidate)
        assert score >= 2.5

    def test_medium_text_medium_score(self) -> None:
        candidate = ColumnCandidate(
            table_name="users",
            table_schema="public",
            column_name="bio",
            data_type="text",
            avg_length=143.0,
            sampled_rows=50,
        )
        score = score_column(candidate)
        assert 1.5 <= score <= 3.0

    def test_short_text_low_score(self) -> None:
        candidate = ColumnCandidate(
            table_name="products",
            table_schema="public",
            column_name="name",
            data_type="character varying",
            avg_length=34.0,
            sampled_rows=50,
        )
        score = score_column(candidate)
        assert score <= 1.5

    def test_low_value_column_forced_low(self) -> None:
        candidate = ColumnCandidate(
            table_name="users",
            table_schema="public",
            column_name="email",
            data_type="character varying",
            avg_length=200.0,
            sampled_rows=50,
        )
        assert score_column(candidate) == 1.0

    def test_semantic_column_name_bonus(self) -> None:
        base = ColumnCandidate(
            table_name="articles",
            table_schema="public",
            column_name="data",
            data_type="text",
            avg_length=500.0,
            sampled_rows=50,
        )
        semantic = ColumnCandidate(
            table_name="articles",
            table_schema="public",
            column_name="content",
            data_type="text",
            avg_length=500.0,
            sampled_rows=50,
        )
        assert score_column(semantic) > score_column(base)

    def test_zero_avg_length(self) -> None:
        candidate = ColumnCandidate(
            table_name="test",
            table_schema="public",
            column_name="notes",
            data_type="text",
            avg_length=0.0,
            sampled_rows=0,
        )
        assert score_column(candidate) >= 0.0


class TestScoreToStars:
    def test_three_stars(self) -> None:
        assert score_to_stars(3.0) == "\u2605\u2605\u2605"
        assert score_to_stars(2.5) == "\u2605\u2605\u2605"

    def test_two_stars(self) -> None:
        assert score_to_stars(2.0) == "\u2605\u2605\u2606"
        assert score_to_stars(1.5) == "\u2605\u2605\u2606"

    def test_one_star(self) -> None:
        assert score_to_stars(1.0) == "\u2605\u2606\u2606"
        assert score_to_stars(0.5) == "\u2605\u2606\u2606"


class TestColumnNameSets:
    def test_semantic_names(self) -> None:
        assert "description" in SEMANTIC_COLUMN_NAMES
        assert "body" in SEMANTIC_COLUMN_NAMES
        assert "content" in SEMANTIC_COLUMN_NAMES

    def test_low_value_names(self) -> None:
        assert "id" in LOW_VALUE_COLUMN_NAMES
        assert "email" in LOW_VALUE_COLUMN_NAMES
        assert "uuid" in LOW_VALUE_COLUMN_NAMES
