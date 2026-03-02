"""Tests for vector operation helpers — content expression, join helpers, etc."""
from pgsemantic.db.vectors import (
    _content_expr,
    _content_text,
    _not_null_check,
    _pk_cast_join,
)


class TestContentExpr:
    def test_single_column(self) -> None:
        result = _content_expr(["description"])
        assert result == '"description"'

    def test_single_column_with_alias(self) -> None:
        result = _content_expr(["description"], alias="s")
        assert result == 's."description"'

    def test_multi_column(self) -> None:
        result = _content_expr(["title", "description"], alias="s")
        assert "'title: '" in result
        assert "'description: '" in result
        assert 's."title"' in result
        assert 's."description"' in result
        assert "E'\\n'" in result


class TestContentText:
    def test_single_column(self) -> None:
        row = {"title": "Widget", "description": "A great widget"}
        result = _content_text(["description"], row)
        assert result == "A great widget"

    def test_multi_column(self) -> None:
        row = {"title": "Widget", "description": "A great widget"}
        result = _content_text(["title", "description"], row)
        assert result == "title: Widget\ndescription: A great widget"


class TestPkCastJoin:
    def test_single_pk(self) -> None:
        result = _pk_cast_join(["id"])
        assert result == 's."id"::TEXT = e.row_id'

    def test_composite_pk(self) -> None:
        result = _pk_cast_join(["org_id", "user_id"])
        assert "s.\"org_id\"::TEXT" in result
        assert "s.\"user_id\"::TEXT" in result
        assert "e.row_id" in result


class TestNotNullCheck:
    def test_single_column(self) -> None:
        result = _not_null_check(["description"], alias="s")
        assert result == 's."description" IS NOT NULL'

    def test_multi_column(self) -> None:
        result = _not_null_check(["title", "description"], alias="s")
        assert 's."title" IS NOT NULL' in result
        assert 's."description" IS NOT NULL' in result
        assert " AND " in result

    def test_no_alias(self) -> None:
        result = _not_null_check(["description"], alias="")
        assert result == '"description" IS NOT NULL'
