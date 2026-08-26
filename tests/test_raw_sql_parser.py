"""
Regression tests for src/data/raw_sql_parser.py -- in particular the
missing-semicolon bug found in the real dataset (Doctor INSERT block has
no trailing ';' before 'Insert Into Nurse'), which a naive semicolon
splitter silently merges into one statement.
"""

import sys
from datetime import date, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.raw_sql_parser import (
    cast_value,
    parse_create_table,
    parse_insert,
    split_statements,
    split_top_level_tuples,
    split_tuple_fields,
    strip_block_comments,
    strip_line_comments,
)


def test_split_statements_handles_missing_semicolon_between_statements():
    sql = (
        "Insert Into Doctor\nValues\n(1, 'a')\n\n"
        "Insert Into Nurse\nValues\n(2, 'b');"
    )
    statements = split_statements(sql)
    assert len(statements) == 2
    assert statements[0].startswith("Insert Into Doctor")
    assert statements[1].startswith("Insert Into Nurse")


def test_split_statements_handles_present_semicolons_too():
    sql = "Create Table X (a Int);\nInsert Into X Values (1);"
    statements = split_statements(sql)
    assert len(statements) == 2


def test_strip_block_comments_removes_disabled_rows():
    sql = "(1, 'a'),\n/*(2, 'b'),\n(3, 'c'),*/\n(4, 'd')"
    cleaned = strip_block_comments(sql)
    assert "'b'" not in cleaned
    assert "'d'" in cleaned


def test_strip_line_comments_removes_dashdash_comments():
    sql = "SELECT 1 -- this is a comment\nSELECT 2"
    cleaned = strip_line_comments(sql)
    assert "comment" not in cleaned
    assert "SELECT 2" in cleaned


def test_parse_create_table_extracts_columns_pk_and_fk():
    stmt = (
        "Create Table Doctor (\n"
        "  doct_Id Int Primary Key,\n"
        "  dept_Id Int,\n"
        "  Foreign Key (dept_Id) References Department(dept_Id)\n"
        ")"
    )
    table = parse_create_table(stmt, order=0)
    assert table.name == "Doctor"
    assert [c.name for c in table.columns] == ["doct_Id", "dept_Id"]
    assert table.columns[0].is_primary_key
    assert not table.columns[1].is_primary_key
    assert table.foreign_keys[0].column == "dept_Id"
    assert table.foreign_keys[0].ref_table == "Department"


def test_split_top_level_tuples_respects_quotes_and_nested_commas():
    blob = "(1, 'a, b'), (2, 'c')"
    tuples = split_top_level_tuples(blob)
    assert tuples == ["(1, 'a, b')", "(2, 'c')"]


def test_split_tuple_fields_respects_quoted_commas():
    fields = split_tuple_fields("(1, 'a, b', NULL)")
    assert fields == ["1", "'a, b'", "NULL"]


def test_parse_insert_extracts_rows():
    stmt = "Insert Into Department\nValues\n(101, 'Cardiology'),\n(102, 'Neurology')"
    name, rows = parse_insert(stmt)
    assert name == "Department"
    assert rows == [["101", "'Cardiology'"], ["102", "'Neurology'"]]


def test_cast_value_handles_all_relevant_types():
    assert cast_value("NULL", "Int") is None
    assert cast_value("42", "Int") == 42
    assert cast_value("'hello'", "Varchar(100)") == "hello"
    assert cast_value("'2024-01-15'", "Date") == date(2024, 1, 15)
    assert cast_value("'8:00:00'", "Time") == time(8, 0, 0)
    assert cast_value("3.14", "Decimal(10,2)") == 3.14
