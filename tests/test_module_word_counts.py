from __future__ import annotations

from scripts.module_word_counts import deep_breakdown, render_breakdown_table, render_table


def test_deep_breakdown_counts_canonical_sections_without_overlap():
    text = """# Module X
intro words

## Before you start
before words

## Where this fits in
where words

## The big idea
big idea words

## A module-specific lecture section
lecture body words

## Concepts to internalize
concept words

## What you'll build
build words

## Exercises
exercise words

## Pitfalls to expect
pitfall words

## M-series notes
mseries words

## Reading
reading words

## Deliverable checklist
deliverable words
"""

    counts = deep_breakdown(text)

    assert counts["intro"] == 5
    assert counts["before_start"] == 6
    assert counts["where_fits"] == 7
    assert counts["big_idea"] == 7
    assert counts["body"] == 8
    assert counts["concepts"] == 6
    assert counts["build"] == 6
    assert counts["exercises"] == 4
    assert counts["pitfalls"] == 6
    assert counts["m_series"] == 5
    assert counts["reading"] == 4
    assert counts["deliverables"] == 5


def test_deep_breakdown_handles_missing_and_out_of_order_sections():
    text = """# Module X
intro

## Before you start
before

## The big idea
idea

## Reading
read

## Deliverable checklist
done

## M-series notes
mseries
"""

    counts = deep_breakdown(text)

    assert counts["where_fits"] == 0
    assert counts["before_start"] == 5
    assert counts["big_idea"] == 5
    assert counts["body"] == 0
    assert counts["reading"] == 3
    assert counts["deliverables"] == 4
    assert counts["m_series"] == 4


def test_render_breakdown_csv_uses_expected_columns():
    rendered = render_breakdown_table(
        [
            {
                "module": "00-test",
                "intro": 1,
                "before_start": 2,
                "where_fits": 3,
                "big_idea": 4,
                "body": 12,
                "concepts": 5,
                "build": 6,
                "exercises": 7,
                "pitfalls": 8,
                "m_series": 9,
                "reading": 10,
                "deliverables": 11,
                "total": 66,
            }
        ],
        "csv",
    )

    assert rendered.splitlines()[0] == (
        "module,intro,before_start,where_fits,big_idea,body,concepts,build,"
        "exercises,pitfalls,m_series,reading,deliverables,total"
    )
    assert rendered.splitlines()[1] == "00-test,1,2,3,4,12,5,6,7,8,9,10,11,66"


def test_render_table_relative_diff_csv():
    rows = [
        {
            "module": "a",
            "lecture": 10,
            "post": 0,
            "total": 10,
            "heading": None,
        },
        {
            "module": "b",
            "lecture": 20,
            "post": 10,
            "total": 30,
            "heading": "## What you'll build",
        },
    ]

    rendered = render_table(rows, "csv", relative="diff")

    assert rendered.splitlines()[0] == "module,lecture,post,total,split_heading"
    assert rendered.splitlines()[1] == "a,-5,-5,-10,"
    assert rendered.splitlines()[2] == "b,+5,+5,+10,## What you'll build"


def test_render_breakdown_relative_pct_csv():
    base = {
        "before_start": 10,
        "where_fits": 10,
        "big_idea": 10,
        "body": 10,
        "concepts": 10,
        "build": 10,
        "exercises": 10,
        "pitfalls": 10,
        "m_series": 10,
        "reading": 10,
        "deliverables": 10,
    }
    rows = [
        {"module": "a", "intro": 10, **base, "total": 10},
        {"module": "b", "intro": 30, **{key: 30 for key in base}, "total": 30},
    ]

    rendered = render_breakdown_table(rows, "csv", relative="pct")

    assert rendered.splitlines()[1] == (
        "a,-50.0%,-50.0%,-50.0%,-50.0%,-50.0%,-50.0%,"
        "-50.0%,-50.0%,-50.0%,-50.0%,-50.0%,-50.0%,-50.0%"
    )
    assert rendered.splitlines()[2] == (
        "b,+50.0%,+50.0%,+50.0%,+50.0%,+50.0%,+50.0%,"
        "+50.0%,+50.0%,+50.0%,+50.0%,+50.0%,+50.0%,+50.0%"
    )
