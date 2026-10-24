"""Sharing metadata contracts: hero URLs, intro text, and fallback behavior."""

from types import SimpleNamespace

from scripts.docs_social import on_page_content


def preview(html, *, meta=None, homepage=False):
    page = SimpleNamespace(
        canonical_url="https://example.com/g2c/briefs/model/",
        title='A "model" & its evidence',
        meta={} if meta is None else meta.copy(),
        file=SimpleNamespace(name="index" if homepage else "model"),
        is_homepage=homepage,
    )
    config = {
        "site_url": "https://example.com/g2c/",
        "site_name": "G2C",
        "site_description": "Build a language model from scratch.",
        "extra": {
            "social_image": "CourseCover.png",
            "social_image_alt": "The course map.",
        },
    }
    assert on_page_content(html, page, config) == html
    return page.meta


def test_hero_url_keeps_project_prefix_and_resolves_rendered_relative_path():
    meta = preview('<p><img src="../model/Hero.webp" alt="An example model"></p>')
    card = meta["social_preview"]
    assert card["image"] == "https://example.com/g2c/briefs/model/Hero.webp"
    assert card["image_type"] == "image/webp"
    assert card["image_alt"] == "An example model"


def test_diagram_after_first_section_does_not_become_a_hero():
    meta = preview('<p>Introduction.</p><h2>Details</h2><img src="diagram.png">')
    card = meta["social_preview"]
    assert card["image"] == "https://example.com/g2c/CourseCover.png"
    assert card["image_alt"] == "The course map."


def test_inline_svg_and_data_urls_do_not_replace_the_supported_fallback():
    meta = preview('<img src="diagram.svg"><img src="data:image/png;base64,AAAA">')
    assert meta["social_preview"]["image"] == "https://example.com/g2c/CourseCover.png"


def test_question_is_preferred_over_intro_and_formatting_is_removed():
    meta = preview(
        '<blockquote><p><strong>Question this Brief answers:</strong> '
        '<em>Can <code>x</code> &amp; y agree?</em></p></blockquote>'
        '<p>A much longer introduction.</p>'
    )
    assert meta["description"] == "Can x & y agree?"
    assert meta["social_preview"]["description"] == meta["description"]


def test_explicit_metadata_wins_and_title_remains_plain_text():
    meta = preview(
        '<h1>Rendered title<a class="headerlink">¶</a></h1><p>Intro.</p>',
        meta={"title": 'Quotes " & angle < signs', "description": "An explicit summary."},
    )
    card = meta["social_preview"]
    assert card["title"] == 'Quotes " & angle < signs'
    assert card["description"] == "An explicit summary."


def test_plain_intro_is_shortened_at_a_word_boundary():
    text = "A useful introduction " * 20
    meta = preview(f"<p>{text}</p>")
    description = meta["description"]
    assert len(description) <= 200
    assert description.endswith("…")
    assert text.startswith(description[:-1])


def test_homepage_uses_site_summary_and_website_type():
    meta = preview("<p>A longer introduction.</p>", homepage=True)
    assert meta["social_preview"]["title"] == "G2C"
    assert meta["description"] == "Build a language model from scratch."
    assert meta["social_preview"]["type"] == "website"


def test_prose_is_used_when_an_unrelated_blockquote_comes_first():
    meta = preview("<blockquote><p>An epigraph.</p></blockquote><p>The real intro.</p>")
    assert meta["description"] == "The real intro."
    assert meta["social_preview"]["type"] == "article"


def test_full_heading_title_excludes_the_theme_permalink():
    meta = preview(
        '<h1>Module 07 — <em>Attention</em>'
        '<a href="#module-07" class="headerlink">¶</a></h1><p>Intro.</p>'
    )
    assert meta["social_preview"]["title"] == "Module 07 — Attention"
