"""MkDocs link previews from existing heroes and introductory text.

The first raster image before any H2 is the hero. Pages without one use
``extra.social_image`` (relative to site_url). Descriptions prefer explicit
front matter, then the lesson/Brief question, then the first prose paragraph.
Only metadata changes; the rendered article and its original images stay intact.
"""

import mimetypes
import re
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

QUESTION = re.compile(
    r"^(?:Question this (?:module|Brief) answers|Release question)\s*:\s*",
    re.IGNORECASE,
)
RASTER_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".gif")


def _shorten(text, limit):
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rsplit(" ", 1)[0] + "…"


class _Intro(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.before_sections = True
        self.hero = None
        self.question = ""
        self.paragraph = ""
        self.title = ""
        self._quote_depth = 0
        self._pre_depth = 0
        self._paragraph_parts = None
        self._heading_parts = None
        self._in_permalink = False

    def handle_starttag(self, tag, attrs):
        if tag == "h2":
            self.before_sections = False
        elif tag == "h1":
            self._heading_parts = []
        elif tag == "a" and "headerlink" in dict(attrs).get("class", "").split():
            self._in_permalink = True
        elif tag == "blockquote":
            self._quote_depth += 1
        elif tag == "pre":
            self._pre_depth += 1
        elif tag == "p" and self.before_sections and not self._pre_depth:
            self._paragraph_parts = []
        elif tag == "img" and self.before_sections and self.hero is None:
            image = dict(attrs)
            src = image.get("src", "")
            parsed = urlsplit(src)
            if parsed.scheme in ("", "http", "https") and parsed.path.lower().endswith(
                RASTER_EXTENSIONS
            ):
                self.hero = (src, image.get("alt", ""))

    def handle_endtag(self, tag):
        if tag == "h1" and self._heading_parts is not None:
            self.title = " ".join("".join(self._heading_parts).split())
            self._heading_parts = None
        elif tag == "a":
            self._in_permalink = False
        elif tag == "p" and self._paragraph_parts is not None:
            text = " ".join("".join(self._paragraph_parts).split())
            if self._quote_depth and QUESTION.match(text):
                self.question = QUESTION.sub("", text)
            elif not self._quote_depth and text and not self.paragraph:
                self.paragraph = text
            self._paragraph_parts = None
        elif tag == "blockquote":
            self._quote_depth -= 1
        elif tag == "pre":
            self._pre_depth -= 1

    def handle_data(self, data):
        if self._paragraph_parts is not None:
            self._paragraph_parts.append(data)
        if self._heading_parts is not None and not self._in_permalink:
            self._heading_parts.append(data)


def on_page_content(html, page, config, **kwargs):
    """Attach sharing metadata after MkDocs has resolved rendered image paths."""
    intro = _Intro()
    intro.feed(html)
    extra = config.get("extra", {})
    if intro.hero:
        image, image_alt = intro.hero
        image_url = urljoin(page.canonical_url, image)
    else:
        image_url = urljoin(config["site_url"], extra["social_image"])
        image_alt = extra.get("social_image_alt", config["site_name"])
    description = _shorten(
        page.meta.get("description")
        or (config.get("site_description") if page.is_homepage else None)
        or intro.question
        or intro.paragraph
        or config.get("site_description", ""),
        200,
    )
    # Also replace the theme's generic HTML description with the page-specific one.
    if not page.meta.get("description"):
        page.meta["description"] = description
    page.meta["social_preview"] = {
        "title": page.meta.get("title")
        or (config["site_name"] if page.is_homepage else intro.title or page.title)
        or config["site_name"],
        "description": description,
        "type": "website" if page.file.name == "index" else "article",
        "image": image_url,
        "image_type": mimetypes.guess_type(urlsplit(image_url).path)[0] or "image/png",
        "image_alt": _shorten(image_alt or page.title or config["site_name"], 420),
    }
    return html
