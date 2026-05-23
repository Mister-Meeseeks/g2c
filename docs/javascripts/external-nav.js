// Mark nav links that point off-site so they open in a new tab and get a
// small external-link glyph (CSS in stylesheets/extra.css).
//
// Uses Material's `document$` observable (provided when `navigation.instant`
// is enabled) so the marking is re-applied after every in-place navigation.
// `navigation.instant` rewrites every internal href to an absolute URL on the
// site's own host, so a `^https?://` check would match everything — compare
// the resolved URL's host against the current page's host instead.
function markExternalNavLinks() {
  document.querySelectorAll(".md-nav__link").forEach(function (link) {
    var href = link.getAttribute("href");
    if (!href) return;
    var url;
    try {
      url = new URL(href, location.href);
    } catch (e) {
      return;
    }
    if (url.protocol !== "http:" && url.protocol !== "https:") return;
    if (url.host === location.host) return;
    link.setAttribute("target", "_blank");
    link.setAttribute("rel", "noopener noreferrer");
    link.classList.add("md-nav__link--external");
  });
}

if (typeof document$ !== "undefined" && document$.subscribe) {
  document$.subscribe(markExternalNavLinks);
} else {
  document.addEventListener("DOMContentLoaded", markExternalNavLinks);
}
