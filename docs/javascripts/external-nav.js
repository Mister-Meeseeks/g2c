// Mark nav links that point off-site so they open in a new tab and get a
// small external-link glyph (CSS in stylesheets/extra.css).
//
// Uses Material's `document$` observable (provided when `navigation.instant`
// is enabled) so the marking is re-applied after every in-place navigation.
function markExternalNavLinks() {
  document.querySelectorAll(".md-nav__link").forEach(function (link) {
    var href = link.getAttribute("href") || "";
    if (/^https?:\/\//.test(href)) {
      link.setAttribute("target", "_blank");
      link.setAttribute("rel", "noopener noreferrer");
      link.classList.add("md-nav__link--external");
    }
  });
}

if (typeof document$ !== "undefined" && document$.subscribe) {
  document$.subscribe(markExternalNavLinks);
} else {
  document.addEventListener("DOMContentLoaded", markExternalNavLinks);
}
