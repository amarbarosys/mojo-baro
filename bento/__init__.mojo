# VENDORED COPY. Upstream is ~/Projects/mojo/mojo-bento/src/bento/__init__.mojo;
# this repo keeps a real file rather than a symlink or an -I path outside the
# tree, because a clone must build without anything outside it. Sync by hand
# if the upstream changes; tools/ci-checks.sh compares the two when the
# upstream is present and says so when they drift.
#
"""BentoML's serving core: import the submodules (bento.server, bento.endpoint, bento.http, ...)."""
