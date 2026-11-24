# VENDORED COPY. Upstream is ~/Projects/mojo/mojo-libs/libs/posix/src/posix/systemd/__init__.mojo;
# this repo keeps a real file rather than a symlink or an -I path outside the
# tree, because a clone must build without anything outside it. Sync by hand
# if the upstream changes; tools/ci-checks.sh compares the two when the
# upstream is present and says so when they drift.
#
from .notify import notify, ready, status, watchdog, stopping, reloading, watchdog_usec
