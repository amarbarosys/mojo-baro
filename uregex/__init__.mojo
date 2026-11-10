# VENDORED COPY. Upstream is ~/Projects/mojo/mojo-uregex/src/uregex/__init__.mojo;
# this repo keeps a real file rather than a symlink or an -I path outside the
# tree, because a clone must build without anything outside it. Sync by hand
# if the upstream changes; tools/ci-checks.sh compares the two when the
# upstream is present and says so when they drift.
#
# Copyright 2026 amarbaro.org. Licensed under the Apache License, Version 2.0.
from .pattern import Pattern, FullPattern, PatternT, Match, compile, findall, search, split_keep
