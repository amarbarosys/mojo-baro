# VENDORED COPY. Upstream is ~/Projects/mojo/mojo-posix/src/posix/__init__.mojo;
# this repo keeps a real file rather than a symlink or an -I path outside the
# tree, because a clone must build without anything outside it. Sync by hand
# if the upstream changes; tools/ci-checks.sh compares the two when the
# upstream is present and says so when they drift.
#
from posix.proc import (
    getpid, getppid, getpgid, kill, killpg, reap, spawn, spawn_pair, try_spawn, exec, environ, block_signals, signal_fd, read_signals,
    SIGHUP, SIGINT, SIGKILL, SIGUSR1, SIGUSR2, SIGTERM, SIGCHLD, SIGCONT, SIGSTOP,
)
from posix.poll import PollSet, POLLIN, POLLOUT, POLLERR, POLLHUP, POLLNVAL
from posix.sock import unix_listen, unix_connect, accept, recv, send_all, shutdown_write, close
from posix._sys import cstr, errno
