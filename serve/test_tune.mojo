from std.testing import assert_equal, assert_raises
from std.os import makedirs
from std.os.path import exists
from tune import load_tune, pick


def write(dir: String, body: String) raises:
    if not exists(dir):
        makedirs(dir)
    with open(dir + "/tune.txt", "w") as f:
        f.write(body)


def main() raises:
    var d = ".work/test-tune"
    write(d, "# N K m_max v\n1024 4096 64 0\n1024 4096 256 1\n\n1024 4096 1073741824 4\n4096 4096 128 3\n")
    var t = load_tune(d)
    assert_equal(len(t), 16)
    assert_equal(pick(t, 1024, 4096, 1), 0)
    assert_equal(pick(t, 1024, 4096, 64), 0)
    assert_equal(pick(t, 1024, 4096, 65), 1)
    assert_equal(pick(t, 1024, 4096, 5000), 4)
    assert_equal(pick(t, 4096, 4096, 128), 3)
    assert_equal(pick(t, 4096, 4096, 129), -1)
    assert_equal(pick(t, 8192, 4096, 16), -1)
    assert_equal(len(load_tune(".work/test-tune-absent")), 0)
    write(d, "1024  4096 64 0\n")
    with assert_raises():
        _ = load_tune(d)
    write(d, "1024 4096 64\n")
    with assert_raises():
        _ = load_tune(d)
    write(d, "1024 4096 64 6\n")
    with assert_raises():
        _ = load_tune(d)
    write(d, "1024 4096 64 x\n")
    with assert_raises():
        _ = load_tune(d)
    print("PASS tune: buckets, fallthrough -1, no file, double space / 3 fields / range / non-int raise")
