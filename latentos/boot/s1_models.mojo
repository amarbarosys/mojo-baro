# s1_models.mojo: S1, the LatentOS models-side boot actor.
#
# 05-boot-path-mojo.md sec.4, S1 ("act") row: this binary performs 05's
# steps for the MODELS side (LOS_MODELS partition, hugetlbfs, reservation
# check) with real mount(2) syscalls, while stock immucore still assembles
# the root the normal way -- unlike S0's shadow.mojo, which only logged what
# it WOULD have done. 09-roadmap.md A5.
#
# Mountpoints are our own, not the ones systemd/immucore already manage
# (/dev/hugepages is already mounted by the time a shell is reached, verified
# in the S0 receipt's findmnt ground truth): mounting LOS_MODELS/hugetlbfs at
# S1-owned paths makes the check that follows about THIS binary's action,
# not something systemd already did, and avoids an EBUSY collision.
#
# Build: mojo build src/latentos/boot/s1_models.mojo -I src -o <out> \
#          -Xlinker <mount_shim.o> (tools/latent-os/mount_shim.c; sys_mount's
#          docstring explains why a shim is needed at all)
# Run (inside the VM, as root): ./s1_models

import latentos.boot.gpt as gpt
import latentos.boot.cmdline as cmdline
import latentos.boot.result as result
import latentos.sys as sys_

comptime LOS_MODELS_MOUNT = "/run/los-models"
comptime LOS_HUGEPAGES_MOUNT = "/run/los-hugepages"


def log_line(line: String):
    """Same belt-and-suspenders serial+stdout logging as S0's shadow.mojo:
    every S1 fact goes to both stdout and /dev/ttyS0, and a failure writing
    to the serial console must never take the binary down."""
    print(line)
    try:
        with open("/dev/ttyS0", "w") as f:
            _ = f.write(line + "\n")
    except _:
        pass


def finish(mounts: String, reservation: String, status: String) raises:
    """The one RESULT line, then a non-zero exit unless the pass completed with no failed mount."""
    var line = result.result_line("S1", mounts, reservation, status)
    log_line(line)
    if result.must_fail(mounts, status):
        raise Error(line)


def main() raises:
    log_line("S1: LatentOS boot actor starting (models side: LOS_MODELS + hugetlbfs, real mounts)")
    var mounts = String("SKIPPED")
    var reservation = String("SKIPPED")

    var cl = cmdline.read_cmdline()
    log_line("S1: /proc/cmdline = " + cl)

    var disk: String
    var parts: List[gpt.GptPartition]
    try:
        var boot = gpt.find_boot_disk()
        disk = boot[0]
        parts = boot[1].copy()
    except e:
        log_line("S1: BLOCKER finding boot disk: " + String(e))
        log_line("S1: models-side pass incomplete")
        finish(mounts, reservation, "incomplete")
        return
    log_line("S1: boot disk = " + disk)

    var found = gpt.find_by_label(parts, "LOS_MODELS")
    if not found.__bool__():
        log_line("S1: LOS_MODELS = not found on " + disk + "; nothing to mount")
        log_line("S1: models-side pass incomplete")
        finish(mounts, reservation, "incomplete")
        return

    var los = found.value().copy()
    var part_dev = disk + String(los.entry_index)
    log_line(
        "S1: LOS_MODELS = found, entry_index=" + String(los.entry_index)
        + ", lba=[" + String(los.first_lba) + "," + String(los.last_lba)
        + "], fs=" + los.fs_kind + ", device=" + part_dev
    )

    var mkdir_rc = sys_.sys_mkdir(LOS_MODELS_MOUNT)
    if mkdir_rc != 0 and sys_.errno_now() != 17:  # 17 = EEXIST, tolerated
        log_line("S1: mkdir " + LOS_MODELS_MOUNT + " FAILED, errno=" + String(sys_.errno_now()))
        log_line("S1: models-side pass incomplete")
        finish("FAIL", reservation, "incomplete")
        return

    var models_ok = False
    var hugepages_ok = False
    var mount_rc = sys_.sys_mount(part_dev, LOS_MODELS_MOUNT, los.fs_kind)
    if mount_rc == 0:
        models_ok = True
        log_line("S1: LOS_MODELS mount PASS (" + part_dev + " -> " + LOS_MODELS_MOUNT + ", fs=" + los.fs_kind + ")")
    else:
        log_line("S1: LOS_MODELS mount FAIL, errno=" + String(sys_.errno_now()))

    var hp_mkdir_rc = sys_.sys_mkdir(LOS_HUGEPAGES_MOUNT)
    if hp_mkdir_rc != 0 and sys_.errno_now() != 17:
        log_line("S1: mkdir " + LOS_HUGEPAGES_MOUNT + " FAILED, errno=" + String(sys_.errno_now()))
        log_line("S1: models-side pass incomplete")
        finish("FAIL", reservation, "incomplete")
        return

    var hp_mount_rc = sys_.sys_mount("hugetlbfs", LOS_HUGEPAGES_MOUNT, "hugetlbfs")
    if hp_mount_rc == 0:
        hugepages_ok = True
        log_line("S1: hugetlbfs mount PASS (" + LOS_HUGEPAGES_MOUNT + ")")
    else:
        log_line("S1: hugetlbfs mount FAIL, errno=" + String(sys_.errno_now()))

    mounts = "PASS" if (models_ok and hugepages_ok) else "FAIL"
    var hp = sys_.sys_read_meminfo_hugepages()
    var total_hp = hp[0]
    var size_kb = hp[1]
    log_line("S1: /proc/meminfo HugePages_Total=" + String(total_hp) + " Hugepagesize_kB=" + String(size_kb))

    var cmd_hugepages = cmdline.cmdline_get(cl, "hugepages")
    var cmd_hugepagesz = cmdline.cmdline_get(cl, "hugepagesz")
    var tier_reserve = cmdline.cmdline_get(cl, "latentos.tier.reserve_gib")

    if tier_reserve != "":
        log_line("S1: latentos.tier.reserve_gib=" + tier_reserve + " present on cmdline")
    else:
        log_line(
            "S1: latentos.tier.reserve_gib not set on this disk (same gap S0 flagged; "
            + "runs/latent-os/boot-S1-plan-2026-09-11.md; comparing against the generic "
            + "reservation instead)"
        )

    var status = String("complete")
    if cmd_hugepages != "":
        var want = result.parse_count(cmd_hugepages)
        if want < 0:
            log_line("S1: reservation check ERROR (cmdline hugepages=" + cmd_hugepages + " is not a number)")
            reservation = "FAIL"
            status = "error"
        elif want == total_hp:
            reservation = "PASS"
            log_line(
                "S1: reservation check PASS (cmdline hugepages=" + cmd_hugepages + " hugepagesz=" + cmd_hugepagesz
                + " == /proc/meminfo HugePages_Total=" + String(total_hp) + ")"
            )
        else:
            reservation = "FAIL"
            log_line(
                "S1: reservation check FAIL (cmdline hugepages=" + cmd_hugepages
                + " != /proc/meminfo HugePages_Total=" + String(total_hp) + ")"
            )
    else:
        log_line("S1: reservation check SKIPPED, no hugepages= cmdline key present")

    if status == "complete":
        log_line("S1: models-side pass complete")
    finish(mounts, reservation, status)
