# shadow.mojo: S0, the LatentOS boot shadow.
#
# Stock immucore still does the real boot. This binary runs after it,
# read-only, and logs what it WOULD have mounted: GPT scan result, chosen
# active/passive/recovery image, COS_STATE/COS_PERSISTENT/LOS_MODELS
# partitions, hugepage reservation vs latentos.tier.reserve.
# 05-boot-path-mojo.md sec.4, S0 row. No mount/ioctl/loop syscall anywhere in
# this file or in latentos.boot.gpt / latentos.boot.cmdline: everything is a
# read() on /proc or a whole-disk block device opened O_RDONLY.
#
# Build: mojo build src/latentos/boot/shadow.mojo -I src -o <out>
# Run (inside the VM, as root): ./shadow

import latentos.boot.gpt as gpt
import latentos.boot.cmdline as cmdline
import latentos.boot.result as result
import latentos.sys as sys_

def log_line(line: String):
    """Every S0 fact goes to both stdout (host dev-testing, SSH-driven
    harness capture) and /dev/ttyS0 (the QEMU serial log, best-effort: a
    failure here must never take down the shadow, and must never fall back
    to any write path other than the serial console itself)."""
    print(line)
    try:
        with open("/dev/ttyS0", "w") as f:
            _ = f.write(line + "\n")
    except _:
        pass


def image_kind(cos_img: String) -> String:
    if cos_img == "":
        return "unknown (cos-img/filename absent from cmdline)"
    if cos_img.endswith("active.img"):
        return "active"
    if cos_img.endswith("passive.img"):
        return "passive"
    if cos_img.endswith("recovery.img"):
        return "recovery"
    return "unknown:" + cos_img


def finish(reservation: String, status: String) raises:
    """The one RESULT line (no mounts key: S0 mounts nothing), then a non-zero exit unless the pass completed."""
    var line = result.result_line("S0", "", reservation, status)
    log_line(line)
    if result.must_fail("", status):
        raise Error(line)


def main() raises:
    log_line("S0: LatentOS boot shadow starting (read-only, no mounts, no block-device writes)")
    var reservation = String("SKIPPED")

    var cl = cmdline.read_cmdline()
    log_line("S0: /proc/cmdline = " + cl)

    var cos_img = cmdline.cmdline_get(cl, "cos-img/filename")
    log_line("S0: chosen image (cos-img/filename) = " + cos_img + " [" + image_kind(cos_img) + "]")

    var disk: String
    var parts: List[gpt.GptPartition]
    try:
        var boot = gpt.find_boot_disk()
        disk = boot[0]
        parts = boot[1].copy()
    except e:
        log_line("S0: BLOCKER finding boot disk: " + String(e))
        log_line("S0: shadow pass incomplete")
        finish(reservation, "incomplete")
        return
    log_line("S0: boot disk = " + disk)

    log_line("S0: GPT scan result, " + String(len(parts)) + " partitions:")
    for p in parts:
        log_line(
            "S0:   gpt_name=" + p.gpt_name
            + " fs_label=" + (p.fs_label if p.fs_label != "" else "(none)")
            + " fs_kind=" + p.fs_kind
            + " lba=[" + String(p.first_lba) + "," + String(p.last_lba) + "]"
        )

    var target_labels: List[String] = ["COS_STATE", "COS_PERSISTENT", "LOS_MODELS"]
    for target in target_labels:
        var found = gpt.find_by_label(parts, target)
        if found.__bool__():
            var p = found.value().copy()
            log_line(
                "S0: " + target + " = found, lba=[" + String(p.first_lba) + "," + String(p.last_lba)
                + "], fs=" + p.fs_kind
            )
        else:
            log_line("S0: " + target + " = not found")

    var hp = sys_.sys_read_meminfo_hugepages()
    var total_hp = hp[0]
    var size_kb = hp[1]
    log_line("S0: /proc/meminfo HugePages_Total=" + String(total_hp) + " Hugepagesize_kB=" + String(size_kb))

    var cmd_hugepages = cmdline.cmdline_get(cl, "hugepages")
    var cmd_hugepagesz = cmdline.cmdline_get(cl, "hugepagesz")
    var tier_reserve = cmdline.cmdline_get(cl, "latentos.tier.reserve_gib")

    if tier_reserve != "":
        log_line("S0: latentos.tier.reserve_gib=" + tier_reserve + " present on cmdline")
    else:
        log_line(
            "S0: latentos.tier.reserve_gib not set on this disk (05 sec.1/4's tier key is unwired here; "
            + "05-boot-S0-plan-2026-09-11.md item 2); comparing against the generic reservation instead"
        )

    var status = String("complete")
    if cmd_hugepages != "":
        var want = result.parse_count(cmd_hugepages)
        if want < 0:
            log_line("S0: reservation check ERROR (cmdline hugepages=" + cmd_hugepages + " is not a number)")
            reservation = "FAIL"
            status = "error"
        elif want == total_hp:
            reservation = "PASS"
            log_line(
                "S0: reservation check PASS (cmdline hugepages=" + cmd_hugepages + " hugepagesz=" + cmd_hugepagesz
                + " == /proc/meminfo HugePages_Total=" + String(total_hp) + ")"
            )
        else:
            reservation = "FAIL"
            log_line(
                "S0: reservation check FAIL (cmdline hugepages=" + cmd_hugepages
                + " != /proc/meminfo HugePages_Total=" + String(total_hp) + ")"
            )
    else:
        log_line("S0: reservation check SKIPPED, no hugepages= cmdline key present")

    if status == "complete":
        log_line("S0: shadow pass complete")
    finish(reservation, status)
