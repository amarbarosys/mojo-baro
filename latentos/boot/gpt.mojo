# gpt.mojo: read-only GPT scan + filesystem-label read for the S0 boot shadow.
#
# S0 is read-only end to end: every function here opens with mode "r" only,
# never "w"/"rw"/"a", and never issues mount/ioctl/loop syscalls. It answers
# "what partitions exist and what are they called" the way immucore's `blkid`
# equivalent would: by filesystem VOLUME LABEL, not by the GPT partition-entry
# name field. On a real Kairos disk those differ (verified against
# labiso/vm/kairos-disk.img, runs/latent-os/boot-S0-plan-2026-09-11.md): the
# GPT entry names are generic ("state", "persistent", ...) and COS_STATE /
# COS_PERSISTENT / LOS_MODELS are ext4 volume labels.

from std.os import listdir
from std.os.path import exists

comptime GPT_SIG_OFFSET = 0
comptime GPT_PART_ENTRIES_LBA_OFFSET = 72
comptime GPT_NUM_ENTRIES_OFFSET = 80
comptime GPT_ENTRY_SIZE_OFFSET = 84
comptime GPT_ENTRY_FIRST_LBA_OFFSET = 32
comptime GPT_ENTRY_LAST_LBA_OFFSET = 40
comptime GPT_ENTRY_NAME_OFFSET = 56
comptime GPT_ENTRY_NAME_UNITS = 36  # UTF-16LE code units, 72 bytes

comptime EXT4_MAGIC_OFFSET = 0x38   # relative to superblock start
comptime EXT4_LABEL_OFFSET = 0x78   # relative to superblock start, 16 bytes
comptime EXT4_SUPERBLOCK_OFFSET = 1024  # relative to partition start

comptime FAT_VOLLAB_OFFSET = 0x47   # relative to partition start, 11 bytes
comptime FAT_BOOTSIG_OFFSET = 0x42


def bytes_to_ascii(b: List[UInt8], start: Int, maxlen: Int) -> String:
    """Decodes a NUL-terminated (or space-padded) ASCII run, stopping at
    the first byte that isn't printable ASCII."""
    var s = String("")
    for i in range(maxlen):
        var c = Int(b[start + i])
        if c == 0:
            break
        if c < 0x20 or c > 0x7E:
            break
        s += String(chr(c))
    return String(s.strip())


def le_u64(b: List[UInt8], off: Int) -> UInt64:
    var v: UInt64 = 0
    for i in range(8):
        v |= UInt64(b[off + i]) << UInt64(8 * i)
    return v


def le_u32(b: List[UInt8], off: Int) -> UInt32:
    var v: UInt32 = 0
    for i in range(4):
        v |= UInt32(b[off + i]) << UInt32(8 * i)
    return v


def le_u16(b: List[UInt8], off: Int) -> UInt16:
    var v: UInt16 = 0
    for i in range(2):
        v |= UInt16(b[off + i]) << UInt16(8 * i)
    return v


def utf16le_name(b: List[UInt8], start: Int, units: Int) -> String:
    var s = String("")
    for i in range(units):
        var lo = b[start + 2 * i]
        var hi = b[start + 2 * i + 1]
        if lo == 0 and hi == 0:
            break
        if hi == 0:
            s += String(chr(Int(lo)))
    return s


struct GptPartition(Copyable, Movable):
    var gpt_name: String       # GPT partition-entry name (generic, not the FS label)
    var fs_label: String       # filesystem volume label (COS_STATE etc.), "" if none/unreadable
    var fs_kind: String        # "ext4" | "fat32" | "unknown"
    var first_lba: UInt64
    var last_lba: UInt64
    var entry_index: Int       # 1-based GPT entry position; on a sequentially-partitioned
                                # virtio/scsi disk with no gaps this equals the kernel partition
                                # number (S1: needed to build the /dev/vdaN mount source, since
                                # nothing here parses /sys/class/block/*/start to confirm it --
                                # that confirmation is what boot_s1_verify.py's findmnt diff is for)

    def __init__(out self, gpt_name: String, fs_label: String, fs_kind: String, first_lba: UInt64, last_lba: UInt64, entry_index: Int = 0):
        self.gpt_name = gpt_name
        self.fs_label = fs_label
        self.fs_kind = fs_kind
        self.first_lba = first_lba
        self.last_lba = last_lba
        self.entry_index = entry_index


def list_whole_disk_candidates() raises -> List[String]:
    """Block devices under /sys/class/block with no `partition` sibling file:
    whole disks, not partitions of one. Mirrors 05 sec.1 step 5's
    "enumerate /sys/class/block/*"."""
    var out = List[String]()
    for e in listdir("/sys/class/block"):
        var name = String(e)
        var has_partition_attr = exists("/sys/class/block/" + name + "/partition")
        if not has_partition_attr:
            out.append(name)
    return out^


def read_ext4_label(mut f: FileHandle, part_start_byte: Int) raises -> String:
    _ = f.seek(part_start_byte + EXT4_SUPERBLOCK_OFFSET + EXT4_MAGIC_OFFSET)
    var magic_b = f.read_bytes(2)
    var magic = le_u16(magic_b, 0)
    if magic != 0xEF53:
        return ""
    _ = f.seek(part_start_byte + EXT4_SUPERBLOCK_OFFSET + EXT4_LABEL_OFFSET)
    var label_b = f.read_bytes(16)
    return bytes_to_ascii(label_b, 0, 16)


def read_fat_label(mut f: FileHandle, part_start_byte: Int) raises -> String:
    _ = f.seek(part_start_byte + FAT_BOOTSIG_OFFSET)
    var sig_b = f.read_bytes(1)
    if Int(sig_b[0]) != 0x29:
        return ""
    _ = f.seek(part_start_byte + FAT_VOLLAB_OFFSET)
    var label_b = f.read_bytes(11)
    return bytes_to_ascii(label_b, 0, 11)


def read_gpt(dev_path: String) raises -> List[GptPartition]:
    """Reads the primary GPT header + partition entries from `dev_path` and
    resolves each partition's filesystem label. Read-only: opens "r", never
    seeks past what it needs, never opens a partition device node directly
    (reads everything as byte offsets on the whole-disk node, which is how
    the values above were verified against kairos-disk.img)."""
    var out = List[GptPartition]()
    with open(dev_path, "r") as f:
        _ = f.seek(512)
        var hdr = f.read_bytes(92)
        if bytes_to_ascii(hdr, GPT_SIG_OFFSET, 8) != "EFI PART":
            return out^

        var entries_lba = le_u64(hdr, GPT_PART_ENTRIES_LBA_OFFSET)
        var num_entries = le_u32(hdr, GPT_NUM_ENTRIES_OFFSET)
        var entry_size = le_u32(hdr, GPT_ENTRY_SIZE_OFFSET)

        _ = f.seek(Int(entries_lba) * 512)
        var entries = f.read_bytes(Int(num_entries) * Int(entry_size))

        for i in range(Int(num_entries)):
            var base = i * Int(entry_size)
            var first_lba = le_u64(entries, base + GPT_ENTRY_FIRST_LBA_OFFSET)
            var last_lba = le_u64(entries, base + GPT_ENTRY_LAST_LBA_OFFSET)
            if first_lba == 0 and last_lba == 0:
                continue
            var gpt_name = utf16le_name(entries, base + GPT_ENTRY_NAME_OFFSET, GPT_ENTRY_NAME_UNITS)

            var part_start_byte = Int(first_lba) * 512
            var label = read_ext4_label(f, part_start_byte)
            var kind = "ext4"
            if label == "":
                label = read_fat_label(f, part_start_byte)
                kind = "fat32"
            if label == "":
                kind = "unknown"

            out.append(GptPartition(gpt_name, label, kind, first_lba, last_lba, i + 1))
    return out^


def find_boot_disk() raises -> Tuple[String, List[GptPartition]]:
    """Returns the (/dev/<name>, partitions) of the first whole-disk block device
    carrying a valid GPT signature. Raises if none is found: a real blocker, not a
    silent empty result."""
    for name in list_whole_disk_candidates():
        var dev_path = "/dev/" + name
        try:
            var parts = read_gpt(dev_path)
            if len(parts) > 0:
                return (dev_path, parts^)
        except _:
            continue
    raise Error("S0: no whole-disk block device with a valid GPT signature found under /sys/class/block")


def find_by_label(parts: List[GptPartition], label: String) -> Optional[GptPartition]:
    for p in parts:
        if p.fs_label == label:
            return Optional(p.copy())
    return Optional[GptPartition](None)
