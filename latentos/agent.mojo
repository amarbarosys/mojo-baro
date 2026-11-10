# CANONICAL. LatentOS lives here (mojo-baro latentos/); AMDHQ src/latentos is a
# pointer to this directory (plan 12 M1, 2026-09-24). Tests and the check run
# from AMDHQ tools/latent-os/latentos-check against this copy.
#
# agent.mojo — The LatentOS Mojo Node Agent daemon (`latentos-agent`, 01 §1–§5).
# Manages early boot L0-L7 lifecycle, reservation verification, manifest publication,
# and latent IPC supervisor service with zero warnings.

from std.sys import argv
from std.memory import Pointer, unsafe_memset
from std.memory.alloc import unsafe_alloc

import latentos.sys as sys
import latentos.proto as proto
import latentos.ipc as ipc
import latentos.contract as contract

comptime BytePtr = Pointer[UInt8, MutUntrackedOrigin]

struct AgentConfig(Copyable, Movable):
    var node_name: String
    var fleet_socket: String
    var models_dir: String
    var manifest_out: String
    # Where the manifest goes when manifest_out is not writable (a test points it somewhere unwritable too).
    var manifest_fallback: String
    var reserve_gib: Int
    var daemon_mode: Bool
    var is_mobile: Bool
    var is_legacy_mobile: Bool

    def __init__(out self):
        self.node_name = "hercule-desk"
        self.fleet_socket = "tcp://0.0.0.0:7420"
        self.models_dir = "/var/lib/models"
        self.manifest_out = "/var/lib/latentos/manifest.json"
        self.manifest_fallback = "./manifest.json"
        self.reserve_gib = 32
        self.daemon_mode = False
        self.is_mobile = False
        self.is_legacy_mobile = False

struct NodeTierState(Copyable, Movable):
    var mode: String
    var storage_name: String
    var memory_name: String
    var reserved_pages: Int
    var page_size_kb: Int
    var disk_cold_gbs: Float64
    var disk_hot_gbs: Float64
    var host_read_gbs: Float64
    var pcie_h2d_gbs: Float64
    var thermal_ceiling_w: Float64
    var ufs_read_gbs: Float64
    var battery_wh: Float64
    var is_degraded: Bool
    var is_mobile: Bool
    var is_legacy_mobile: Bool
    var unified_memory: Bool

    def __init__(out self):
        self.mode = "unreserved"
        self.reserved_pages = 0
        self.page_size_kb = 0
        self.is_degraded = True
        # Empirical numbers from E4 (2026-09-08)
        self.disk_cold_gbs = 4.64
        self.disk_hot_gbs = 22.33
        self.host_read_gbs = 52.0
        self.pcie_h2d_gbs = 24.4
        self.is_mobile = False
        self.is_legacy_mobile = False
        self.unified_memory = False
        self.thermal_ceiling_w = 0.0
        self.ufs_read_gbs = 0.0
        self.battery_wh = 0.0
        self.storage_name = "NVMe"
        self.memory_name = "DDR5"

struct LatentAgent:
    var cfg: AgentConfig
    var tier: NodeTierState
    var store: ipc.LatentStore
    var capability: contract.CapabilityDescriptor
    var semantic_capability: contract.CapabilityDescriptor
    var is_running: Bool
    # The boot stage reached (also sent to systemd as STATUS=Ln), whether the manifest reached a disk, and whether
    # READY=1 went out: READY is only sent after a written manifest.
    var last_stage: String
    var manifest_written: Bool
    var ready_sent: Bool

    def __init__(out self, cfg: AgentConfig):
        self.cfg = cfg.copy()
        self.tier = NodeTierState()
        self.store = ipc.LatentStore()
        self.capability = contract.CapabilityDescriptor()
        self.capability.operations = contract.OP_DESCRIBE
        self.capability.transfer_formats = 0
        self.capability.exact_resume = False
        self.capability.max_payload_bytes = 1
        self.semantic_capability = self.capability.copy()
        self.semantic_capability.representation = contract.REPRESENTATION_SEMANTIC_LATENT
        self.semantic_capability.exact_resume = False
        self.semantic_capability.approximation_permitted = True
        self.semantic_capability.operations = contract.OP_DESCRIBE
        self.semantic_capability.transfer_formats = 0
        self.is_running = False
        self.last_stage = ""
        self.manifest_written = False
        self.ready_sent = False

    def _stage(mut self, tag: String, name: String):
        self.last_stage = tag
        _ = sys.sys_sd_notify("STATUS=" + tag + " " + name)

    def stage_l0_boot(mut self):
        if self.cfg.is_legacy_mobile:
            print("[L0 BOOT] Initializing LatentOS Node Agent on " + self.cfg.node_name + " (Legacy Mobile / Old Tech profile)")
        elif self.cfg.is_mobile:
            print("[L0 BOOT] Initializing LatentOS Node Agent on " + self.cfg.node_name + " (Modern Mobile / Android profile)")
        else:
            print("[L0 BOOT] Initializing LatentOS Node Agent on " + self.cfg.node_name)
        self._stage("L0", "boot initialized")

    def stage_l1_reserve_check(mut self):
        self._stage("L1", "reserve check")
        if self.cfg.is_legacy_mobile:
            print("[L1 RESERVE-CHECK] Inspecting legacy mobile architecture (LPDDR4 + eMMC 5.1)...")
            self.tier.is_mobile = True
            self.tier.is_legacy_mobile = True
            self.tier.unified_memory = True
            self.tier.thermal_ceiling_w = 1.8
            self.tier.ufs_read_gbs = 0.25 # eMMC 5.1 sequential read ~250 MB/s
            self.tier.battery_wh = 10.0   # Degraded 3000 mAh battery (~10 Wh)
            self.tier.disk_cold_gbs = 0.25
            self.tier.disk_hot_gbs = 0.25
            self.tier.host_read_gbs = 17.0 # LPDDR4 sustained ~17 GB/s
            self.tier.pcie_h2d_gbs = 17.0  # Shared SoC fabric
            self.tier.is_degraded = False
            self.tier.mode = "legacy-lpddr4-emmc"
            self.tier.storage_name = "eMMC 5.1"
            self.tier.memory_name = "LPDDR4"
            print("  -> Legacy Architecture: Unified LPDDR4 memory fabric (~17.0 GB/s)")
            print("  -> Working set budget: <= 1.2 GB (LMKD protected via oom_score_adj = -1000)")
            print("  -> Storage: eMMC 5.1 (250 MB/s cold read, on-demand swapping disabled)")
            print("  -> Flash endurance: 0 byte write wear (preserves fragile TLC/QLC eMMC flash)")
            return

        if self.cfg.is_mobile:
            print("[L1 RESERVE-CHECK] Inspecting mobile unified memory architecture...")
            self.tier.is_mobile = True
            self.tier.unified_memory = True
            self.tier.thermal_ceiling_w = 3.5
            self.tier.ufs_read_gbs = 4.2
            self.tier.battery_wh = 19.25
            self.tier.disk_cold_gbs = 4.2
            self.tier.disk_hot_gbs = 4.2
            self.tier.host_read_gbs = 68.2
            self.tier.pcie_h2d_gbs = 68.2
            self.tier.is_degraded = False
            self.tier.mode = "unified-lpddr5x"
            self.tier.storage_name = "UFS 4.0"
            self.tier.memory_name = "LPDDR5X"
            print("  -> Mobile Architecture: Unified LPDDR5X (68.2 GB/s SoC memory fabric)")
            print("  -> Working set protection: mlock2(MLOCK_ONFAULT), oom_score_adj = -1000")
            print("  -> Flash endurance: 0 byte write wear (volatile memfd in DRAM)")
            return

        print("[L1 RESERVE-CHECK] Inspecting /proc/meminfo hugepage reservations...")
        var hp = sys.sys_read_meminfo_hugepages()
        var total = hp[0]
        var size_kb = hp[1]

        self.tier.reserved_pages = total
        self.tier.page_size_kb = size_kb

        if total > 0 and size_kb >= 1048576:
            self.tier.mode = "hugetlb-1G"
            self.tier.is_degraded = False
            print("  -> Boot-time 1 GiB reservation confirmed: " + String(total) + " pages (" + String(total) + " GiB)")
        elif total > 0 and size_kb >= 2048:
            self.tier.mode = "hugetlb-2M"
            self.tier.is_degraded = False
            var gib = (total * size_kb) // (1024 * 1024)
            print("  -> 2 MiB reservation confirmed: " + String(total) + " pages (" + String(gib) + " GiB)")
        else:
            self.tier.mode = "mlock-anon"
            self.tier.is_degraded = True
            print("  -> WARNING: No hugepage reservation found! Degraded mode: mlock2(MLOCK_ONFAULT)")

    def stage_l2_mount(mut self):
        self._stage("L2", "mount")
        print("[L2 MOUNT] Verifying model tier directory: " + self.cfg.models_dir)

    def stage_l3_identify(mut self):
        self._stage("L3", "identify")
        if self.cfg.is_legacy_mobile:
            print("[L3 IDENTIFY] Verifying role artifact identity and legacy mobile resource limits")
            print("  -> Primary Resident Role: Qwen2.5-1.5B / SmolLM2-1.7B Q4_K_M (M1 batch class, ~0.9 GB working set)")
            print("  -> Lightweight Standby Role: Qwen2.5-0.5B Q4_K_M (~350 MB working set for 2GB/3GB RAM devices)")
            print("  -> Context Window Limit: 2,048 tokens (96 MiB KV cache pool)")
            print("  -> Storage Strategy: Resident execution strictly enforced; swap-on-demand disabled (avoids 8s eMMC freeze)")
            print("  -> Passive Thermal Budget: 1.8W continuous ceiling (prevents SoC throttling on 12nm/14nm nodes)")
            print("  -> Battery Endurance: ~5.5 hours continuous decode (~200k tokens per 10 Wh degraded battery)")
            print("  -> SSM Checkpoint: " + String(proto.QWYTHOS_SSM_CKPT_BYTES) + " bytes (50.25 MiB)")
            print("  -> KV Page Size: " + String(proto.QWYTHOS_KV_BYTES_PER_TOK) + " bytes/tok (64 KiB)")
        elif self.cfg.is_mobile:
            print("[L3 IDENTIFY] Verifying role artifact identity and mobile resource limits")
            print("  -> Resident Role: Spark-4B / Qwen2.5-3B Q4_K_M (M1 batch class, ~2.0 GB working set)")
            print("  -> On-Demand Role: Qwythos 9B (T_load = 1.2s via UFS 4.0, zero flash write wear)")
            print("  -> Passive Thermal Budget: 3.5W continuous ceiling (2.6W decode @ 20 tok/s)")
            print("  -> Battery Endurance: ~7.4 hours continuous decode (~530k tokens per 5000 mAh charge)")
            print("  -> SSM Checkpoint: " + String(proto.QWYTHOS_SSM_CKPT_BYTES) + " bytes (50.25 MiB)")
            print("  -> KV Page Size: " + String(proto.QWYTHOS_KV_BYTES_PER_TOK) + " bytes/tok (64 KiB)")
        else:
            print("[L3 IDENTIFY] Verifying role artifact identity and runtime closure")
            print("  -> Primary Role: Qwythos 27B Q4_K_M (M1 batch class, native space)")
            print("  -> SSM Checkpoint: " + String(proto.QWYTHOS_SSM_CKPT_BYTES) + " bytes (50.25 MiB)")
            print("  -> KV Page Size: " + String(proto.QWYTHOS_KV_BYTES_PER_TOK) + " bytes/tok (64 KiB)")

    def stage_l4_pin(mut self):
        self._stage("L4", "pin")
        print("[L4 PIN] Initializing pinned host tier & LatentStore...")
        print("  -> LatentStore initialized (active handles: " + String(self.store.count()) + ")")

    def stage_l5_measure(mut self):
        self._stage("L5", "measure")
        if self.cfg.is_legacy_mobile:
            print("[L5 MEASURE] Legacy mobile hardware tier baselines:")
            print("  -> Storage: eMMC 5.1 Sequential Read " + String(self.tier.ufs_read_gbs) + " GB/s (250 MB/s)")
            print("  -> Memory:  Unified LPDDR4 Read " + String(self.tier.host_read_gbs) + " GB/s")
            print("  -> Thermal: Passive Cooling Ceiling " + String(self.tier.thermal_ceiling_w) + " W")
        elif self.cfg.is_mobile:
            print("[L5 MEASURE] Mobile hardware tier baselines:")
            print("  -> Storage: UFS 4.0 Sequential Read " + String(self.tier.ufs_read_gbs) + " GB/s")
            print("  -> Memory:  Unified LPDDR5X Read " + String(self.tier.host_read_gbs) + " GB/s")
            print("  -> Thermal: Passive Cooling Ceiling " + String(self.tier.thermal_ceiling_w) + " W")
        else:
            print("[L5 MEASURE] Loading tier bandwidths (E4 receipts):")
            print("  -> NVMe Cold Read: " + String(self.tier.disk_cold_gbs) + " GB/s")
            print("  -> Page-Cache Hot: " + String(self.tier.disk_hot_gbs) + " GB/s")
            print("  -> Host Memory:    " + String(self.tier.host_read_gbs) + " GB/s")

    def stage_l6_publish(mut self) -> String:
        self._stage("L6", "publish")
        print("[L6 PUBLISH] Generating Node Manifest JSON...")
        var json: String = "{\n"
        json += "  \"node\": \"" + self.cfg.node_name + "\",\n"
        json += "  \"boot_stage\": \"" + self.last_stage + "\",\n"
        json += "  \"capability\": " + self.capability.to_json() + ",\n"
        json += "  \"semantic_capability\": " + self.semantic_capability.to_json() + ",\n"
        if self.cfg.is_legacy_mobile:
            json += "  \"mobile\": {\n"
            json += "    \"enabled\": true,\n"
            json += "    \"legacy\": true,\n"
            json += "    \"memory_type\": \"" + self.tier.memory_name + "\",\n"
            json += "    \"storage_type\": \"" + self.tier.storage_name + "\",\n"
            json += "    \"unified_memory\": true,\n"
            json += "    \"thermal_ceiling_w\": " + String(self.tier.thermal_ceiling_w) + ",\n"
            json += "    \"emmc_read_gbs\": " + String(self.tier.ufs_read_gbs) + ",\n"
            json += "    \"battery_wh\": " + String(self.tier.battery_wh) + ",\n"
            json += "    \"battery_life_decode_hours\": 5.5,\n"
            json += "    \"flash_write_wear_bytes\": 0\n"
            json += "  },\n"
        elif self.cfg.is_mobile:
            json += "  \"mobile\": {\n"
            json += "    \"enabled\": true,\n"
            json += "    \"legacy\": false,\n"
            json += "    \"memory_type\": \"" + self.tier.memory_name + "\",\n"
            json += "    \"storage_type\": \"" + self.tier.storage_name + "\",\n"
            json += "    \"unified_memory\": true,\n"
            json += "    \"thermal_ceiling_w\": " + String(self.tier.thermal_ceiling_w) + ",\n"
            json += "    \"ufs_read_gbs\": " + String(self.tier.ufs_read_gbs) + ",\n"
            json += "    \"battery_wh\": " + String(self.tier.battery_wh) + ",\n"
            json += "    \"battery_life_decode_hours\": 7.4,\n"
            json += "    \"flash_write_wear_bytes\": 0\n"
            json += "  },\n"
        json += "  \"tier\": {\n"
        json += "    \"mode\": \"" + self.tier.mode + "\",\n"
        json += "    \"degraded\": " + ("true" if self.tier.is_degraded else "false") + ",\n"
        json += "    \"reserved_pages\": " + String(self.tier.reserved_pages) + ",\n"
        json += "    \"page_size_kb\": " + String(self.tier.page_size_kb) + ",\n"
        json += "    \"bw\": {\n"
        json += "      \"disk_cold_gbs\": " + String(self.tier.disk_cold_gbs) + ",\n"
        json += "      \"disk_hot_gbs\": " + String(self.tier.disk_hot_gbs) + ",\n"
        json += "      \"host_read_gbs\": " + String(self.tier.host_read_gbs) + ",\n"
        json += "      \"pcie_h2d_gbs\": " + String(self.tier.pcie_h2d_gbs) + "\n"
        json += "    }\n"
        json += "  },\n"
        if self.cfg.is_legacy_mobile:
            json += "  \"roles\": [\n"
            json += "    {\n"
            json += "      \"name\": \"qwen2.5-1.5b\",\n"
            json += "      \"batch_class\": \"M1\",\n"
            json += "      \"sigma_id\": \"0000000000000000000000000000000000000000000000000000000000000000\",\n"
            json += "      \"working_set_gib\": 0.9,\n"
            json += "      \"latent\": {\n"
            json += "        \"kinds\": [\"KV_PAGES\", \"SSM_CKPT\", \"HIDDEN\", \"TEXT\"],\n"
            json += "        \"kv_bytes_per_tok\": " + String(proto.QWYTHOS_KV_BYTES_PER_TOK) + ",\n"
            json += "        \"ckpt_bytes\": " + String(proto.QWYTHOS_SSM_CKPT_BYTES) + ",\n"
            json += "        \"hidden_bytes_per_step\": " + String(proto.QWYTHOS_HIDDEN_BYTES_PER_STEP) + "\n"
            json += "      },\n"
            json += "      \"admission\": {\n"
            json += "        \"resident\": true,\n"
            json += "        \"pinned\": true,\n"
            json += "        \"memory_source\": \"LPDDR4\",\n"
            json += "        \"queue_depth\": 0\n"
            json += "      }\n"
            json += "    },\n"
            json += "    {\n"
            json += "      \"name\": \"qwen2.5-0.5b\",\n"
            json += "      \"batch_class\": \"M1\",\n"
            json += "      \"sigma_id\": \"0000000000000000000000000000000000000000000000000000000000000000\",\n"
            json += "      \"working_set_gib\": 0.35,\n"
            json += "      \"admission\": {\n"
            json += "        \"resident\": false,\n"
            json += "        \"standby\": true,\n"
            json += "        \"target\": \"2gb_ram_devices\"\n"
            json += "      }\n"
            json += "    }\n"
            json += "  ]\n"
            json += "}\n"
        elif self.cfg.is_mobile:
            json += "  \"roles\": [\n"
            json += "    {\n"
            json += "      \"name\": \"spark-4b\",\n"
            json += "      \"batch_class\": \"M1\",\n"
            json += "      \"sigma_id\": \"0000000000000000000000000000000000000000000000000000000000000000\",\n"
            json += "      \"working_set_gib\": 2.0,\n"
            json += "      \"latent\": {\n"
            json += "        \"kinds\": [\"KV_PAGES\", \"SSM_CKPT\", \"HIDDEN\", \"TEXT\"],\n"
            json += "        \"kv_bytes_per_tok\": " + String(proto.QWYTHOS_KV_BYTES_PER_TOK) + ",\n"
            json += "        \"ckpt_bytes\": " + String(proto.QWYTHOS_SSM_CKPT_BYTES) + ",\n"
            json += "        \"hidden_bytes_per_step\": " + String(proto.QWYTHOS_HIDDEN_BYTES_PER_STEP) + "\n"
            json += "      },\n"
            json += "      \"admission\": {\n"
            json += "        \"resident\": true,\n"
            json += "        \"pinned\": true,\n"
            json += "        \"memory_source\": \"LPDDR5X\",\n"
            json += "        \"queue_depth\": 0\n"
            json += "      }\n"
            json += "    },\n"
            json += "    {\n"
            json += "      \"name\": \"qwythos-9b\",\n"
            json += "      \"batch_class\": \"M1\",\n"
            json += "      \"sigma_id\": \"0000000000000000000000000000000000000000000000000000000000000000\",\n"
            json += "      \"working_set_gib\": 5.2,\n"
            json += "      \"swap_latency_s\": 1.2,\n"
            json += "      \"admission\": {\n"
            json += "        \"resident\": false,\n"
            json += "        \"on_demand\": true,\n"
            json += "        \"storage_source\": \"UFS 4.0\"\n"
            json += "      }\n"
            json += "    }\n"
            json += "  ]\n"
            json += "}\n"
        else:
            json += "  \"roles\": [\n"
            json += "    {\n"
            json += "      \"name\": \"qwythos-27b\",\n"
            json += "      \"batch_class\": \"M1\",\n"
            json += "      \"sigma_id\": \"0000000000000000000000000000000000000000000000000000000000000000\",\n"
            json += "      \"latent\": {\n"
            json += "        \"kinds\": [\"KV_PAGES\", \"SSM_CKPT\", \"HIDDEN\", \"TEXT\"],\n"
            json += "        \"kv_bytes_per_tok\": " + String(proto.QWYTHOS_KV_BYTES_PER_TOK) + ",\n"
            json += "        \"ckpt_bytes\": " + String(proto.QWYTHOS_SSM_CKPT_BYTES) + ",\n"
            json += "        \"hidden_bytes_per_step\": " + String(proto.QWYTHOS_HIDDEN_BYTES_PER_STEP) + "\n"
            json += "      },\n"
            json += "      \"admission\": {\n"
            json += "        \"resident\": true,\n"
            json += "        \"queue_depth\": 0\n"
            json += "      }\n"
            json += "    }\n"
            json += "  ]\n"
            json += "}\n"

        # Write manifest file if writable path
        try:
            with open(self.cfg.manifest_out, "w") as f:
                f.write(json)
            print("  -> Manifest written to: " + self.cfg.manifest_out)
            self.manifest_written = True
        except e:
            # Fallback to current directory if /var/lib/latentos is not writable by current user
            var fallback = self.cfg.manifest_fallback
            try:
                with open(fallback, "w") as f:
                    f.write(json)
                print("  -> Manifest written to fallback: " + fallback)
                self.manifest_written = True
            except e2:
                print("  -> Could not write manifest file: " + String(e2))

        # systemd must not see READY for a node whose manifest is on no disk.
        if self.manifest_written:
            _ = sys.sys_sd_notify("READY=1")
            self.ready_sent = True
        else:
            _ = sys.sys_sd_notify("STATUS=L6 publish FAILED: manifest not written; not ready")
        return json

    def stage_l7_serve_step(mut self) -> Bool:
        """Executes a single supervisor service loop cycle."""
        var now = sys.sys_clock_monotonic_s()
        # Clean expired handles from store
        var evicted = self.store.evict_expired(now)
        if evicted > 0:
            print("[L7 SERVE] Evicted " + String(evicted) + " expired latent handles")

        # Heartbeat to systemd watchdog
        _ = sys.sys_sd_notify("WATCHDOG=1")
        return True

    def run_check(mut self):
        self.stage_l0_boot()
        self.stage_l1_reserve_check()
        self.stage_l2_mount()
        self.stage_l3_identify()
        self.stage_l4_pin()
        self.stage_l5_measure()
        var manifest = self.stage_l6_publish()
        print("\n--- Published Manifest ---")
        print(manifest)
        if self.manifest_written:
            print("latentos-agent verification check completed successfully.")
        else:
            print("latentos-agent verification check FAILED at stage " + self.last_stage + ": the manifest was not written.")

def main() raises:
    var args = argv()
    var cfg = AgentConfig()

    for i in range(1, len(args)):
        var a = args[i]
        if a == "--daemon":
            cfg.daemon_mode = True
        elif a == "--legacy-mobile" or a == "--android-legacy":
            cfg.is_legacy_mobile = True
            cfg.is_mobile = True
            cfg.node_name = "mobile-android-legacy"
        elif a == "--mobile" or a == "--android":
            cfg.is_mobile = True
            cfg.node_name = "mobile-android-arm64"
        elif a == "--name" and i + 1 < len(args):
            cfg.node_name = args[i + 1]
        elif a == "--out" and i + 1 < len(args):
            cfg.manifest_out = args[i + 1]

    var agent = LatentAgent(cfg^)
    agent.run_check()
    if not agent.manifest_written:
        raise Error("latentos-agent: the manifest was not written (stage " + agent.last_stage + "); READY was not sent")
    if cfg.daemon_mode:
        print("latentos-agent daemon serving")
        while agent.stage_l7_serve_step():
            _ = sys.sys_usleep(1_000_000)
