# CANONICAL. LatentOS lives here (mojo-baro latentos/); AMDHQ src/latentos is a
# pointer to this directory (plan 12 M1, 2026-09-24). Tests and the check run
# from AMDHQ tools/latent-os/latentos-check against this copy.
#
# __init__.mojo — liblatentos package root

from .sys import (
    sys_memfd_create,
    sys_ftruncate,
    sys_fcntl_add_seals,
    sys_fcntl_get_seals,
    sys_mmap,
    sys_munmap,
    sys_mlock2,
    sys_munlock,
    sys_mincore,
    sys_close,
    sys_clock_monotonic_ns,
    sys_clock_monotonic_us,
    sys_clock_monotonic_s,
    sys_posix_spawn,
    sys_waitpid,
    sys_kill,
    sys_sd_notify,
    sys_read_meminfo_hugepages,
)

from .proto import (
    LATENT_MAGIC,
    LATENT_VERSION,
    LATENT_HEADER_SIZE,
    KIND_KV_PAGES,
    KIND_SSM_CKPT,
    KIND_HIDDEN,
    KIND_LOGITS_TOPK,
    KIND_SEMANTIC_LATENT,
    KIND_TEXT,
    DTYPE_F32,
    DTYPE_BF16,
    DTYPE_INT8_BLOCK,
    BATCH_M1,
    BATCH_FIXED_M,
    BATCH_DYN,
    QWYTHOS_KV_BYTES_PER_TOK,
    QWYTHOS_SSM_CKPT_BYTES,
    QWYTHOS_HIDDEN_BYTES_PER_STEP,
    LatentHeader,
    byte_to_hex,
    u64_to_hex,
    headers_equal,
)

from .contract import (
    CONTRACT_VERSION,
    OP_DESCRIBE,
    OP_CREATE,
    OP_FORK,
    OP_RESUME,
    OP_RELEASE,
    OP_EXPORT,
    OP_IMPORT,
    OP_CONSUME_SEMANTIC,
    REPRESENTATION_EXECUTION_CHECKPOINT,
    REPRESENTATION_SEMANTIC_LATENT,
    TIER_GPU,
    TIER_PINNED,
    TIER_DISK,
    EngineIdentity,
    CapabilityDescriptor,
    Authority,
    BindingManifest,
    StateEnvelope,
)

from .store import StoreLimits, StateHandle, ExportedState
from .client import LatentOSClient
from .binding import AffineBinding, load_affine
from .router import RouteTerms, RouteInput, RouteDecision, choose_route, route_name

from .ipc import (
    create_ipc_pair,
    mint_memfd_rw,
    seal_and_finalize,
    map_readonly,
    send_handle,
    recv_handle,
    LatentStore,
)
