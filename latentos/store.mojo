import latentos.contract as contract
import latentos.proto as proto


comptime FNV_OFFSET: UInt64 = 0xcbf29ce484222325
comptime FNV_PRIME: UInt64 = 0x100000001b3


def fnv_payload_tag(payload: List[UInt8]) -> UInt64:
    var h = FNV_OFFSET
    for i in range(len(payload)):
        h = (h ^ UInt64(payload[i])) * FNV_PRIME
    return h


struct StoreLimits(Copyable, Movable):
    var max_handles: Int
    var max_bytes: UInt64
    var tier_bytes: Array[UInt64, 3]
    var default_ttl_s: Int64

    def __init__(out self):
        self.max_handles = 64
        self.max_bytes = UInt64(1) << 30
        self.tier_bytes = Array[UInt64, 3](fill=UInt64(1) << 28)
        self.default_ttl_s = 600

    @staticmethod
    def tier_index(tier: UInt8) raises -> Int:
        if tier == contract.TIER_GPU:
            return 0
        if tier == contract.TIER_PINNED:
            return 1
        if tier == contract.TIER_DISK:
            return 2
        raise Error("TIER:unknown")



struct StateHandle(Copyable, Movable):
    var address: String
    var generation: UInt64
    var role: String

    def __init__(out self, address: String = "", generation: UInt64 = 0, role: String = ""):
        self.address = address
        self.generation = generation
        self.role = role


struct ExportedState(Movable):
    var representation: UInt8
    var header: proto.LatentHeader
    var binding: contract.BindingManifest
    var authority: contract.Authority
    var payload: List[UInt8]
    var role: String
    var parent_address: String
    var tier: UInt8

    def __init__(
        out self,
        representation: UInt8,
        header: proto.LatentHeader,
        binding: contract.BindingManifest,
        authority: contract.Authority,
        var payload: List[UInt8],
        role: String,
        parent_address: String,
        tier: UInt8,
    ):
        self.representation = representation
        self.header = header.copy()
        self.binding = binding.copy()
        self.authority = authority.copy()
        self.payload = payload^
        self.role = role
        self.parent_address = parent_address
        self.tier = tier


struct StateRecord(Movable):
    var address: String
    var generation: UInt64
    var representation: UInt8
    var header: proto.LatentHeader
    var binding: contract.BindingManifest
    var authority: contract.Authority
    var payload: List[UInt8]
    var payload_len: Int
    var payload_tag: UInt64
    var role: String
    var parent_address: String
    var payload_sha: Array[UInt8, 32]
    var tier: UInt8
    var mutable: Bool
    var leases: Int
    var child_refs: Int
    var expires_at_s: Int64
    var weight_owner: String

    def __init__(
        out self,
        address: String,
        generation: UInt64,
        representation: UInt8,
        header: proto.LatentHeader,
        binding: contract.BindingManifest,
        authority: contract.Authority,
        var payload: List[UInt8],
        role: String,
        parent_address: String,
        tier: UInt8,
        mutable: Bool,
        expires_at_s: Int64,
        weight_owner: String,
    ):
        self.address = address
        self.generation = generation
        self.representation = representation
        self.header = header.copy()
        self.binding = binding.copy()
        self.authority = authority.copy()
        self.payload = payload^
        self.payload_len = len(self.payload)
        self.payload_tag = fnv_payload_tag(self.payload)
        self.payload_sha = contract.payload_sha256(self.payload)
        self.role = role
        self.parent_address = parent_address
        self.tier = tier
        self.mutable = mutable
        self.leases = 1
        self.child_refs = 0
        self.expires_at_s = expires_at_s
        self.weight_owner = weight_owner


struct ReferenceStateStore(Movable):
    var records: List[StateRecord]
    var limits: StoreLimits
    var identity: contract.EngineIdentity
    var capability: contract.CapabilityDescriptor
    var semantic_capability: contract.CapabilityDescriptor
    var next_generation: UInt64
    var retained_bytes: UInt64
    var tier_used: Array[UInt64, 3]
    var loaded_weight_count: Int
    var weight_owner: String
    var tenant: String
    var room: String
    var session: String
    var recipient: String

    def __init__(
        out self,
        identity: contract.EngineIdentity,
        limits: StoreLimits,
        tenant: String = "tenant",
        room: String = "room",
        session: String = "session",
        recipient: String = "recipient",
    ):
        self.records = List[StateRecord]()
        self.limits = limits.copy()
        self.identity = identity.copy()
        self.capability = contract.CapabilityDescriptor()
        self.capability.weights_digest = contract.identity_weights_digest(identity)
        self.capability.tokenizer_digest = identity.tokenizer_sha.copy()
        self.capability.runtime_digest = identity.runtime.copy()
        self.capability.operations = contract.OP_DESCRIBE | contract.OP_CREATE | contract.OP_FORK | contract.OP_RESUME | contract.OP_RELEASE | contract.OP_EXPORT | contract.OP_IMPORT
        self.capability.transfer_formats = 15
        self.capability.max_payload_bytes = limits.max_bytes
        self.semantic_capability = self.capability.copy()
        self.semantic_capability.representation = contract.REPRESENTATION_SEMANTIC_LATENT
        self.semantic_capability.exact_resume = False
        self.semantic_capability.approximation_permitted = True
        self.semantic_capability.operations = contract.OP_DESCRIBE | contract.OP_IMPORT | contract.OP_CONSUME_SEMANTIC
        self.semantic_capability.transfer_formats = 15
        self.semantic_capability.max_payload_bytes = limits.max_bytes
        self.next_generation = 1
        self.retained_bytes = 0
        self.tier_used = Array[UInt64, 3](fill=0)
        self.loaded_weight_count = 1
        self.weight_owner = "reference"
        self.tenant = tenant
        self.room = room
        self.session = session
        self.recipient = recipient

    def describe(self) raises -> String:
        self.capability.validate()
        return self.capability.to_json()

    def describe_semantic(self) raises -> String:
        self.semantic_capability.validate()
        return self.semantic_capability.to_json()

    def _index(self, handle: StateHandle, now_s: Int64) raises -> Int:
        if handle.address == "" or handle.generation == 0:
            raise Error("STATE:invalid_handle")
        for i in range(len(self.records)):
            if self.records[i].address == handle.address and self.records[i].generation == handle.generation:
                if now_s >= self.records[i].expires_at_s:
                    raise Error("STATE:expired")
                return i
        raise Error("STATE:stale_handle")

    def _admit(self, payload_len: Int, tier: UInt8) raises:
        if len(self.records) >= self.limits.max_handles:
            raise Error("BACKPRESSURE:handle_limit")
        var bytes = UInt64(payload_len)
        if bytes > self.limits.max_bytes or self.retained_bytes > self.limits.max_bytes - bytes:
            raise Error("BACKPRESSURE:byte_limit")
        var ti = StoreLimits.tier_index(tier)
        var limit = self.limits.tier_bytes[ti]
        if self.tier_used[ti] > limit or bytes > limit - self.tier_used[ti]:
            raise Error("BACKPRESSURE:tier_limit")

    def _address(self, header: proto.LatentHeader) raises -> String:
        if not header.is_valid() and header.kind != proto.KIND_SEMANTIC_LATENT:
            raise Error("STATE:invalid_header")
        return header.content_address() + ":" + String(self.next_generation)

    def create(
        mut self,
        header: proto.LatentHeader,
        payload: List[UInt8],
        role: String,
        authority: contract.Authority,
        tier: UInt8,
        now_s: Int64,
    ) raises -> StateHandle:
        var h = header.copy()
        if h.ttl_s == 0:
            if self.limits.default_ttl_s <= 0 or self.limits.default_ttl_s > 4294967295:
                raise Error("STATE:invalid_ttl")
            h.ttl_s = UInt32(self.limits.default_ttl_s)
        var lease = Int64(h.ttl_s)
        if lease <= 0:
            raise Error("STATE:invalid_ttl")
        if h.payload_len == 0:
            h.payload_len = UInt64(len(payload))
        if h.payload_len != UInt64(len(payload)):
            raise Error("STATE:length_mismatch")
        if h.kind != proto.KIND_SEMANTIC_LATENT:
            self.identity.check_exact(h)
        authority.validate(self.tenant, self.room, self.session, self.recipient, now_s)
        if h.kind == proto.KIND_SEMANTIC_LATENT:
            raise Error("STATE:semantic_requires_binding")
        var digest = contract.payload_sha256(payload)
        if contract.nonzero32(h.payload_sha) and not contract.bytes_equal(h.payload_sha, digest):
            raise Error("STATE:payload_sha")
        h.payload_sha = digest.copy()
        self._admit(len(payload), tier)
        var address = self._address(h)
        var generation = self.next_generation
        self.next_generation += 1
        var record = StateRecord(
            address,
            generation,
            contract.REPRESENTATION_EXECUTION_CHECKPOINT,
            h,
            contract.BindingManifest(),
            authority,
            payload.copy(),
            role,
            "",
            tier,
            False,
            now_s + lease,
            self.weight_owner,
        )
        self.records.append(record^)
        self.retained_bytes += UInt64(len(payload))
        self.tier_used[StoreLimits.tier_index(tier)] += UInt64(len(payload))
        return StateHandle(address, generation, role)

    def create_semantic(
        mut self,
        binding: contract.BindingManifest,
        payload: List[UInt8],
        authority: contract.Authority,
        role: String,
        tier: UInt8,
        now_s: Int64,
    ) raises -> StateHandle:
        var b = binding.copy()
        if b.payload_len != UInt64(len(payload)):
            raise Error("STATE:semantic_length_mismatch")
        b.validate(now_s)
        authority.validate(self.tenant, self.room, self.session, self.recipient, now_s)
        var digest = contract.payload_sha256(payload)
        if contract.nonzero32(b.payload_digest) and not contract.bytes_equal(b.payload_digest, digest):
            raise Error("BINDING:payload_sha")
        b.payload_digest = digest.copy()
        self._admit(len(payload), tier)
        if self.limits.default_ttl_s <= 0 or self.limits.default_ttl_s > 4294967295:
            raise Error("STATE:invalid_ttl")
        var semantic_lease = self.limits.default_ttl_s
        var header = proto.LatentHeader()
        header.kind = proto.KIND_SEMANTIC_LATENT
        header.dtype = b.dtype
        header.payload_len = UInt64(len(payload))
        header.ttl_s = UInt32(semantic_lease)
        var address = self._address(header)
        var generation = self.next_generation
        self.next_generation += 1
        var record = StateRecord(
            address,
            generation,
            contract.REPRESENTATION_SEMANTIC_LATENT,
            header,
            b,
            authority,
            payload.copy(),
            role,
            "",
            tier,
            False,
            now_s + semantic_lease,
            self.weight_owner,
        )
        self.records.append(record^)
        self.retained_bytes += UInt64(len(payload))
        self.tier_used[StoreLimits.tier_index(tier)] += UInt64(len(payload))
        return StateHandle(address, generation, role)

    def fork(mut self, parent: StateHandle, role: String, now_s: Int64) raises -> StateHandle:
        var pi = self._index(parent, now_s)
        if self.records[pi].representation != contract.REPRESENTATION_EXECUTION_CHECKPOINT:
            raise Error("STATE:semantic_fork_forbidden")
        self._admit(self.records[pi].payload_len, self.records[pi].tier)
        var record_bytes = self.records[pi].payload_len
        var record_tier = self.records[pi].tier
        var h = self.records[pi].header.copy()
        var address = self._address(h)
        var generation = self.next_generation
        self.next_generation += 1
        var record = StateRecord(
            address,
            generation,
            contract.REPRESENTATION_EXECUTION_CHECKPOINT,
            h,
            self.records[pi].binding.copy(),
            self.records[pi].authority.copy(),
            self.records[pi].payload.copy(),
            role,
            self.records[pi].address,
            self.records[pi].tier,
            True,
            now_s + Int64(h.ttl_s),
            self.weight_owner,
        )
        self.records[pi].child_refs += 1
        self.records.append(record^)
        self.retained_bytes += UInt64(record_bytes)
        self.tier_used[StoreLimits.tier_index(record_tier)] += UInt64(record_bytes)
        return StateHandle(address, generation, role)

    def append_branch(mut self, branch: StateHandle, suffix: List[UInt8], now_s: Int64) raises:
        var i = self._index(branch, now_s)
        if not self.records[i].mutable:
            raise Error("STATE:immutable_parent")
        if len(suffix) == 0:
            raise Error("STATE:empty_mutation")
        var add_bytes = UInt64(len(suffix))
        if add_bytes > self.limits.max_bytes or self.retained_bytes > self.limits.max_bytes - add_bytes:
            raise Error("BACKPRESSURE:byte_limit")
        var tier_index = StoreLimits.tier_index(self.records[i].tier)
        if add_bytes > self.limits.tier_bytes[tier_index] or self.tier_used[tier_index] > self.limits.tier_bytes[tier_index] - add_bytes:
            raise Error("BACKPRESSURE:tier_limit")
        for j in range(len(suffix)):
            self.records[i].payload.append(suffix[j])
        self.records[i].payload_len = len(self.records[i].payload)
        self.records[i].header.payload_len = UInt64(self.records[i].payload_len)
        self.records[i].payload_tag = fnv_payload_tag(self.records[i].payload)
        self.records[i].expires_at_s = now_s + Int64(self.records[i].header.ttl_s)
        self.records[i].payload_sha = contract.payload_sha256(self.records[i].payload)
        self.records[i].header.payload_sha = self.records[i].payload_sha.copy()
        self.retained_bytes += UInt64(len(suffix))
        self.tier_used[StoreLimits.tier_index(self.records[i].tier)] += UInt64(len(suffix))

    def check_authority(
        self,
        handle: StateHandle,
        tenant: String,
        room: String,
        session: String,
        recipient: String,
        now_s: Int64,
    ) raises:
        var i = self._index(handle, now_s)
        self.records[i].authority.validate(tenant, room, session, recipient, now_s)

    def resume(mut self, handle: StateHandle, now_s: Int64) raises -> List[UInt8]:
        var i = self._index(handle, now_s)
        if self.records[i].representation != contract.REPRESENTATION_EXECUTION_CHECKPOINT:
            raise Error("STATE:semantic_requires_consume")
        if fnv_payload_tag(self.records[i].payload) != self.records[i].payload_tag:
            raise Error("STATE:payload_corrupt")
        if not contract.bytes_equal(self.records[i].header.payload_sha, self.records[i].payload_sha):
            raise Error("STATE:payload_sha")
        self.records[i].expires_at_s = now_s + Int64(self.records[i].header.ttl_s)
        return self.records[i].payload.copy()

    def consume_semantic(self, handle: StateHandle, now_s: Int64) raises -> List[UInt8]:
        var i = self._index(handle, now_s)
        if self.records[i].representation != contract.REPRESENTATION_SEMANTIC_LATENT:
            raise Error("STATE:exact_requires_resume")
        self.records[i].binding.validate(now_s)
        if not contract.bytes_equal(self.records[i].binding.payload_digest, self.records[i].payload_sha):
            raise Error("BINDING:payload_sha")
        if not self.records[i].binding.matches_receiver(self.identity):
            raise Error("BINDING:receiver_mismatch")
        if fnv_payload_tag(self.records[i].payload) != self.records[i].payload_tag:
            raise Error("STATE:payload_corrupt")
        return self.records[i].payload.copy()

    def export(self, handle: StateHandle, now_s: Int64) raises -> ExportedState:
        var i = self._index(handle, now_s)
        if fnv_payload_tag(self.records[i].payload) != self.records[i].payload_tag:
            raise Error("STATE:payload_corrupt")
        if self.records[i].representation == contract.REPRESENTATION_EXECUTION_CHECKPOINT:
            if not contract.bytes_equal(self.records[i].header.payload_sha, self.records[i].payload_sha):
                raise Error("STATE:payload_sha")
        elif not contract.bytes_equal(self.records[i].binding.payload_digest, self.records[i].payload_sha):
            raise Error("BINDING:payload_sha")
        var out = ExportedState(
            self.records[i].representation,
            self.records[i].header,
            self.records[i].binding,
            self.records[i].authority,
            self.records[i].payload.copy(),
            self.records[i].role,
            self.records[i].parent_address,
            self.records[i].tier,
        )
        return out^

    def import_exact(
        mut self,
        state: ExportedState,
        now_s: Int64,
    ) raises -> StateHandle:
        if state.representation != contract.REPRESENTATION_EXECUTION_CHECKPOINT:
            raise Error("STATE:import_representation")
        if not state.header.is_valid() or state.header.payload_len != UInt64(len(state.payload)):
            raise Error("STATE:import_invalid")
        self.identity.check_exact(state.header)
        state.authority.validate(self.tenant, self.room, self.session, self.recipient, now_s)
        var imported_header = state.header.copy()
        var digest = contract.payload_sha256(state.payload)
        if contract.nonzero32(imported_header.payload_sha) and not contract.bytes_equal(imported_header.payload_sha, digest):
            raise Error("STATE:payload_sha")
        if imported_header.ttl_s == 0:
            raise Error("STATE:invalid_ttl")
        imported_header.payload_sha = digest.copy()
        self._admit(len(state.payload), state.tier)
        var address = self._address(imported_header)
        var generation = self.next_generation
        self.next_generation += 1
        var record = StateRecord(
            address,
            generation,
            state.representation,
            imported_header,
            state.binding,
            state.authority,
            state.payload.copy(),
            state.role,
            "",
            state.tier,
            False,
            now_s + Int64(imported_header.ttl_s),
            self.weight_owner,
        )
        self.records.append(record^)
        self.retained_bytes += UInt64(len(state.payload))
        self.tier_used[StoreLimits.tier_index(state.tier)] += UInt64(len(state.payload))
        return StateHandle(address, generation, state.role)

    def _remove_ancestors(mut self, address: String) raises:
        var next = address
        while next != "":
            var found = -1
            for j in range(len(self.records)):
                if self.records[j].address == next:
                    found = j
                    break
            if found < 0:
                return
            if self.records[found].child_refs > 0:
                self.records[found].child_refs -= 1
            if self.records[found].leases > 0 or self.records[found].child_refs > 0:
                return
            var parent = self.records[found].parent_address
            var tier = self.records[found].tier
            var bytes = UInt64(self.records[found].payload_len)
            _ = self.records.pop(found)
            self.retained_bytes -= bytes
            self.tier_used[StoreLimits.tier_index(tier)] -= bytes
            next = parent

    def release(mut self, handle: StateHandle, now_s: Int64) raises -> Bool:
        var i = self._index(handle, now_s)
        if self.records[i].leases <= 0:
            # Every lease is already released and the record stays only because a child references it: False
            # means "still referenced" and nothing else, so a second release is named.
            raise Error("STATE:already_released")
        self.records[i].leases -= 1
        if self.records[i].leases > 0 or self.records[i].child_refs > 0:
            return False
        var parent = self.records[i].parent_address
        var tier = self.records[i].tier
        var bytes = UInt64(self.records[i].payload_len)
        _ = self.records.pop(i)
        self.retained_bytes -= bytes
        self.tier_used[StoreLimits.tier_index(tier)] -= bytes
        if parent != "":
            self._remove_ancestors(parent)
        return True

    def sweep(mut self, now_s: Int64) raises -> Int:
        var removed = 0
        var i = len(self.records) - 1
        while i >= 0:
            if now_s >= self.records[i].expires_at_s:
                var parent = self.records[i].parent_address
                var tier = self.records[i].tier
                var bytes = UInt64(self.records[i].payload_len)
                _ = self.records.pop(i)
                self.retained_bytes -= bytes
                self.tier_used[StoreLimits.tier_index(tier)] -= bytes
                if parent != "":
                    self._remove_ancestors(parent)
                removed += 1
            i -= 1
        return removed

    def count(self) -> Int:
        return len(self.records)

    def bytes_retained(self) -> UInt64:
        return self.retained_bytes

    def loaded_weights(self) -> Int:
        return self.loaded_weight_count
