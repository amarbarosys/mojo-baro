import latentos.contract as contract
import latentos.proto as proto
import latentos.store as store


struct LatentOSClient(Movable):
    var _backend: store.ReferenceStateStore
    var identity: contract.EngineIdentity
    var tenant: String
    var room: String
    var session: String
    var recipient: String

    def __init__(
        out self,
        identity: contract.EngineIdentity,
        limits: store.StoreLimits,
        tenant: String = "tenant",
        room: String = "room",
        session: String = "session",
        recipient: String = "recipient",
    ):
        self.identity = identity.copy()
        self._backend = store.ReferenceStateStore(identity, limits, tenant, room, session, recipient)
        self.tenant = tenant
        self.room = room
        self.session = session
        self.recipient = recipient

    def describe(self) raises -> String:
        return self._backend.describe()

    def describe_semantic(self) raises -> String:
        return self._backend.describe_semantic()

    def create(
        mut self,
        header: proto.LatentHeader,
        payload: List[UInt8],
        authority: contract.Authority,
        role: String,
        tier: UInt8,
        now_s: Int64,
    ) raises -> store.StateHandle:
        authority.validate(self.tenant, self.room, self.session, self.recipient, now_s)
        return self._backend.create(header, payload, role, authority, tier, now_s)

    def fork(mut self, parent: store.StateHandle, role: String, now_s: Int64) raises -> store.StateHandle:
        self._backend.check_authority(parent, self.tenant, self.room, self.session, self.recipient, now_s)
        return self._backend.fork(parent, role, now_s)

    def resume(mut self, handle: store.StateHandle, now_s: Int64) raises -> List[UInt8]:
        self._backend.check_authority(handle, self.tenant, self.room, self.session, self.recipient, now_s)
        return self._backend.resume(handle, now_s)

    def release(mut self, handle: store.StateHandle, now_s: Int64) raises -> Bool:
        self._backend.check_authority(handle, self.tenant, self.room, self.session, self.recipient, now_s)
        return self._backend.release(handle, now_s)

    def export(mut self, handle: store.StateHandle, now_s: Int64) raises -> store.ExportedState:
        self._backend.check_authority(handle, self.tenant, self.room, self.session, self.recipient, now_s)
        return self._backend.export(handle, now_s)

    def import_exact(mut self, state: store.ExportedState, now_s: Int64) raises -> store.StateHandle:
        state.authority.validate(self.tenant, self.room, self.session, self.recipient, now_s)
        return self._backend.import_exact(state, now_s)

    def append_branch(mut self, handle: store.StateHandle, suffix: List[UInt8], now_s: Int64) raises:
        self._backend.check_authority(handle, self.tenant, self.room, self.session, self.recipient, now_s)
        self._backend.append_branch(handle, suffix, now_s)

    def create_semantic(
        mut self,
        binding: contract.BindingManifest,
        payload: List[UInt8],
        authority: contract.Authority,
        role: String,
        tier: UInt8,
        now_s: Int64,
    ) raises -> store.StateHandle:
        authority.validate(self.tenant, self.room, self.session, self.recipient, now_s)
        return self._backend.create_semantic(binding, payload, authority, role, tier, now_s)

    def import_semantic(
        mut self,
        binding: contract.BindingManifest,
        payload: List[UInt8],
        authority: contract.Authority,
        role: String,
        tier: UInt8,
        now_s: Int64,
    ) raises -> store.StateHandle:
        return self.create_semantic(binding, payload, authority, role, tier, now_s)

    def consume_semantic(mut self, handle: store.StateHandle, now_s: Int64) raises -> List[UInt8]:
        self._backend.check_authority(handle, self.tenant, self.room, self.session, self.recipient, now_s)
        return self._backend.consume_semantic(handle, now_s)

    def envelope(mut self, handle: store.StateHandle, now_s: Int64) raises -> contract.StateEnvelope:
        var state = self.export(handle, now_s)
        var out = contract.StateEnvelope()
        out.representation = state.representation
        out.header = state.header.copy()
        out.binding = state.binding.copy()
        out.authority = state.authority.copy()
        out.payload_len = UInt64(len(state.payload))
        out.payload_tag = store.fnv_payload_tag(state.payload)
        return out^

    def sweep(mut self, now_s: Int64) raises -> Int:
        return self._backend.sweep(now_s)

    def count(self) -> Int:
        return self._backend.count()

    def retained_bytes(self) -> UInt64:
        return self._backend.bytes_retained()

    def loaded_weights(self) -> Int:
        return self._backend.loaded_weights()
