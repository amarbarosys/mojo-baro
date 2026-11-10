from grammar.vocab import Vocab


struct TrieNode(Copyable, Movable):
    var token_id: Int32
    var child_byte: List[UInt8]
    var child_idx: List[Int32]

    def __init__(out self):
        self.token_id = -1
        self.child_byte = []
        self.child_idx = []

    def find_child(self, b: UInt8) -> Int32:
        for i in range(len(self.child_byte)):
            if self.child_byte[i] == b:
                return self.child_idx[i]
        return -1


struct TokenTrie(Copyable, Movable):
    var nodes: List[TrieNode]
    var root_dispatch: List[Int32]

    def __init__(out self):
        self.nodes = [TrieNode()]
        self.root_dispatch = []
        for _ in range(256):
            self.root_dispatch.append(-1)

    def insert(mut self, tok_bytes: List[UInt8], token_id: Int):
        if len(tok_bytes) == 0:
            return
        var first = tok_bytes[0]
        var cur: Int
        var rd = self.root_dispatch[Int(first)]
        if rd < 0:
            self.nodes.append(TrieNode())
            cur = len(self.nodes) - 1
            self.root_dispatch[Int(first)] = Int32(cur)
            self.nodes[0].child_byte.append(first)
            self.nodes[0].child_idx.append(Int32(cur))
        else:
            cur = Int(rd)
        for i in range(1, len(tok_bytes)):
            var b = tok_bytes[i]
            var nxt = self.nodes[cur].find_child(b)
            if nxt < 0:
                self.nodes.append(TrieNode())
                nxt = Int32(len(self.nodes) - 1)
                self.nodes[cur].child_byte.append(b)
                self.nodes[cur].child_idx.append(Int32(nxt))
            cur = Int(nxt)
        self.nodes[cur].token_id = Int32(token_id)

    def root_child(self, b: UInt8) -> Int32:
        return self.root_dispatch[Int(b)]


def build_trie(vocab: Vocab) -> TokenTrie:
    var t = TokenTrie()
    for tid in range(vocab.vocab_size):
        if vocab.is_special[tid]:
            continue
        t.insert(vocab.token_bytes[tid], tid)
    return t^
