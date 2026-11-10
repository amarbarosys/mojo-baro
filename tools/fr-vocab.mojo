# Frequency-ranked draft vocabulary for FR-Spec (arXiv 2506.07900, sec. 4.1.1).
#
# usage: mojo run tools/fr-vocab.mojo MODEL.gguf VOCAB OUT_IDS K CORPUS_FILE [CORPUS_FILE ...]
#
# Concatenates the corpus files into .work/fr-corpus.txt, tokenizes it with the
# engine's own tokenizer (.work/baro-tokenize, bit-equal to llama.cpp), counts
# ids, and writes the top K to OUT_IDS one per line: count descending, lower id
# first on ties, ids never seen ranked by id after every seen id. Prints the
# share of corpus tokens the top K cover (P-FR5 in bench/mtp-protocol.md).
from std.subprocess import run
from std.sys import argv

from tokenizer import Tokenizer


def main() raises:
    var a = argv()
    if len(a) < 6:
        print("usage: fr-vocab MODEL.gguf VOCAB OUT_IDS K CORPUS_FILE [CORPUS_FILE ...]")
        return
    var model = String(a[1])
    var vocab = Int(String(a[2]))
    var out = String(a[3])
    var k = Int(String(a[4]))
    if vocab >= 262144 or k <= 0 or k > vocab:
        raise Error("fr-vocab: need 0 < K <= VOCAB < 262144")
    var text = String()
    for i in range(5, len(a)):
        with open(String(a[i]), "r") as f:
            text += f.read()
        text += "\n"
    with open(".work/fr-corpus.txt", "w") as f:
        f.write(text)
    var ids = run(".work/baro-tokenize encode .work/fr-corpus.txt " + model)
    var counts = List[Int](length=vocab, fill=0)
    var total = 0
    for line in ids.splitlines():
        var s = String(line.strip())
        if s.byte_length() == 0:
            continue
        var c0 = Int(s.as_bytes()[0])
        if c0 < 48 or c0 > 57:
            continue
        counts[Int(s)] += 1
        total += 1
    if total == 0:
        raise Error("fr-vocab: tokenizer produced no ids")
    # Ids the corpus never produced are ranked by the BPE merge that builds
    # them: the merge list is ordered by pair frequency in the tokenizer's own
    # (large, multilingual) training corpus, so an early merge is a common
    # token. Ids with no merge (bytes, specials) rank first.
    var tok = Tokenizer(model)
    var mrank = List[Int](length=vocab, fill=0)
    for e in tok.merge_of.items():
        var got = e.value & 0xFFFFFFFF
        if got < vocab:
            mrank[got] = (e.value >> 32) + 1
    var keys = List[Int](capacity=vocab)
    for t in range(vocab):
        if counts[t] > 0:
            keys.append((counts[t] + 262144) * 262144 + (262143 - t))
        else:
            keys.append((262143 - min(mrank[t], 262143)) * 262144 + (262143 - t))
    sort(keys)
    var body = String()
    var covered = 0
    var distinct = 0
    for t in range(vocab):
        if counts[t] > 0:
            distinct += 1
    for r in range(k):
        var key = keys[vocab - 1 - r]
        var t = 262143 - key % 262144
        covered += counts[t]
        body += String(t) + "\n"
    with open(out, "w") as f:
        f.write(body)
    print("corpus", total, "tokens,", distinct, "distinct of", vocab, "; top", k, "cover",
          Float64(covered) / Float64(total), "->", out)
