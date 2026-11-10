// usage: llama-force MODEL.gguf PROMPTS_DIR KVTYPE gen|force REF.jsonl [QUICK]
//   KVTYPE f32|f16|q8_0 (K and V), flash attention on, n_ctx 32768, one shared-document prefill.
//   gen:   greedy 64 ids per prompt of bench/a2-prompts.sh's sets, written to REF.jsonl
//   force: teacher-forced agreement of this KV type against REF.jsonl's ids, printed per prompt;
//          one token per decode step like gen (a 63-token forced batch reads 127/128 against f32
//          itself: llama.cpp kernels are not batch-invariant)
//   QUICK=N: first N prompts of the 32k set only.
// llama.cpp side of the A2 step 2 bar (bench/PROTOCOL-RULES.md P14): the document prefix lives on
// seq 0 and is extended 8k -> 16k -> 32k; each prompt copies it to seq 1, runs, and drops seq 1.
// C++ because libllama's API passes structs by value; the Mojo engine is not involved.
// Build: g++ -O2 -std=c++17 tools/llama-force.cpp -I ~/llama.cpp/include -I ~/llama.cpp/ggml/include
//        -L ~/llama.cpp/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$HOME/llama.cpp/build/bin
#include "llama.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using ids_t = std::vector<llama_token>;

static ids_t read_ids(const fs::path & p) {
    std::ifstream f(p);
    ids_t v;
    long x;
    while (f >> x) v.push_back((llama_token) x);
    return v;
}

static void die(const std::string & m) {
    fprintf(stderr, "FAIL llama-force: %s\n", m.c_str());
    exit(1);
}

static int decode(llama_context * ctx, llama_batch & b, const llama_token * t, int n, int pos0, int seq, bool last_logits) {
    int nb = llama_n_batch(ctx);
    for (int s = 0; s < n; s += nb) {
        int k = std::min(nb, n - s);
        b.n_tokens = k;
        for (int i = 0; i < k; i++) {
            b.token[i] = t[s + i];
            b.pos[i] = pos0 + s + i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = seq;
            b.logits[i] = last_logits && s + i == n - 1;
        }
        if (llama_decode(ctx, b) != 0) die("llama_decode failed at pos " + std::to_string(pos0 + s));
    }
    return n;
}

static llama_token argmax(const float * l, int nv) {
    return (llama_token) (std::max_element(l, l + nv) - l);
}

int main(int argc, char ** argv) {
    if (argc < 6) die("usage: llama-force MODEL PROMPTS_DIR f32|f16|q8_0 gen|force REF.jsonl [QUICK]");
    std::string model_path = argv[1], mode = argv[4], kvs = argv[3];
    fs::path dir = argv[2], ref = argv[5];
    int quick = argc > 6 ? atoi(argv[6]) : 0;
    ggml_type kvt = kvs == "f32" ? GGML_TYPE_F32 : kvs == "f16" ? GGML_TYPE_F16 : kvs == "q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_COUNT;
    if (kvt == GGML_TYPE_COUNT) die("unknown KV type " + kvs);
    if (mode != "gen" && mode != "force") die("mode must be gen or force");

    std::vector<int> sets = quick ? std::vector<int>{32768} : std::vector<int>{8192, 16384, 32768};
    int nprompt = quick ? quick : 20;
    // a `sets` file in the prompt dir (one L per line) overrides the a2 layout; a set with doclen 0 is a
    // flat set of standalone prompts (short-prompt bars for models whose tokenizer is not the 9B's)
    if (fs::exists(dir / "sets")) { sets.clear(); std::ifstream f(dir / "sets"); int v; while (f >> v) sets.push_back(v); }

    std::map<std::string, ids_t> refids;
    if (mode == "force") {
        std::ifstream f(ref);
        std::string line;
        while (std::getline(f, line)) {
            auto a = line.find("\"key\":\""), b = line.find("\"ids\":[");
            if (a == std::string::npos || b == std::string::npos) continue;
            std::string key = line.substr(a + 7, line.find('"', a + 7) - a - 7);
            std::string arr = line.substr(b + 7, line.find(']', b) - b - 7);
            std::replace(arr.begin(), arr.end(), ',', ' ');
            std::istringstream is(arr);
            ids_t v;
            long x;
            while (is >> x) v.push_back((llama_token) x);
            refids[key] = v;
        }
        if (refids.empty()) die("no reference ids in " + ref.string());
    }

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) die("model load");
    auto cp = llama_context_default_params();
    cp.n_ctx = 32768;
    cp.n_batch = 2048;
    cp.n_ubatch = 512;
    cp.n_seq_max = 2;
    cp.kv_unified = true;
    cp.type_k = kvt;
    cp.type_v = kvt;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_threads = cp.n_threads_batch = 8;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) die("context init");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int nv = llama_vocab_n_tokens(vocab);
    llama_memory_t mem = llama_get_memory(ctx);
    llama_batch b = llama_batch_init(2048, 0, 1);
    printf("LLAMA_FORCE: kv=%s mode=%s n_ctx=%u fa=on quick=%d model=%s\n", kvs.c_str(), mode.c_str(), llama_n_ctx(ctx), quick, model_path.c_str());

    ids_t doc = fs::exists(dir / "L32768" / "p01-water.tokens") ? read_ids(dir / "L32768" / "p01-water.tokens") : ids_t{};
    FILE * out = mode == "gen" ? fopen(ref.c_str(), "w") : nullptr;
    if (mode == "gen" && !out) die("cannot write " + ref.string());
    int have = 0, total_a = 0, total_n = 0;
    for (int L : sets) {
        ids_t dl = read_ids(dir / ("L" + std::to_string(L)) / "doclen");
        if (dl.empty()) die("doclen missing for L" + std::to_string(L));
        int d = dl[0];
        if (d < 0 || (size_t) d > doc.size()) die("document has " + std::to_string(doc.size()) + " tokens (L32768/p01-water.tokens), L" + std::to_string(L) + " doclen needs " + std::to_string(d));
        if (d < have || (d == have && d != 0)) die("doclen not increasing at L" + std::to_string(L));
        if (d > have) decode(ctx, b, doc.data() + have, d - have, have, 0, false);
        have = d;
        std::vector<fs::path> files;
        for (auto & e : fs::directory_iterator(dir / ("L" + std::to_string(L))))
            if (e.path().extension() == ".tokens") files.push_back(e.path());
        std::sort(files.begin(), files.end());
        int k = 0;
        for (auto & p : files) {
            if (++k > nprompt) break;
            ids_t all = read_ids(p);
            if ((int) all.size() <= d || !std::equal(doc.begin(), doc.begin() + d, all.begin())) die(p.string() + " does not extend the document prefix");
            std::string key = "L" + std::to_string(L) + "/" + p.stem().string();
            llama_memory_seq_rm(mem, 1, -1, -1);
            llama_memory_seq_cp(mem, 0, 1, -1, -1);
            int pos = d;
            pos += decode(ctx, b, all.data() + d, (int) all.size() - d, pos, 1, true);
            if (mode == "gen") {
                ids_t g;
                llama_token t = argmax(llama_get_logits_ith(ctx, -1), nv);
                for (int j = 0; j < 64; j++) {
                    g.push_back(t);
                    if (j == 63) break;
                    pos += decode(ctx, b, &t, 1, pos, 1, true);
                    t = argmax(llama_get_logits_ith(ctx, -1), nv);
                }
                fprintf(out, "{\"key\":\"%s\",\"ids\":[", key.c_str());
                for (int j = 0; j < 64; j++) fprintf(out, j ? ",%d" : "%d", g[j]);
                fprintf(out, "]}\n");
                fflush(out);
                printf("gen %s prompt=%zu\n", key.c_str(), all.size() - d);
            } else {
                auto it = refids.find(key);
                if (it == refids.end() || it->second.size() != 64) die("no 64 reference ids for " + key);
                const ids_t & r = it->second;
                int a = argmax(llama_get_logits_ith(ctx, -1), nv) == r[0];
                for (int j = 0; j < 63; j++) {
                    pos += decode(ctx, b, &r[j], 1, pos, 1, true);
                    a += argmax(llama_get_logits_ith(ctx, -1), nv) == r[j + 1];
                }
                total_a += a;
                total_n += 64;
                printf("%s forced agreement: %d / 64\n", key.c_str(), a);
            }
            fflush(stdout);
            llama_memory_seq_rm(mem, 1, -1, -1);
        }
    }
    if (out) fclose(out);
    if (mode == "force") printf("TOTAL forced agreement: %d / %d = %.2f%%\n", total_a, total_n, 100.0 * total_a / std::max(total_n, 1));
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
