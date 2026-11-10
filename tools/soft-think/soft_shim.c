// Thin C glue between tools/soft-think/soft-think.mojo and libllama (llama.cpp): Mojo cannot pass llama.cpp's
// parameter structs by value, so every call here takes and returns scalars and pointers only. One model, one
// context, one sequence. st_embed_row dequantizes one token_embd row with ggml's own to_float from the mmapped
// GGUF, so a soft (probability-weighted) input embedding is built from the exact table llama.cpp embeds with.
// Build: tools/soft-think/build.sh
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "llama.h"
#include "gguf.h"
#include "ggml.h"

static struct llama_model *g_model;
static struct llama_context *g_ctx;
static const struct llama_vocab *g_vocab;
static struct llama_batch g_tb, g_eb;
static int g_pos, g_nbatch, g_embd;
static const uint8_t *g_map;
static size_t g_maplen, g_emb_off, g_row_bytes;
static ggml_to_float_t g_to_float;

int st_open(const char *path, int n_ctx, int ngl) {
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    g_model = llama_model_load_from_file(path, mp);
    if (!g_model) return -1;
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = 2048;
    cp.n_ubatch = 512;
    g_ctx = llama_init_from_model(g_model, cp);
    if (!g_ctx) return -2;
    g_vocab = llama_model_get_vocab(g_model);
    g_embd = llama_model_n_embd(g_model);
    g_nbatch = 2048;
    g_tb = llama_batch_init(g_nbatch, 0, 1);
    g_eb = llama_batch_init(1, g_embd, 1);
    struct gguf_init_params gp = {.no_alloc = true, .ctx = NULL};
    struct gguf_context *gg = gguf_init_from_file(path, gp);
    if (!gg) return -3;
    int64_t ti = gguf_find_tensor(gg, "token_embd.weight");
    if (ti < 0) return -4;
    enum ggml_type ty = gguf_get_tensor_type(gg, ti);
    g_emb_off = gguf_get_data_offset(gg) + gguf_get_tensor_offset(gg, ti);
    g_row_bytes = ggml_row_size(ty, g_embd);
    g_to_float = ggml_get_type_traits(ty)->to_float;
    gguf_free(gg);
    if (!g_to_float) return -5;
    int fd = open(path, O_RDONLY);
    struct stat sb;
    if (fd < 0 || fstat(fd, &sb) != 0) return -6;
    g_maplen = (size_t)sb.st_size;
    g_map = mmap(NULL, g_maplen, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (g_map == MAP_FAILED) return -7;
    g_pos = 0;
    return 0;
}

int st_n_vocab(void) { return llama_vocab_n_tokens(g_vocab); }
int st_n_embd(void) { return g_embd; }
int st_pos(void) { return g_pos; }
int st_is_eog(int id) { return llama_vocab_is_eog(g_vocab, id); }

int st_tokenize(const char *text, int len, int *out, int max, int special) {
    return llama_tokenize(g_vocab, text, len, out, max, false, special != 0);
}

int st_piece(int id, char *buf, int n) { return llama_token_to_piece(g_vocab, id, buf, n, 0, true); }

void st_reset(void) {
    llama_memory_clear(llama_get_memory(g_ctx), true);
    g_pos = 0;
}

int st_eval_tokens(const int *ids, int n) {
    for (int at = 0; at < n; at += g_nbatch) {
        int m = n - at < g_nbatch ? n - at : g_nbatch;
        g_tb.n_tokens = m;
        for (int i = 0; i < m; i++) {
            g_tb.token[i] = ids[at + i];
            g_tb.pos[i] = g_pos + i;
            g_tb.n_seq_id[i] = 1;
            g_tb.seq_id[i][0] = 0;
            g_tb.logits[i] = (at + i == n - 1);
        }
        if (llama_decode(g_ctx, g_tb) != 0) return -1;
        g_pos += m;
    }
    return 0;
}

int st_eval_embd(const float *e) {
    g_eb.n_tokens = 1;
    memcpy(g_eb.embd, e, sizeof(float) * (size_t)g_embd);
    g_eb.pos[0] = g_pos;
    g_eb.n_seq_id[0] = 1;
    g_eb.seq_id[0][0] = 0;
    g_eb.logits[0] = 1;
    if (llama_decode(g_ctx, g_eb) != 0) return -1;
    g_pos += 1;
    return 0;
}

float *st_logits(void) { return llama_get_logits_ith(g_ctx, -1); }

int st_embed_row(int id, float *out) {
    if (id < 0 || id >= st_n_vocab()) return -1;
    g_to_float(g_map + g_emb_off + (size_t)id * g_row_bytes, out, g_embd);
    return 0;
}
