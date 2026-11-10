// mtmd-oracle: the reference arm for the vision encoder (P2). Feeds one raw RGB image through
// llama.cpp's libmtmd and writes the final image embeddings plus named intermediate tensors.
//
//   mtmd-oracle TEXT_MODEL.gguf MMPROJ.gguf IMAGE.rgb NX NY OUTDIR [--gpu]
//
// IMAGE.rgb is NX*NY*3 bytes, row-major, RGB, no header (tools/vision-ref.py writes it), so both
// arms start from the same bytes and no image decoder sits between them. OUTDIR gets
//   embd.f32          final embeddings, [n_tokens, n_embd] row-major float32
//   <name>.f32        every f32 tensor whose graph name starts with one of WANT below
//   manifest.tsv      name, ne0, ne1, ne2, ne3, bytes
// Flash attention is off unless ORACLE_FA=on: clip's FA path casts K and V to f16, which put 1.6e-2
// relative error into attn_out-0 and dropped layer_out-26 cosine from 0.9997 to 0.9907.
// The text model is loaded vocab-only: libmtmd wants it for the marker tokens, never for weights.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "mtmd.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static const char * WANT[] = {
    "patch_bias", "inp_pos_emb", "ln1-0", "Qcur-0", "Qcur_rope-0", "Kcur_rope-0", "attn_out-0",
    "ffn_inp-0", "ffn_out-0", "layer_out-0", "layer_out-1", "layer_out-13", "layer_out-26",
};
static std::string g_out;
static FILE * g_manifest = nullptr;

static void die(const char * step, const std::string & why) {
    fprintf(stderr, "FAIL %s: %s\n", step, why.c_str());
    exit(1);
}

static bool wanted(const char * name) {
    for (const char * w : WANT) {
        if (strcmp(name, w) == 0) return true;
    }
    return false;
}

static bool on_eval(struct ggml_tensor * t, bool ask, void *) {
    if (ask) return wanted(t->name);
    if (!wanted(t->name) || t->type != GGML_TYPE_F32) return true;
    struct ggml_tensor * src = t;
    std::vector<float> buf(ggml_nelements(src));
    if (!ggml_is_contiguous(src)) {
        fprintf(stderr, "skip %s: not contiguous\n", t->name);
        return true;
    }
    ggml_backend_tensor_get(src, buf.data(), 0, ggml_nbytes(src));
    std::ofstream f(g_out + "/" + t->name + ".f32", std::ios::binary);
    f.write((const char *) buf.data(), buf.size() * sizeof(float));
    fprintf(g_manifest, "%s\t%lld\t%lld\t%lld\t%lld\t%zu\n", t->name, (long long) t->ne[0], (long long) t->ne[1],
            (long long) t->ne[2], (long long) t->ne[3], buf.size() * sizeof(float));
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 7) die("usage", "mtmd-oracle TEXT_MODEL.gguf MMPROJ.gguf IMAGE.rgb NX NY OUTDIR [--gpu]");
    const char * model_path = argv[1];
    const char * mmproj = argv[2];
    const char * img_path = argv[3];
    const uint32_t nx = atoi(argv[4]), ny = atoi(argv[5]);
    g_out = argv[6];
    const bool gpu = argc > 7 && strcmp(argv[7], "--gpu") == 0;

    std::ifstream fi(img_path, std::ios::binary);
    std::vector<unsigned char> rgb((std::istreambuf_iterator<char>(fi)), std::istreambuf_iterator<char>());
    if (rgb.size() != (size_t) nx * ny * 3) die("image", "expected " + std::to_string((size_t) nx * ny * 3) + " bytes, got " + std::to_string(rgb.size()));

    g_manifest = fopen((g_out + "/manifest.tsv").c_str(), "w");
    if (!g_manifest) die("outdir", "cannot write " + g_out + "/manifest.tsv");

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) die("model", std::string("cannot load ") + model_path);

    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = gpu;
    cp.n_threads = 8;
    cp.flash_attn_type = (getenv("ORACLE_FA") && strcmp(getenv("ORACLE_FA"), "on") == 0) ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.cb_eval = on_eval;
    cp.cb_eval_user_data = nullptr;
    mtmd_context * ctx = mtmd_init_from_file(mmproj, model, cp);
    if (!ctx) die("mmproj", std::string("cannot load ") + mmproj);

    mtmd_bitmap * bmp = mtmd_bitmap_init(nx, ny, rgb.data());
    std::string prompt = mtmd_default_marker();
    mtmd_input_text text = { prompt.c_str(), prompt.size(), false, true };
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const mtmd_bitmap * bmps[1] = { bmp };
    if (mtmd_tokenize(ctx, chunks, &text, bmps, 1) != 0) die("tokenize", "mtmd_tokenize failed");

    bool done = false;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        const mtmd_input_chunk * ch = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(ch) != MTMD_INPUT_CHUNK_TYPE_IMAGE) continue;
        const size_t n_tok = mtmd_input_chunk_get_n_tokens(ch);
        if (mtmd_encode_chunk(ctx, ch) != 0) die("encode", "mtmd_encode_chunk failed");
        int n_embd = 0; // a vocab-only text model reports 0, so the width comes from the projector file
        {
            gguf_init_params gp = { true, nullptr };
            gguf_context * g = gguf_init_from_file(mmproj, gp);
            const int64_t k = g ? gguf_find_key(g, "clip.vision.projection_dim") : -1;
            if (k < 0) die("mmproj", "no clip.vision.projection_dim");
            n_embd = (int) gguf_get_val_u32(g, k);
            gguf_free(g);
        }
        const float * e = mtmd_get_output_embd(ctx);
        std::ofstream f(g_out + "/embd.f32", std::ios::binary);
        f.write((const char *) e, n_tok * n_embd * sizeof(float));
        fprintf(g_manifest, "embd\t%d\t%zu\t1\t1\t%zu\n", n_embd, n_tok, n_tok * n_embd * sizeof(float));
        printf("image tokens %zu  n_embd %d  gpu %d\n", n_tok, n_embd, gpu ? 1 : 0);
        done = true;
    }
    fclose(g_manifest);
    if (!done) die("chunks", "no image chunk came out of mtmd_tokenize");
    return 0;
}
