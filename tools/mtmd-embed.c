// mtmd-embed: encode one image with llama.cpp's vision projector (libmtmd) and write the
// projected token embeddings plus their mrope decoder positions for mojo-baro's engine.
//   mtmd-embed MMPROJ.gguf TEXT.gguf IMAGE OUT.bemb [--gpu] [--max-tokens N]
// OUT.bemb (little endian): "BEMB" u32 version=1, u32 n_tokens, u32 n_embd, u32 n_pos,
//   n_tokens x {i32 t, i32 x, i32 y} (pos_0 = 0), then n_tokens x n_embd f32.
// stdout: one JSON line {"n_tokens","n_embd","n_pos","nx","ny"}. The text gguf is opened
// vocab-only (header + tokenizer, no weights). Encode runs on the CPU unless --gpu.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "llama.h"
#include "ggml-backend.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "gguf.h"

static void quiet(enum ggml_log_level level, const char * text, void * ud) {
    (void)ud;
    if (level >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: mtmd-embed MMPROJ.gguf TEXT.gguf IMAGE OUT.bemb [--gpu] [--max-tokens N]\n"); return 2; }
    const char * mmproj = argv[1], * text_gguf = argv[2], * image = argv[3], * out_path = argv[4];
    int use_gpu = 0, max_tokens = 0;
    for (int a = 5; a < argc; a++) {
        if (strcmp(argv[a], "--gpu") == 0) use_gpu = 1;
        else if (strcmp(argv[a], "--max-tokens") == 0 && a + 1 < argc) max_tokens = atoi(argv[++a]);
        else { fprintf(stderr, "FAIL unknown arg %s\n", argv[a]); return 2; }
    }
    // the projector's output width comes from the mmproj header: the text model is vocab-only here
    struct gguf_init_params gp = { true, NULL };
    struct gguf_context * g = gguf_init_from_file(mmproj, gp);
    if (!g) { fprintf(stderr, "FAIL read mmproj header: %s\n", mmproj); return 1; }
    int64_t kid = gguf_find_key(g, "clip.vision.projection_dim");
    if (kid < 0) { fprintf(stderr, "FAIL mmproj lacks clip.vision.projection_dim\n"); gguf_free(g); return 1; }
    uint32_t n_embd = gguf_get_val_u32(g, kid);
    gguf_free(g);
    llama_log_set(quiet, NULL);
    mtmd_helper_log_set(quiet, NULL);
    ggml_backend_load_all();
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    struct llama_model * model = llama_model_load_from_file(text_gguf, mp);
    if (!model) { fprintf(stderr, "FAIL load text vocab: %s\n", text_gguf); return 1; }
    struct mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = use_gpu;
    cp.print_timings = false;
    cp.n_threads = 8;
    cp.warmup = false;
    if (max_tokens > 0) cp.image_max_tokens = max_tokens;
    mtmd_context * ctx = mtmd_init_from_file(mmproj, model, cp);
    if (!ctx) { fprintf(stderr, "FAIL load mmproj: %s\n", mmproj); return 1; }
    if (!mtmd_support_vision(ctx)) { fprintf(stderr, "FAIL mmproj has no vision encoder\n"); return 1; }
    struct mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_file(ctx, image, false, mtmd_helper_init_opt_default());
    if (!bw.bitmap) { fprintf(stderr, "FAIL read image: %s\n", image); return 1; }
    const mtmd_bitmap * bitmaps[1] = { bw.bitmap };
    mtmd_input_text text = { mtmd_default_marker(), strlen(mtmd_default_marker()), false, true };
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    if (mtmd_tokenize(ctx, chunks, &text, bitmaps, 1) != 0) { fprintf(stderr, "FAIL mtmd_tokenize\n"); return 1; }
    const mtmd_input_chunk * img = NULL;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_IMAGE) { img = c; break; }
    }
    if (!img) { fprintf(stderr, "FAIL no image chunk\n"); return 1; }
    const mtmd_image_tokens * it = mtmd_input_chunk_get_tokens_image(img);
    uint32_t n_tokens = (uint32_t)mtmd_image_tokens_get_n_tokens(it);
    uint32_t n_pos = (uint32_t)mtmd_image_tokens_get_n_pos(it);
    uint32_t nx = (uint32_t)mtmd_image_tokens_get_nx(it), ny = (uint32_t)mtmd_image_tokens_get_ny(it);
    if (mtmd_encode_chunk(ctx, img) != 0) { fprintf(stderr, "FAIL mtmd_encode_chunk\n"); return 1; }
    const float * embd = mtmd_get_output_embd(ctx);
    if (!embd) { fprintf(stderr, "FAIL no output embeddings\n"); return 1; }
    FILE * f = fopen(out_path, "wb");
    if (!f) { fprintf(stderr, "FAIL open %s\n", out_path); return 1; }
    uint32_t ver = 1;
    fwrite("BEMB", 1, 4, f); fwrite(&ver, 4, 1, f); fwrite(&n_tokens, 4, 1, f); fwrite(&n_embd, 4, 1, f); fwrite(&n_pos, 4, 1, f);
    for (uint32_t i = 0; i < n_tokens; i++) {
        struct mtmd_decoder_pos p = mtmd_image_tokens_get_decoder_pos(it, 0, i);
        int32_t txy[3] = { (int32_t)p.t, (int32_t)p.x, (int32_t)p.y };
        fwrite(txy, 4, 3, f);
    }
    fwrite(embd, sizeof(float), (size_t)n_tokens * n_embd, f);
    if (ferror(f) || fclose(f) != 0) { fprintf(stderr, "FAIL write %s\n", out_path); return 1; }
    printf("{\"n_tokens\":%u,\"n_embd\":%u,\"n_pos\":%u,\"nx\":%u,\"ny\":%u,\"mrope\":%s}\n", n_tokens, n_embd, n_pos, nx, ny, mtmd_decode_use_mrope(ctx) ? "true" : "false");
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bw.bitmap);
    mtmd_free(ctx);
    llama_model_free(model);
    return 0;
}
