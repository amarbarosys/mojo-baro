> **Caveat added 2026-09-11 (lane-dattn):** every row here timed 200 iterations per target
> (op_bench default), which is shorter than the GPU clock ramp on the short targets. The three
> decode-attention rows re-timed at 5000 iterations with warmup excluded, 10 repeats, spread
> <= 0.9 % (`bench/dattn-confirm.sh`, `.work/dattn-confirm/c4`) read **37.29 / 28.24 / 41.32 us**
> for fa_hd256_qwen35 / fa_hd64_granite / fa_hd128_qwen25, not 55.99 / 34.78 / 67.03. Treat
> every short row below as an upper bound until it is re-timed the same way.

| target | device us/iter | GB/s | % of 960 GB/s | TFLOPS | % of 122.8 TFLOPS | roofline % | cache-cold | top kernel |
|---|---|---|---|---|---|---|---|---|
| mmvq_q4K_qwen25_ffn_up | 66.90 | 572 | 59.6 | 2.03 | 1.7 | 59.6 | yes | mul_mat_vec_q |
| mmvq_q4K_27b_ffn_up | 76.34 | 658 | 68.5 | 2.33 | 1.9 | 68.5 | yes | mul_mat_vec_q |
| mmvq_q4K_spark_ffn_up | 39.05 | 379 | 39.5 | 1.34 | 1.1 | 39.5 | yes | mul_mat_vec_q |
| mmvq_q6K_qwen25_ffn_down | 95.71 | 583 | 60.7 | 1.42 | 1.2 | 60.7 | yes | mul_mat_vec_q |
| mmvq_q6K_lmhead_248k | 940.53 | 888 | 92.5 | 2.16 | 1.8 | 92.5 | yes | mul_mat_vec_q |
| mmvq_q8_0_qwythos_ffn_up | 81.71 | 655 | 68.3 | 1.23 | 1.0 | 68.3 | yes | mul_mat_vec_q |
| mmvf_bf16_granite_ffn_up | 76.36 | 550 | 57.3 | 0.55 | 0.4 | 57.3 | yes | mul_mat_vec_f |
| mmvf_bf16_granite_lmhead | 592.41 | 868 | 90.4 | 0.87 | 0.7 | 90.4 | yes | mul_mat_vec_f |
| mmvf_bf16_qwythos_ffn_up | 141.81 | 710 | 74.0 | 0.71 | 0.6 | 74.0 | yes | mul_mat_vec_f |
| mmq_q4K_qwen25_ffn_up | 1058.97 | 80 | 8.3 | 65.65 | 53.5 | 53.5 | yes | mul_mat_q |
| mmq_q4K_27b_ffn_up | 1340.87 | 72 | 7.5 | 68.07 | 55.4 | 55.4 | yes | mul_mat_q |
| gemm_bf16_granite_ffn_up | 507.74 | 126 | 13.1 | 42.30 | 34.4 | 34.4 | yes | Cijk (rocBLAS Tensile GEMM) |
| gemm_bf16_qwythos_ffn_up | 1273.71 | 105 | 11.0 | 40.46 | 33.0 | 33.0 | yes | Cijk (rocBLAS Tensile GEMM) |
| fa_hd256_qwen35_decode | 55.99 | 300 | 31.3 | 1.20 | 1.0 | 31.3 | yes | flash_attn_ext_vec |
| fa_hd256_qwen35_prompt | 311.32 | 61 | 6.3 | 13.80 | 11.2 | 11.2 | NO | flash_attn_tile |
| fa_hd64_granite_decode | 34.78 | 242 | 25.2 | 1.21 | 1.0 | 25.2 | yes | flash_attn_ext_vec |
| fa_hd64_granite_prompt | 352.38 | 33 | 3.4 | 7.62 | 6.2 | 6.2 | NO | flash_attn_tile |
| fa_hd128_qwen25_decode | 67.03 | 126 | 13.1 | 0.88 | 0.7 | 13.1 | yes | flash_attn_ext_vec |
| rmsnorm_4096_decode | 6.57 | 5 | 0.5 | 0.00 | 0.0 | 0.5 | NO | rms_norm_f32 |
| rmsnorm_4096_prompt | 32.61 | 515 | 53.6 | 0.00 | 0.0 | 53.6 | yes | rms_norm_f32 |
