# Overnight jobs (the maintainer's orders, 2026-11-16), run in this order

Rules for every job: toolscript is mandatory; run the smallest test that answers the question on GPU and CPU; every GPU job goes through gpu-wait; Claude spend stops near 24% of the 7-day meter; no fable; nothing goes off-machine except the push pane wY6 (the maintainer assigned it); close panes that are done; no job stays stuck (a monitor plus a stall check on each).

1. [x] gemma4 merged cc1e9b7 (tests 3291/3297, 6 fixture-sidecar FAILs as on main).
2. [ ] BLOCKED 27B k sweep: k2 arm OOMs 3x (idle GPU, after pack load 14.76 GB a 4 GB alloc fails with about 9 GB free; FR pack, BARO_SPEC=1, TMAX 16384). Suspect: KV pool x SEQ_CAP=4 in one alloc, or a fragmentation/limit. Needs a one-arm repro with BARO_KV_POOL=1, not more sweeps. Log .work/speed/k2-vs-k1/k2.log.
3. [x] llama.cpp ceiling (3 prompts x 512 tok): plain 31.7, MTP 56.7/58.6/74.7, DFlash2 13.6/12.7/26.5 at ngl52 (ngl99 and ngl60 OOM, so it is not a fair DFlash2 number). Ours: plain 44.4, MTP k2 68.7.
4. [ ] P1 mega z-overlap, then P2 turbo_rows LDS overlap. A winner becomes the default and goes into BASELINE.
5. [x] dflash2 merged 76de5cf (opt-in) D1 to D3; P3 falsifier. D1 GO: tau_break 3.2-3.8, ids 14/20 vs control 18/20, D3 uses a forced-agreement gate; D2 running. D3 PASS: 73.58 vs MTP2 71.99 (1.022), G1 0.99844, R1 3.11 vs 4.97; merge prep (wavewait, run-tests receipt).
6. [x] nerkyor vs forge merged 3e1fec8, S3-S5 done: A plain 42.15, MTP k2 68.03, k4 63.42; B (Forge) 116.11; tasks A 83/120 (llama 89), B 78/120, diff not significant; merge prep.
7. [x] local-jobs: J1 true findings 0/111 (qwythos9b, gemma4e2b), J2 digest defects 6/11, J3 Mojo survivors 1/3; gemma4 J3 cancelled (GPU to dflash2/k sweep). Report docs/lanes/2026-11-16-local-jobs-report.md.
8. [ ] thinkers design (wYC); codex is out of usage, so it switches to OpenRouter free models.
9. [ ] push pane wY6: GitHub + amarbaro.com; close it on DONE.
10. [x] Claude Code via HQ proxy: orfree route PASS (boss check: claude -p --model orfree ran echo, ORFREE_OK), proxy-subs f2715132. codex 429 (out of usage), gemini 401 (the maintainer: Antigravity login, then scripts/proxy_oauth.py import --provider gemini), baro not run; leak fix has no test yet.
11. [ ] BARO_KV_POOL default + needle test at about 150k.
12. [ ] Parked: Kaggle PLE, latent papers firing, ctxd-in-engine.
13. [ ] MTP+DFlash2 hybrid H1/H2 falsifier after D3 (docs/research/2026-11-16-mtp-dflash2-hybrid.md).
