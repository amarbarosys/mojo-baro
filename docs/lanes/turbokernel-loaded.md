# Lane turbokernel: mandatory skills and protocols, as loaded on 2026-11-02

One line per item the brief and the maintainer's mid-turn message name. "loaded" = Skill tool in this session; "read" = file read in full.

| item | kind | state |
|---|---|---|
| inference-kernels | skill | loaded (Skill tool) |
| wavewait | skill | loaded (Skill tool) |
| mojo-syntax | skill | loaded (Skill tool) |
| mojo-gpu-fundamentals | skill | loaded (Skill tool) |
| gate-authoring | skill | loaded (Skill tool) |
| mojo-nightly-lane-builder | skill | loaded (Skill tool) |
| arch-port | skill | not loaded: qwen35 is an existing arch, no new shape appeared (load if one does) |
| prefix-cache-discipline, memory-optimization | skills | loaded at session start by session_skills.sh |
| corewait | skill | read `~/Brain/Skills/own/corewait/SKILL.md` head after the maintainer's note; for CPU-side hot spots |
| bench/PROTOCOL-RULES.md (P1 to P24) | protocol | read in full |
| bench/coldcache-protocol.md | protocol | read in full |
| ~/Brain/protocols/README.md | protocol index | read in full |
| ~/Brain/protocols/gpu-and-rocm/README.md | protocol | read in full |
| ~/Brain/protocols/mojo-performance/README.md | protocol | read in full |
| ~/Brain/protocols/model-bring-up/README.md | protocol | read in full |
| ~/Brain/protocols/debugging/README.md | protocol | read: method and the recurring-failures table |
| docs/BASELINE.md | repo truth | read: toolchain, megakernel decode path, K-quant megakernel rows (623 to 647) |
| docs/KERNELS.md | repo truth | read in full |
| exchange/lane-MEGA-BARRIER-report.md | lane report | read in full (the 87.9 vs 98.3 receipt) |
| memories mega-kq-lane-landed, rmsnorm-fold-closed, megakernel-lottery-fingerprint, moe-r60-launch-folds-landed | memory | read in full |
| .work/turbo27/{new,ctl}/arm.txt, agree.log | brief receipts | read: new = this model vs llama.cpp 93.75% median, ctl = OrcaSAQ 96.9% |
| turbobake lane state | lane | read: verify PASS 64/64 on the bake at c71dbdc, no report in docs/lanes yet |
