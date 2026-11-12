# Report: rdna4-isa and rdna4-port skills (2026-11-12)

Brief: `docs/lanes/2026-11-12-rdna4-skill-brief.md`. Model: Sonnet, effort high.

## Skills loaded

`~/Brain/Skills/own/rdna3-isa/SKILL.md` (template and gfx11 side). Nothing else loaded.

## Sources read, with line ranges

`~/Projects/AMD/docs/rdna4-isa.txt` (43,409 lines, footer `of 697`). Read in full (every line): 393 to 1092 of the
whitespace-squeezed copy = chapters 1 and 2, chapter 3 up to 3.4.1 (state, EXEC, SGPR/VGPR rules, dynamic VGPR, LDS, wave
registers), and in the raw file 5210 to 5659 (7.10 to 7.12: pseudo-scalar ops, VGPR indexing, WMMA tables, hazards, lane maps,
sparsity). Read by section, not every line: 3.4.3 MODE, 3.4.6 to 3.4.8, 3.4.11 time; 4.1.1 cache controls (Tables 13 to 16);
5.2 and 5.3 clauses; 5.4 messages (Tables 21, 22); 5.5 branching; 5.6 barriers (Tables 24, 25); 5.7 data dependency and
counters (all of it) plus 5.7.2; 5.8 `s_delay_alu`; 7.2.2.2 source restrictions, 7.2.7, 7.2.8; 7.6 conversions; 7.7 packed
math header; 7.8 VOPD (all); 7.9 DPP (permutation table and Table 38); 8.1 SMEM encoding; 11.1, 11.2, 11.5, 11.6 (flat, global,
block, transpose loads); 12.1, 12.5, 12.5.2 (LDS). Opcode semantics read through `isa-op.sh`: V_DOT4_I32_IU8, V_DOT2_F32_F16,
V_PERMLANE16_B32, V_PERMLANE16_VAR_B32, S_SETPRIO, S_SENDMSG_RTN_B64, S_WAITCNT, S_WAIT_LOADCNT, V_WMMA_I32_16X16X32_IU4.
NOT read: chapter 6 (SALU, only the opcode names), chapter 9 buffer details beyond headings, chapter 13 atomics, chapter 15
microcode formats, chapter 16 body (used only as an opcode name list: 1393 names). Skipped on purpose: chapters 10 and 14, ray tracing,
interpolation, export. "Compute chapters in full" was NOT achieved for chapters 4 to 9 and 11 to 13; the sections above are
the ones the two skills depend on.

`matrix_calculator.py`: class `InstCalcGfx12` lines 6797 to 7400 (all lane, register and compression-index functions) and the
equation printers 7499 to 7580 and 7361 to 7470; `InstCalcGfx11` output and input lane functions (6234 to 6330); instruction
table `'rdna4'` (2435 to ~2900, opcode, in/out type, K, cycles, sparse) and `'rdna3'` cycles. README only grep (gfx12 lines 18, 91, 1290 to 1309). Never executed.

Stdlib lead: `mma_amd_rdna.mojo` (modular-nightly checkout `c0bfec57fa`) lines 330 to 640, read only.

## Files written

Brain (committed with pathspec in `~/Brain`):
- `Skills/own/rdna4-isa/SKILL.md` (241 lines), `scripts/isa-op.sh`, `scripts/manual-split.sh`
- `Skills/own/rdna4-isa/examples/`: `smoke.s`, `deltas.sh`, `deltas.txt`, `deltas.out`, `llc-probe.sh`, `llc-probe.tsv`, `llc-probe.out`
- `Skills/own/rdna4-port/SKILL.md` (112 lines)
- `research/rdna4-isa/` 301 files (300 section notes + `_index.md`); every note links back to `Skills/own/rdna4-isa/SKILL` (0 notes without the link)
- `Skills/own/gpu-kernels/SKILL.md`: two rows in the Route table; `Skills/active.tsv`: two rows (skill-new)
- Symlinks `.claude/skills/rdna4-isa` and `.claude/skills/rdna4-port` in mojo-baro (gitignored, not committed); `skilldex` run, `skilldex --check` PASS.
mojo-baro: this report only (pathspec commit).

## llvm-mc and llc results (ROCm 7.2.4, LLVM 22.0.0git, CPU only)

- `examples/smoke.s` assembles clean with `-mcpu=gfx1201` and `-mcpu=gfx1200` (WMMA f16/bf16/iu8/iu4/fp8, SWMMAC, dot4, transpose loads,
  `s_wait_*`, barrier signal/wait, permlane, `s_sendmsg_rtn_b64` REALTIME, `s_setprio`, `s_wait_alu`).
- `examples/deltas.out`: 78 instruction forms assembled for gfx1100, gfx1200, gfx1201. gfx1200 and gfx1201 agree on every row.
- `examples/llc-probe.out`: every `arch.mojo` intrinsic (`sdot4`, `sudot4`, `perm`, `s.setprio`, `s.sleep`, `s.sendmsg.rtn(131)`,
  `readlane`) selects on gfx1100 and gfx1201. WMMA f16 with `<8 x half>` selects on gfx1201 only, `<16 x half>` on gfx1100 only; iu8
  `<2 x i32>` gfx12 only, `<4 x i32>` gfx11 only; fp8 gfx12 only. The failing combinations abort in isel ("Cannot select").
- `scripts/isa-op.sh` and `manual-split.sh` run clean against the RDNA4 text.

## Facts that answer the brief's open items

- sudot4: `llvm.amdgcn.sudot4` and `sdot4` select `v_dot4_i32_iu8` on gfx1201 (same opcode 22 as gfx11). The plan's "may be gfx11-only (UNSURE)" is falsified for this LLVM (not for Mojo's own LLVM, UNSURE).
- s_memrealtime: absent from both manuals, rejected on gfx1100, gfx1200, gfx1201. `s_sendmsg_rtn_b64` REALTIME is message 0x83 = 131 (ISA Table 22), accepted on both.
- s_setprio: same opcode 53, unchanged semantics, accepted on both.
- `S_WAITCNT` (SOPP 9) on gfx12 is `S_WAIT_IDLE`, operand ignored (ISA line 15516). LLVM still assembles `s_waitcnt vmcnt(0)`. A legacy `s_waitcnt` in gfx12 inline asm over-waits. mojo-baro has no inline asm (grep).
- Global/flat immediate offset is 24-bit signed on gfx12 (13-bit on gfx11); SMEM 24-bit.
- RDNA4 hazards: new `s_wait_alu` SGPR/VCC rules (ISA s5.7), WMMA table (s7.12.1) states the C-chain as a stall rule.
- Dynamic VGPR mode (`s_alloc_vgpr`) is new.
- Rocminfo on this card: 96 CUs, L3 98304 KB, 32 waves/CU. MEGA_G = 96 equals the CU count; `coldcache-protocol.md:25` hardcodes 96 MB.

## CONFLICTS

1. Manual vs assembler: the RDNA4 manual lists `GLOBAL_LOAD_BLOCK`, `GLOBAL_STORE_BLOCK`, `SCRATCH_*_BLOCK` (s11.5); LLVM 22 rejects `global_load_block` for gfx1200 and gfx1201. Both written in `rdna4-isa`; not picked.
2. ISA vs calculator on WMMA lane maps: NONE. Every dense row (wave32 and wave64, f16/bf16, 8-bit, iu4 K=16, iu4 K=32, C/D f32, C/D f16) was compared field by field and agrees; the closed forms in `rdna4-isa` were also checked for inverse consistency (awk, 0 mismatches).

## UNSURE (leads, tagged in the skills)

- Whether Mojo's own LLVM selects each intrinsic for gfx1200/gfx1201 (only ROCm's llc probed); `_is_amd_rdna4()` true for both targets.
- Real CU counts, IC sizes (32/64 MB), VRAM sizes and ROCm minimum (>= 6.4) for RDNA4 cards: web-sourced, not in the ISA.
- The D-to-A/B no-shuffle chaining (P*V) with contiguous-per-half partner operand: derived from the lane tables, no run.
- LDS bank-conflict behaviour with lanes i and i+16 no longer reading the same address; gfx11 pads/strides unverified on gfx12.
- Whether the compiler drops the unused half of a 16-wide fragment load on gfx12 (cost of the stdlib slice).
- `mmq_scale_slot` on gfx12 equals `r` (derived, no kernel run).
- SWMMAC lane maps copied from the calculator only (no ISA table for sparse A/B lane order beyond the VGPR counts); no mojo-baro use.
- The `cycles` unit in the calculator table (used only as an f16:iu8 ratio of 2:1 on gfx12).
- The gfx1100-side tools (`isa-hot`, `isa-loops`, `isa-diff`, `rowify-check`, `ww.sh`) on gfx12 ISA text.

## Facts contradicting today's Brain note (`2026-11-12-rdna4-wmma-layout.md`)

None on layout. Two additions to it: (a) the note says "iu8: contiguous halves (lane = 16*(k/8)+i)": confirmed. (b) the note says "gfx12 transpose loads" matter for order: confirmed, `global_load_tr_b64/b128` exist (ISA s11.6.2) and there is no `ds_load_tr` (manual and assembler). One remark about the retraction: the correction (stdlib not a bug) is consistent with the ISA tables, and `rdna4-isa` adds the derived consequence that D tiles chain into A or B without shuffles when the partner uses the stdlib convention (UNSURE, no run).
Also in `docs/rules.md` line 106 still says three sources "agreed on a layout the ISA contradicts": true as a layout statement (the contiguous slice is not the ISA order), not as a correctness claim; lines 107 and 108 already carry the correction.

## Not done

- No `tools/rdna4-probe.sh`, no gfx1201 build in `bench/preflight.sh`, no kernel change (plan phases 1 to 3, out of this brief).
- No commit of the two symlinks (`.claude/` is gitignored).
