# Provenance: Freivalds integer-GEMM check (`kernels/freivalds.mojo`)

Clean-room build (skill `clean-room-build`, the maintainer's rule 2026-10-12: the idea, never the code).

## The idea, in our words

To verify that a computed product `C` equals `A x B` without recomputing the GEMM, pick a random
vector `r`, and compare `C r` with `A (B r)`. Both sides are three matrix-vector products, so the
check reads `A`, `B` and `C` once each (O(mn + nk + mk)) instead of doing O(mnk) work. If `C` is
correct the two sides agree for every `r`. If `C` is wrong, `(C - AB) r` is a nonzero matrix times
a random vector, and it vanishes only when `r` happens to lie in the kernel of a nonzero row of
`C - AB`. Over a field of size `p`, that happens with probability at most `1/p` per nonzero row.

Our design works over the prime field `p = 2^61 - 1`. Every entry of `A`, `B` (int8, int16, int32
by parameter) and `C` (int32) is an exact integer of magnitude below `p`, so a wrong entry of `C`
differs from the true value by a nonzero integer of magnitude below `2^32 < p`, which cannot vanish
modulo `p`. One round therefore detects any corruption of `C` with probability at least
`1 - 1/p ≈ 1 - 2^-61`. The coefficients `r_j` come from the OS CSPRNG (`/dev/urandom`), reduced
into `[0, p)`; a clock seed would let a corrupted `C` be crafted to pass. The dot products split each
61-bit residue into three 21-bit limbs so every partial sum stays an exact int64; limbs recombine
with Mersenne shifts (`2^61 ≡ 1`). One generic row-dot kernel body is launched three times
(`u = B r`, `v = A u`, `w = C r`) and a device compare writes a flag, so the host reads 16 bytes.

## Primary sources

- R. Freivalds, "Probabilistic Machines Can Use Less Running Time", IFIP Congress 1977, pp. 839-842
  (the randomized `A(Br) = Cr` check and its one-sided error bound).
- J. T. Schwartz, "Fast probabilistic algorithms for verification of polynomial identities", JACM
  27(4), 1980; R. Zippel, "Probabilistic algorithms for sparse polynomials", EUROSAM 1979 (the
  Schwartz-Zippel lemma: a nonzero degree-`d` polynomial over a field vanishes at a random point of a
  set `S` with probability at most `d/|S|`; here `d = 1`, `|S| = p`).
- `2^61 - 1` is a Mersenne prime (Pervushin 1883); reduction by `(x & p) + (x >> 61)` is the
  standard Mersenne trick.

## Where we saw it

An ANN post for a proof-of-useful-work chain (notes: `~/Brain/research/2026-10-12-vearl-pouw.md`)
verifies int8 tensor products with 24 Freivalds rounds for a `2^-24` error bound. We read a static
audit summary of that project, never its source files, and this kernel was written with no external
source open. `clean-room-check` against the imported audit tree is run before merge and pasted into
`exchange/lane-freivalds-report.md`.

## Beat-the-original target (frozen before the build, with `exchange/freivalds-prereg-2026-10-12.md`)

| property | theirs (as reported) | ours, target |
|---|---|---|
| rounds per verified product | 24 | 1 |
| error bound | `2^-24` | `<= 2^-61` (`1/p`) |
| randomness | per-round PRNG | OS CSPRNG, 61-bit residues |
| cost at 4096^3 | not reported | `<= 1.5x` the HBM read of `A + B + C`, `< 5%` of the int8 GEMM |

Patent note: Freivalds' algorithm is a 1977 published result; no patent question known.
