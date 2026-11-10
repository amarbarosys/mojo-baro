"""Host reference sampler gate (bench/chat-protocol.md C3, P-I1..P-I4).

No GPU needed -- serve/sample_ref.mojo is pure host CPU. Build:
  ./.venv/bin/mojo build kernels/test_sample_ref.mojo -I kernels -I serve -o .work/test_sample_ref
"""
from std.math import sqrt

from sample_ref import philox4x32, sample_row_ref, sample_probs_ref, spec_accept_ref, apply_penalties, is_valid

comptime FMAX = Float32(3.4028234663852886e38)


def make_nan() -> Float32:
    var b: UInt32 = 0x7FC00000
    return b.cast[DType.float32]()


def hexu(v: UInt32) -> String:
    var digits = "0123456789abcdef"
    var s = String("0x")
    for shift in range(7, -1, -1):
        s += digits[byte = Int((v >> UInt32(shift * 4)) & 0xF)]
    return s


def check_philox_kat(mut fails: Int):
    var r0 = philox4x32(SIMD[DType.uint32, 4](0, 0, 0, 0), 0, 0)
    var want0 = SIMD[DType.uint32, 4](0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)
    var r1 = philox4x32(SIMD[DType.uint32, 4](0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF), UInt32(0xFFFFFFFF), UInt32(0xFFFFFFFF))
    var want1 = SIMD[DType.uint32, 4](0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)
    var r2 = philox4x32(SIMD[DType.uint32, 4](0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), UInt32(0xA4093822), UInt32(0x299F31D0))
    var want2 = SIMD[DType.uint32, 4](0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1)
    var ok = (r0 == want0) and (r1 == want1) and (r2 == want2)
    if ok:
        print("PASS philox4x32-10 KAT (zero, all-ones, pi vectors)")
    else:
        print("FAIL philox4x32-10 KAT: r0", hexu(r0[0]), hexu(r0[1]), hexu(r0[2]), hexu(r0[3]))
        fails += 1


def check_temperature_zero(mut fails: Int):
    print("== temperature 0: sample_row_ref must equal greedy argmax, ties by lowest index")
    var cases = List[List[Float32]]()
    var c1 = List[Float32]()
    for i in range(20):
        c1.append(Float32(i) * Float32(0.37).__mul__(1) - Float32(3))
    cases.append(c1^)
    var c2 = List[Float32]()  # a tie at the max
    for i in range(20):
        c2.append(Float32(5) if i == 3 or i == 11 else Float32(i))
    cases.append(c2^)
    var c3 = List[Float32]()  # NaN sprinkled, real max elsewhere
    var nanv = make_nan()
    for i in range(20):
        c3.append(nanv if i % 3 == 0 else Float32(i) - Float32(10))
    cases.append(c3^)
    var c4 = List[Float32]()  # all -inf: no valid token
    for i in range(20):
        c4.append(-FMAX * Float32(2))
    cases.append(c4^)

    var local_fails = 0
    for ci in range(len(cases)):
        var row = cases[ci].copy()
        var best_i = -1
        var best_v = -FMAX
        for i in range(len(row)):
            var v = row[i]
            if v >= -FMAX and v <= FMAX and v > best_v:
                best_v = v
                best_i = i
        var got = sample_row_ref(row, Float32(0), 0, Float32(1.0), Float32(0.0), UInt64(0), UInt64(0), ci)
        var want_prob = Float32(1) if best_i >= 0 else Float32(0)
        if got[0] != best_i or got[1] != want_prob:
            print("  FAIL case", ci, "got", got[0], got[1], "want", best_i, want_prob)
            local_fails += 1
    if local_fails == 0:
        print("  PASS 4/4 temperature-0 cases (plain, tie, NaN-sprinkled, all-invalid)")
    else:
        fails += local_fails


def wilson_hilferty_critical(df: Int) -> Float64:
    # Chi-square upper-tail critical value at p = 0.001 (z = 3.090232306167813
    # for the one-sided 0.999 quantile of the standard normal), Wilson-Hilferty
    # cube-root normal approximation. Reference-grade, not exact, but the
    # gates below run with comfortable margin (matching KSAMP's own numbers).
    var d = Float64(df)
    var z = Float64(3.090232306167813)
    var t = 1.0 - 2.0 / (9.0 * d) + z * sqrt(2.0 / (9.0 * d))
    return d * t * t * t


def chi2_pooled(counts: List[Int], expected: List[Float64]) -> Tuple[Float64, Int]:
    # Pool adjacent bins (in index order) until every bin's expected count
    # is >= 5, then return (chi2, df = bins - 1).
    var pc = List[Int]()
    var pe = List[Float64]()
    var acc_c = 0
    var acc_e: Float64 = 0
    for i in range(len(counts)):
        acc_c += counts[i]
        acc_e += expected[i]
        if acc_e >= 5.0:
            pc.append(acc_c)
            pe.append(acc_e)
            acc_c = 0
            acc_e = 0
    if acc_e > 0:
        if len(pe) > 0:
            pc[len(pc) - 1] += acc_c
            pe[len(pe) - 1] += acc_e
        else:
            pc.append(acc_c)
            pe.append(acc_e)
    var chi2: Float64 = 0
    for i in range(len(pc)):
        var d = Float64(pc[i]) - pe[i]
        chi2 += d * d / pe[i]
    return (chi2, max(len(pc) - 1, 1))


def draw_distribution_check(
    label: String, logits: List[Float32], temperature: Float32, top_k: Int, top_p: Float32, min_p: Float32,
    n_draws: Int, seed: UInt64, mut fails: Int,
):
    var exact = sample_probs_ref(logits, temperature, top_k, top_p, min_p)
    var n = len(logits)
    var counts = List[Int](unsafe_uninit_length=n)
    for i in range(n):
        counts[i] = 0
    var outside = 0
    for draw in range(n_draws):
        var got = sample_row_ref(logits, temperature, top_k, top_p, min_p, seed, UInt64(draw), 0)
        if got[0] < 0 or exact[got[0]] <= 0:
            outside += 1
        else:
            counts[got[0]] += 1
    var expected = List[Float64](unsafe_uninit_length=n)
    for i in range(n):
        expected[i] = Float64(exact[i]) * Float64(n_draws)
    var chi2df = chi2_pooled(counts, expected)
    var crit = wilson_hilferty_critical(chi2df[1])
    if chi2df[0] <= crit and outside == 0:
        print("  PASS", label, "chi2", chi2df[0], "df", chi2df[1], "critical", crit, "outside", outside)
    else:
        print("  FAIL", label, "chi2", chi2df[0], "df", chi2df[1], "critical", crit, "outside", outside)
        fails += 1


def check_distributions(mut fails: Int):
    print("== distribution: 10,000 draws vs exact target (chi-square, p=0.001)")
    var v = 48
    var row = List[Float32](unsafe_uninit_length=v)
    for i in range(v):
        # a peaked-ish row: five boosted tokens plus a broad tail, deterministic.
        var base = Float32(-0.05) * Float32((i - 24) * (i - 24))
        row[i] = base
    row[5] += 6
    row[12] += 5.5
    row[20] += 5
    row[30] += 4.5
    row[40] += 4

    var tie_row = List[Float32](unsafe_uninit_length=v)
    for i in range(v):
        tie_row[i] = Float32(i % 6)

    draw_distribution_check("plain T1 no truncation", row, Float32(1.0), 0, Float32(1.0), Float32(0.0), 10000, UInt64(1), fails)
    draw_distribution_check("T0.7 k16 p0.8", row, Float32(0.7), 16, Float32(0.8), Float32(0.0), 10000, UInt64(2), fails)
    draw_distribution_check("T1.3 k12 p0.9 minp0.05", row, Float32(1.3), 12, Float32(0.9), Float32(0.05), 10000, UInt64(3), fails)
    draw_distribution_check("T0.5 p0.6", row, Float32(0.5), 0, Float32(0.6), Float32(0.0), 10000, UInt64(4), fails)
    draw_distribution_check("ties T1 k10 (cut inside a tie group)", tie_row, Float32(1.0), 10, Float32(1.0), Float32(0.0), 10000, UInt64(5), fails)
    draw_distribution_check("ties T0.9 p0.35 (mass cut inside ties)", tie_row, Float32(0.9), 0, Float32(0.35), Float32(0.0), 10000, UInt64(6), fails)


def check_reproducibility(mut fails: Int):
    print("== reproducibility: same (seed, counter) -> same token")
    var v = 48
    var row = List[Float32](unsafe_uninit_length=v)
    for i in range(v):
        row[i] = Float32(-0.05) * Float32((i - 24) * (i - 24))
    row[5] += 6
    row[12] += 5.5
    var n = 2000
    var same = 0
    for i in range(n):
        var a = sample_row_ref(row, Float32(0.8), 20, Float32(0.9), Float32(0.0), UInt64(42), UInt64(i), 0)
        var b = sample_row_ref(row, Float32(0.8), 20, Float32(0.9), Float32(0.0), UInt64(42), UInt64(i), 0)
        if a[0] == b[0]:
            same += 1
    var different_seed = 0
    for i in range(n):
        var a = sample_row_ref(row, Float32(0.8), 20, Float32(0.9), Float32(0.0), UInt64(42), UInt64(i), 0)
        var b = sample_row_ref(row, Float32(0.8), 20, Float32(0.9), Float32(0.0), UInt64(43), UInt64(i), 0)
        if a[0] != b[0]:
            different_seed += 1
    if same == n and different_seed > n // 2:
        print("  PASS", same, "/", n, "same-seed reproduced; ", different_seed, "/", n, "changed under a different seed")
    else:
        print("  FAIL same-seed", same, "/", n, " different-seed-changed", different_seed, "/", n)
        fails += 1


def check_speculation(mut fails: Int):
    print("== speculation: accept-or-resample reproduces the exact target distribution")
    var v = 48
    var pt_logits = List[Float32](unsafe_uninit_length=v)
    var pd_logits = List[Float32](unsafe_uninit_length=v)
    for i in range(v):
        pt_logits[i] = Float32(-0.05) * Float32((i - 24) * (i - 24))
        pd_logits[i] = Float32(-0.04) * Float32((i - 18) * (i - 18))
    pt_logits[5] += 6
    pt_logits[30] += 4
    pd_logits[10] += 5
    pd_logits[36] += 3

    var pt = sample_probs_ref(pt_logits, Float32(0.9), 24, Float32(0.95), Float32(0.0))
    var pd = sample_probs_ref(pd_logits, Float32(0.9), 24, Float32(0.95), Float32(0.0))

    var n = 10000
    var seed = UInt64(7)
    var accepted = 0
    var counts = List[Int](unsafe_uninit_length=v)
    for i in range(v):
        counts[i] = 0
    for draw in range(n):
        # The drafted token is itself drawn from p_d, matching KSAMP's own
        # "mismatched draft" setup (a draft head that samples its own row).
        var dtok = sample_row_ref(pd_logits, Float32(0.9), 24, Float32(0.95), Float32(0.0), seed + UInt64(1_000_000), UInt64(draw), 0)
        var acc = spec_accept_ref(pt, pd, dtok[0], seed, UInt64(draw), 0)
        if acc[1]:
            accepted += 1
        if acc[0] >= 0:
            counts[acc[0]] += 1

    var sum_min: Float64 = 0
    for i in range(v):
        sum_min += Float64(min(pt[i], pd[i]))
    var rate = Float64(accepted) / Float64(n)
    var sigma = sqrt(sum_min * (1.0 - sum_min) / Float64(n))
    var within = abs(rate - sum_min) <= 4.0 * sigma

    var expected = List[Float64](unsafe_uninit_length=v)
    for i in range(v):
        expected[i] = Float64(pt[i]) * Float64(n)
    var chi2df = chi2_pooled(counts, expected)
    var crit = wilson_hilferty_critical(chi2df[1])

    if within and chi2df[0] <= crit:
        print("  PASS accept rate", rate, "vs sum min(pt,pd)", sum_min, "(", sigma, "sigma band); accept-or-resample vs p_t chi2", chi2df[0], "df", chi2df[1], "critical", crit)
    else:
        print("  FAIL accept rate", rate, "vs", sum_min, "sigma", sigma, "; chi2", chi2df[0], "df", chi2df[1], "critical", crit)
        fails += 1


def check_temperature_zero_real_row(mut fails: Int) raises:
    # P-I1's literal fixture: the draft receipt's logits row, a real decode
    # output at full vocab size (VOCAB f32s), a side effect of run-tests.sh /
    # the one-shot path (serve/engine.mojo). Skipped, not failed, when the
    # file is not present (a from-scratch checkout before any engine run).
    var path = ".work/draft-logits.bin"
    try:
        with open(path, "r") as f:
            var data = f.read_bytes()
            var n = len(data) // 4
            var row = List[Float32](unsafe_uninit_length=n)
            var p = data.unsafe_ptr().unsafe_bitcast[Float32]()
            for i in range(n):
                row[i] = p[i]
            var best_i = -1
            var best_v = -FMAX
            for i in range(n):
                if is_valid(row[i]) and row[i] > best_v:
                    best_v = row[i]
                    best_i = i
            var got = sample_row_ref(row, Float32(0), 0, Float32(1.0), Float32(0.0), UInt64(0), UInt64(0), 0)
            if got[0] == best_i and got[1] == Float32(1):
                print("  PASS real decode row (", n, "logits,", path, "): token", got[0], "matches greedy argmax")
            else:
                print("  FAIL real decode row: got", got[0], got[1], "want", best_i)
                fails += 1
    except:
        print("  SKIP real decode row: " + path + " not present (run run-tests.sh first)")


def check_penalties(mut fails: Int):
    print("== presence/frequency penalties")
    var logits = List[Float32]()
    for i in range(10):
        logits.append(Float32(i))
    var generated = List[Int]()
    generated.append(9)
    generated.append(9)
    generated.append(9)
    apply_penalties(logits, generated, Float32(1.0), Float32(0.5))
    # token 9 started at 9.0; presence 1.0 once + frequency 0.5*3 = 2.5 -> 9 - 1 - 1.5 = 6.5
    if abs(logits[9] - Float32(6.5)) < Float32(1e-5) and logits[8] == Float32(8):
        print("  PASS token 9 penalized to", logits[9], "; untouched token 8 stays", logits[8])
    else:
        print("  FAIL token 9 =", logits[9], "want 6.5; token 8 =", logits[8], "want 8")
        fails += 1


def main() raises:
    var fails = 0
    check_philox_kat(fails)
    check_temperature_zero(fails)
    check_temperature_zero_real_row(fails)
    check_distributions(fails)
    check_reproducibility(fails)
    check_speculation(fails)
    check_penalties(fails)
    if fails == 0:
        print("PASS: host reference sampler")
    else:
        print("FAIL: host reference sampler,", fails, "checks failed")
