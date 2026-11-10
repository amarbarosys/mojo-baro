# run one nt fp16 GEMM (C = A @ W.T) N times, synchronizing after each launch, to measure the hang rate
# MOJO_GEMM=1 takes the Mojo WMMA path, MOJO_GEMM=0 tinygrad's own kernel
import os, statistics, sys, threading, time, traceback
from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import getenv
from tinygrad.engine.realize import run_linear

M, K, N, CNT = getenv("M", 2048), getenv("K", 4096), getenv("N", 4096), getenv("CNT", 1000)

def watch_allocs(dev) -> dict[int, tuple[int, str]]:
  # live allocations by address: every alloc (fresh or an LRU cache hit) and free, to find what overlaps the timeline
  live, al = {}, dev.allocator
  orig_alloc, orig_free = al.alloc, al.free
  def alloc(size, options=None):
    st = orig_alloc(size, options)
    if (old:=live.get(st.buf)) is not None: print(f"ALIAS 0x{st.buf:x} handed out again: live {old}, new {size} {options}")
    live[st.buf] = (size, str(options))
    return st
  def free(st, size, options=None):
    live.pop(st.buf, None)
    return orig_free(st, size, options)
  al.alloc, al.free = alloc, free
  return live

def watch_signal(tl, log:list, stop:threading.Event):
  # every value the timeline signal takes, polled from the host, so a reset shows what it was before and when
  last = tl[0]
  while not stop.is_set():
    if (v:=tl[0]) != last: log.append((time.perf_counter(), last, v)); last = v

if __name__ == "__main__":
  live = watch_allocs(Device[Device.DEFAULT]) if getenv("WATCH") else None
  a = Tensor.randn(M, K, dtype=dtypes.half).realize()
  w = Tensor.randn(N, K, dtype=dtypes.half).realize()
  if getenv("MOJO_GEMM"):
    from tinygrad.runtime.support import mojo_gemm
    out = mojo_gemm.dot(a, w.T)
    if out is None: sys.exit(f"FAIL setup: mojo path not taken for {M}x{K}x{N}")
  else: out = a @ w.T
  linear = out.schedule_linear()
  dev = Device[Device.DEFAULT]
  def u64(b): return b.host.view(fmt='Q')[0]
  def probe():
    # the launch path as stages: host put -> doorbell -> CP read pointer -> GPU work -> signal write -> host sees it
    q, tl = dev.compute_queue, dev.timeline.host.view(fmt='Q')
    return {"put": u64(q.put_value), "wptr": u64(q.write_ptr), "rptr": u64(q.read_ptr), "signal": tl[0], "target": tl[1]}
  times, last_signal = [], 0
  sig_log, stop = [], threading.Event()
  if live is not None:
    tl_addr = dev.timeline._buf
    threading.Thread(target=watch_signal, args=(dev.timeline.host.view(fmt='Q'), sig_log, stop), daemon=True).start()
  def report_watch():
    if live is None: return
    stop.set()
    over = {f"0x{a:x}": v for a, v in live.items() if a < tl_addr + 16 and tl_addr < a + v[0] and a != tl_addr}
    print(f"timeline 0x{tl_addr:x}; live allocs overlapping it: {over or 'none'}")
    print(f"signal changes: {len(sig_log)}; backwards: {[(round(t, 3), a, b) for t, a, b in sig_log if b < a][:5]}; last 5: {sig_log[-5:]}")
  for i in range(CNT):
    st = time.perf_counter()
    try:
      run_linear(linear)
      dev.synchronize()
    except RuntimeError as e:
      traceback.print_exc()
      at_hang = probe()
      time.sleep(10)
      late = probe()
      print(f"probe at hang: {at_hang}\nprobe +10 s:  {late}")
      stage = ("doorbell/CP never fetched (rptr < wptr)" if late["rptr"] < late["wptr"] else
               "signal landed late: wait timeout, not a hang" if late["signal"] >= late["target"] else
               f"signal went backwards to {late['signal']} (was >= {last_signal}): overwritten, not stuck" if late["signal"] < last_signal else
               "CP fetched all, signal never written: GPU work or EOP stuck")
      print(f"broken stage: {stage}")
      report_watch()
      if times: print(f"launch ms before hang: median {statistics.median(times)*1e3:.2f} last5 {[round(t*1e3, 2) for t in times[-5:]]}")
      print(f"FAIL launch {i+1}/{CNT}: {e!r}", flush=True)
      os._exit(1) # no atexit: the device finalizer would wait another full timeout on the dead signal
    times.append(time.perf_counter() - st)
    last_signal = dev.timeline.host.view(fmt='Q')[0]
  report_watch()
  print(f"OK {CNT} launches {M}x{K}x{N} nt mojo={getenv('MOJO_GEMM')} launch ms median {statistics.median(times)*1e3:.3f} probe {probe()}")
