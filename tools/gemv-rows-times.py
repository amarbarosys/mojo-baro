"""gemv-rows-times.py KERNEL_TRACE_CSV: device time per variant of bench/gemv-kq-rows.mojo's timed rounds.
The rounds launch amar_gemv_kq, R1, R2, R4, R8 in turn, 16 launches each; the last 5 * 9 * 16 gemv launches are those rounds."""
import csv, statistics, sys
WEIGHT_BYTES, N = 7741440, 5 * 9 * 16
rows = sorted((r for r in csv.DictReader(open(sys.argv[1])) if 'gemv' in r['Kernel_Name']), key=lambda r: int(r['Start_Timestamp']))[-N:]
for v, name in enumerate(['amar_gemv_kq', 'R1', 'R2', 'R4', 'R8']):
  g = [r for i, r in enumerate(rows) if (i // 16) % 5 == v]
  if len({r['Kernel_Name'] for r in g}) != 1: sys.exit(f'FAIL times: variant {name} mixes kernels')
  t = [int(r['End_Timestamp']) - int(r['Start_Timestamp']) for r in g]
  med = statistics.median(t)
  print(f'{name:13} median {med / 1000:6.2f} us  min {min(t) / 1000:6.2f}  max {max(t) / 1000:6.2f}  {WEIGHT_BYTES / med:5.0f} GB/s')
