#!/usr/bin/env python3
# Baseline: tinygrad's Llama (examples/llama3.py) greedy decode on the same GGUF and prompt ids as tests/ref_*.txt.
# Prints tok/s over the warm tokens (the JIT captures in the first steps) and how many ids equal the llama.cpp reference.
# usage: PYTHONPATH=<tinygrad>:<tinygrad>/examples DEV=AMD python3 tools/tinygrad_decode.py MODEL.gguf tests/ref_story.txt
import sys, time
from pathlib import Path
import llama3
from tinygrad import Device, Tensor

def main():
  llama3.device, llama3.TEMPERATURE = Device.DEFAULT, 0.0
  model = llama3.build_transformer(Path(sys.argv[1]), model_size="1B", quantize="float16")
  prompt, ref = ([int(x) for x in line.split(",")] for line in open(sys.argv[2]).read().split("\n")[:2])
  pos, out, times = 0, [], []
  for tok in prompt[:-1]:  # one token per step, as llama3.py's own prefill does
    model(Tensor([[tok]], device=llama3.device), pos, 0.0).realize()
    pos += 1
  tok = prompt[-1]
  for _ in range(len(ref)):
    st = time.perf_counter()
    tok = model(Tensor([[tok]], device=llama3.device), pos, 0.0).item()  # item() waits for the GPU
    times.append(time.perf_counter() - st)
    out.append(tok)
    pos += 1
  warm = times[10:]
  same = next((i for i, (a, b) in enumerate(zip(out, ref)) if a != b), len(ref))
  print(f"tinygrad decode {len(warm) / sum(warm):.1f} tok/s (warm tokens 10..{len(ref) - 1}), match_prefix {same} of {len(ref)}")

if __name__ == "__main__": main()
