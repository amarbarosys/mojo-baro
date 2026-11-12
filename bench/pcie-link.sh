#!/usr/bin/env bash
# Machine-state read-back (PROTOCOL-RULES P1) for every timed arm that crosses PCIe: the CPU root port link of the card.
# Never read 03:00.0, it always says 16 GT/s. A boot can train the link at 2.5 GT/s (3.4 GB/s instead of 28; ~/Brain/OS/gpu-pcie-link-downtraining.md).
# source this file; pcie_link prints one receipt line, pcie_ok is true only at full speed.
pcie_link() { local d=/sys/bus/pci/devices/0000:00:01.1; echo "pcie 00:01.1 $(cat $d/current_link_speed) x$(cat $d/current_link_width)"; }
pcie_ok() { [[ "$(pcie_link)" == *"16.0 GT/s"*"x16" ]]; }
pcie_flag() { pcie_ok && echo "" || echo " (PCIe LINK DEGRADED: miss-DMA timings void, rerun after a reboot with a full link)"; }
