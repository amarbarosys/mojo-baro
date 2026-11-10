#!/usr/bin/env python3
"""usage: fork-link-forwarder.py LISTEN_IP BASE_PORT DEST_IP DEST_PORT

The far end of the shaped link for bench/p1-fork-gate.sh. It runs inside the peer network
namespace, so a fork from node A to LISTEN_IP crosses the shaped veth hop once, and is relayed back
to node B on the host. Four ports, one behaviour each, so the harness picks an arm by address and
nothing about a request decides how it is treated:

  BASE+0  pass    bytes relayed untouched
  BASE+1  flip    one payload byte of a POST /v1/state/import is inverted. The LAT1 payload_sha no
                  longer matches, so node B must answer 409 state_identity and node A must relay it
  BASE+2  swapkv  the K and V halves of the state body are exchanged AND payload_sha is recomputed
                  (hmac is zero until P0b pairing, so the header is forgeable). Node B accepts and
                  restores it; only the generated ids can show the state is wrong (P11)
  BASE+3  sink    reads and discards, then answers "BYTES SECONDS\\n": the link-rate read-back,
                  timed at the receiver so it measures the shaped hop and nothing else

A verifier and a fault injector. Nothing at serve time uses it."""
import hashlib
import socket
import sys
import threading
import time

LAT1 = 256
SHA_AT = slice(192, 224)


def recv_until(sock, marker, limit=1 << 20):
    buf = b""
    while marker not in buf:
        chunk = sock.recv(65536)
        if not chunk:
            break
        buf += chunk
        if len(buf) > limit:
            break
    return buf


def recv_exact(sock, n, have=b""):
    parts, got = [have], len(have)
    while got < n:
        chunk = sock.recv(min(1 << 20, n - got))
        if not chunk:
            break
        parts.append(chunk)
        got += len(chunk)
    return b"".join(parts)


def pump(src, dst):
    try:
        while True:
            chunk = src.recv(1 << 20)
            if not chunk:
                break
            dst.sendall(chunk)
    except OSError:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def mutate(body, mode):
    if mode == "flip":
        b = bytearray(body)
        b[LAT1 + len(b[LAT1:]) // 2] ^= 0xFF
        return bytes(b)
    # swapkv: BAROST0x = 72-byte header, 4*pos tokens, conv, ssm, then two equal halves (K, V) in
    # both the f32 and the int8 format
    hdr, st = bytearray(body[:LAT1]), body[LAT1:]
    pos = int.from_bytes(st[8:16], "little")
    conv_n = int.from_bytes(st[16:24], "little")
    ssm_n = int.from_bytes(st[24:32], "little")
    off = 72 + 4 * pos + 4 * (conv_n + ssm_n)
    rest = st[off:]
    half = len(rest) // 2
    if st[:6] != b"BAROST" or len(rest) % 2 or rest[:half] == rest[half:]:
        raise ValueError("not a BAROST body with two distinct equal K/V halves")
    st = st[:off] + rest[half:] + rest[:half]
    hdr[SHA_AT] = hashlib.sha256(st).digest()
    return bytes(hdr) + st


def handle(client, mode, dest):
    try:
        if mode == "sink":
            t0, n = None, 0
            while True:
                chunk = client.recv(1 << 20)
                if not chunk:
                    break
                t0 = t0 or time.monotonic()
                n += len(chunk)
            client.sendall(f"{n} {time.monotonic() - (t0 or time.monotonic()):.6f}\n".encode())
            return
        upstream = socket.create_connection(dest, timeout=10)
        upstream.settimeout(None)
        if mode != "pass":
            head = recv_until(client, b"\r\n\r\n")
            sep = head.index(b"\r\n\r\n") + 4
            first, lower = head[:sep].split(b"\r\n", 1)[0], head[:sep].lower()
            if first.startswith(b"POST /v1/state/import") and b"content-length:" in lower:
                n = int(lower.split(b"content-length:")[1].split(b"\r\n")[0])
                body = mutate(recv_exact(client, n, head[sep:]), mode)
                print(f"{mode}: rewrote a {n}-byte state", flush=True)
                upstream.sendall(head[:sep] + body)
            else:
                upstream.sendall(head)
        t = threading.Thread(target=pump, args=(client, upstream), daemon=True)
        t.start()
        pump(upstream, client)
        t.join(timeout=5)
        upstream.close()
    except Exception as e:  # a fault injector must never take the rig down silently
        print(f"{mode}: connection failed: {e!r}", flush=True)
    finally:
        client.close()


def serve(ip, port, mode, dest):
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((ip, port))
    s.listen(16)
    while True:
        c, _ = s.accept()
        c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        threading.Thread(target=handle, args=(c, mode, dest), daemon=True).start()


def main():
    ip, base, dip, dport = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    for k, mode in enumerate(("pass", "flip", "swapkv", "sink")):
        threading.Thread(target=serve, args=(ip, base + k, mode, (dip, dport)), daemon=True).start()
    print(f"forwarder ready {ip}:{base}..{base + 3} -> {dip}:{dport}", flush=True)
    threading.Event().wait()


if __name__ == "__main__":
    main()
