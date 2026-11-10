/* bench/latentos-ipc-probe.c -- C-language B3 HIP IPC probe.
 * briefs/2026-09-15-p2-a03-ipc-c.md, deciding between the two explanations
 * bench/latentos-ipc-probe.mojo (commit 47e2896) could not separate:
 * Mojo's external_call ABI for the 64-byte by-value hipIpcMemHandle_t, or a
 * real driver restriction. hipcc compiles the actual C struct-by-value ABI
 * and links libamdhip64 itself, so if THIS also fails the same way, the
 * cause is the driver, not Mojo's FFI. No MAX runtime involved.
 *
 * Build: /opt/rocm/bin/hipcc -D__HIP_PLATFORM_AMD__ -I/opt/rocm/include \
 *   bench/latentos-ipc-probe.c -o .work/latentos-ipc-probe-c -lcrypto
 * (hipcc's own driver did not add -I/opt/rocm/include or define the
 * platform macro when fed a .c file under -x c on this box; both are
 * required or hip_runtime.h's own #error fires.)
 * Run (from the repo root, the socket path is relative):
 *   ~/.local/bin/gpu-wait run --vram 5 -- .work/latentos-ipc-probe-c
 */
#include <hip/hip_runtime.h>
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>

#define SOCK_PATH ".work/latentos-ipc-probe-c.sock"
#define SIZE (2UL * 1024 * 1024 * 1024) /* 2 GiB, same as the Mojo probe */
#define CANARY 64
#define HASH_CHUNK (64UL * 1024 * 1024) /* divides SIZE */

static void check(hipError_t rc, const char *what) {
    if (rc != hipSuccess) {
        fprintf(stderr, "HIP call failed: %s rc=%d (%s)\n", what, rc, hipGetErrorString(rc));
        exit(1);
    }
}

static void send_all(int fd, const void *buf, size_t n) {
    size_t sent = 0;
    while (sent < n) {
        ssize_t w = write(fd, (const char *)buf + sent, n - sent);
        if (w <= 0) { perror("write"); exit(1); }
        sent += (size_t)w;
    }
}

static void recv_all(int fd, void *buf, size_t n) {
    size_t got = 0;
    while (got < n) {
        ssize_t r = read(fd, (char *)buf + got, n - got);
        if (r <= 0) { perror("read"); exit(1); }
        got += (size_t)r;
    }
}

static long now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000L + ts.tv_nsec;
}

/* Streams the buffer through one HASH_CHUNK host bounce: the whole 2 GiB never sits in host memory. */
static void hash_device(void *dptr, unsigned char *out) {
    void *h = malloc(HASH_CHUNK);
    EVP_MD_CTX *md = EVP_MD_CTX_new();
    if (!h || !md) { fprintf(stderr, "hash_device: out of memory\n"); exit(1); }
    EVP_DigestInit_ex(md, EVP_sha256(), NULL);
    for (size_t off = 0; off < SIZE; off += HASH_CHUNK) {
        check(hipMemcpy(h, (char *)dptr + off, HASH_CHUNK, hipMemcpyDeviceToHost), "hipMemcpy D2H (hash)");
        EVP_DigestUpdate(md, h, HASH_CHUNK);
    }
    unsigned len = 0;
    EVP_DigestFinal_ex(md, out, &len);
    EVP_MD_CTX_free(md);
    free(h);
}

int main(void) {
    unlink(SOCK_PATH);
    int listen_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (listen_fd < 0) { perror("socket"); return 1; }
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCK_PATH, sizeof(addr.sun_path) - 1);
    if (bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) { perror("bind"); return 1; }
    if (listen(listen_fd, 1) < 0) { perror("listen"); return 1; }

    long t_start = now_ns();
    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return 1; }

    if (pid == 0) {
        /* child: owns the source buffer */
        int sock = socket(AF_UNIX, SOCK_STREAM, 0);
        if (sock < 0 || connect(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) { perror("connect"); _exit(10); }

        void *dptr;
        check(hipMalloc(&dptr, SIZE), "hipMalloc (child)");
        check(hipMemset(dptr, 0xAB, SIZE), "hipMemset bulk");
        unsigned char canary[CANARY];
        for (int i = 0; i < CANARY; i++) canary[i] = (unsigned char)i;
        check(hipMemcpy(dptr, canary, CANARY, hipMemcpyHostToDevice), "hipMemcpy head canary");
        for (int i = 0; i < CANARY; i++) canary[i] = (unsigned char)(255 - i);
        check(hipMemcpy((char *)dptr + SIZE - CANARY, canary, CANARY, hipMemcpyHostToDevice), "hipMemcpy tail canary");
        check(hipDeviceSynchronize(), "hipDeviceSynchronize (child fill)");

        hipIpcMemHandle_t handle;
        check(hipIpcGetMemHandle(&handle, dptr), "hipIpcGetMemHandle");
        send_all(sock, &handle, sizeof(handle));

        unsigned char digest[SHA256_DIGEST_LENGTH];
        hash_device(dptr, digest);
        send_all(sock, digest, sizeof(digest));

        unsigned char ack;
        recv_all(sock, &ack, 1);
        close(sock);
        _exit(0);
    }

    /* parent: opens the handle, times the D2D copy */
    int conn = accept(listen_fd, NULL, NULL);
    if (conn < 0) { perror("accept"); return 1; }

    hipIpcMemHandle_t handle;
    recv_all(conn, &handle, sizeof(handle));
    long handoff_us = (now_ns() - t_start) / 1000;
    printf("parent: received handle, %zu bytes, handoff %ld us\n", sizeof(handle), handoff_us);

    unsigned char child_digest[SHA256_DIGEST_LENGTH];
    recv_all(conn, child_digest, sizeof(child_digest));

    void *dst;
    check(hipMalloc(&dst, SIZE), "hipMalloc (parent)");

    void *src;
    hipError_t rc_open = hipIpcOpenMemHandle(&src, handle, hipIpcMemLazyEnablePeerAccess);
    if (rc_open != hipSuccess) {
        fprintf(stderr, "HIP call failed: hipIpcOpenMemHandle rc=%d (%s)\n", rc_open, hipGetErrorString(rc_open));
        unsigned char ack = 1;
        send_all(conn, &ack, 1);
        waitpid(pid, NULL, 0);
        return 1;
    }

    long t0 = now_ns();
    check(hipMemcpy(dst, src, SIZE, hipMemcpyDeviceToDevice), "hipMemcpy D2D");
    check(hipDeviceSynchronize(), "hipDeviceSynchronize (copy)");
    long t1 = now_ns();
    long copy_us = (t1 - t0) / 1000;
    printf("parent: device-to-device copy took %ld us for %lu bytes\n", copy_us, SIZE);

    check(hipIpcCloseMemHandle(src), "hipIpcCloseMemHandle");

    unsigned char parent_digest[SHA256_DIGEST_LENGTH];
    hash_device(dst, parent_digest);
    int match = memcmp(parent_digest, child_digest, SHA256_DIGEST_LENGTH) == 0;
    printf("sha256 match: %s\n", match ? "PASS" : "FAIL");
    printf("copy under 10ms: %s (%ld us)\n", copy_us < 10000 ? "PASS" : "FAIL", copy_us);

    unsigned char ack = 1;
    send_all(conn, &ack, 1);
    close(conn);
    close(listen_fd);
    unlink(SOCK_PATH);
    int status = 0;
    waitpid(pid, &status, 0);
    printf("child exit status: %d\n", WEXITSTATUS(status));

    return (match && copy_us < 10000) ? 0 : 1;
}
