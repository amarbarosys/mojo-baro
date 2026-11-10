#!/usr/bin/env python3
"""Mock /v1/chat/completions for bench/tests/baro-chat-test.sh: HTTP/1.1, chunked SSE in OpenAI chunk shape.
The reply echoes the last user message, streamed 3 bytes per SSE event and re-chunked at odd byte offsets (so lines
and UTF-8 sequences split across chunks); the finish event carries usage + timings like baro-serve. A user message
'ERR' answers 400 with a JSON error. Every request body is appended to $MOCK_LOG. usage: PORT_FILE"""
import json, os, socketserver, sys
from http.server import BaseHTTPRequestHandler
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        open(os.environ["MOCK_LOG"], "ab").write(body + b"\n")
        req = json.loads(body); last = req["messages"][-1]["content"]
        if last == "ERR":
            e = json.dumps({"error": {"message": "messages is empty", "type": "invalid_request_error"}}).encode()
            self.send_response(400); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(e))); self.end_headers(); self.wfile.write(e); return
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
        sse = b""
        txt = last; step = 2
        for i in range(0, len(txt), step):
            sse += b"data: " + json.dumps({"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": txt[i:i + step]}, "finish_reason": None}]}, ensure_ascii=False).encode() + b"\n\n"
        sse += b"data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 9, "completion_tokens": 7, "total_tokens": 16, "baro": {"cached_tokens": 4, "prefill_rows": 5}}, "timings": {"tok_s_gen": 61.27, "prefill_s": 0.042}}).encode() + b"\n\n"
        sse += b"data: [DONE]\n\n"
        for i in range(0, len(sse), 37):
            c = sse[i:i + 37]; self.wfile.write(b"%x\r\n" % len(c) + c + b"\r\n")
        self.wfile.write(b"0\r\n\r\n")
s = socketserver.TCPServer(("127.0.0.1", 0), H)
open(sys.argv[1], "w").write(str(s.server_address[1]))
s.serve_forever()
