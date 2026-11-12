# turn-api gate, server A

- PASS create 8k checkpoint 200 (got 200 {'created': True, 'checkpoint': {'id': 'lat:state@pos7999:0ea1877791c22a14', 'object': 'checkpoint', 'pos': 7999, 'bytes': 581205316, 'created': 1794458215, 'tt)
- PASS fork reference for suffix 0 200
- PASS fork reference for suffix 1 200
- PASS fork reference for suffix 2 200
- PASS fork reference for suffix 3 200
- PASS fork reference for suffix 4 200
- PASS fresh15: 20/20 turns ok (cached_tokens 7999, ttft present); first bad: []
- PASS fresh15: ids == fork reference for the same suffix
- INFO fresh15: median ttft_s 0.0683, max 0.0770, tiers ['hot']
- PASS fresh15: every turn hit the hot tier
- PASS fresh8: 20/20 turns ok (cached_tokens 7999, ttft present); first bad: []
- PASS fresh8: ids == fork reference for the same suffix
- INFO fresh8: median ttft_s 0.0414, max 0.0462, tiers ['hot']
- PASS fresh8 (hot prefix): median ttft_s 0.0414 < 0.060 (QS bar)
- PASS fresh8: every turn hit the hot tier
- PASS repeat: 20/20 turns ok (cached_tokens 7999, ttft present); first bad: []
- PASS repeat: ids == fork reference for the same suffix
- INFO repeat: median ttft_s 0.0114, max 0.0697, tiers ['hot']
- PASS repeat: median ttft_s 0.0114 < 0.035
- PASS file: 5/5 turns ok (cached_tokens 7999, ttft present); first bad: []
- PASS file: ids == fork reference for the same suffix
- INFO file: median ttft_s 0.1498, max 0.1647, tiers ['file']
- PASS metrics: scrape parses
- PASS metrics: baro_turns_total 65 == turns made 65
- PASS metrics: forks total 70 == turns + reference forks 70
- INFO chat section skipped: TOKENIZER not set
- PASS file pull 200 (581205316 bytes)
- PASS X-Baro-Meta carries no token ids (curl 100 KB header cap)

# import section, server B

- PASS pulled file sha256 == X-Baro-Sha256
- PASS import on server B 200 created (got 200 {'created': True, 'checkpoint': {'id': 'lat:state@pos7999:0ea1877791c22a14', 'object': 'checkpoint', 'pos': 7999, 'bytes)
- PASS server B fork suffix 0: ids == server A's, cached 7999
- PASS server B fork suffix 1: ids == server A's, cached 7999
- PASS server B fork suffix 2: ids == server A's, cached 7999
