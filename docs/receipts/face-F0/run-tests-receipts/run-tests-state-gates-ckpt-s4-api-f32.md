# checkpoint-api DONE check

- PASS ref completion 200 (got 200)
- PASS flush 1
- PASS create 200 (got 200: {"created": true, "checkpoint": {"id": "lat:state@pos12:d62c627e8a1598ad", "object": "checkpoint", "pos": 12, "bytes": 61079672, "created": 1794458206, "ttl_s": 600, "expires_at": 1794458806, "identit)
- PASS create returned id lat:state@pos12:d62c627e8a1598ad
- PASS pos == len(P)-1 (12 vs 12)
- PASS one state file, bytes 61079672
- PASS flush 2
- PASS fork 200 (got 200: {"object": "checkpoint.fork", "model": "e11b-qwythos-pack", "checkpoint": "lat:state@pos12:d62c627e8a1598ad", "pos": 12, "branches": [{"tier": "file", "tier_source": "engine", "state_source": "explici)
- PASS fork ids == ref ids (32 tokens)
- PASS restored from file, cached_tokens 12 == 12
- PASS wrong identity refused 409 field pack (got 409 {"error": {"code": 409, "message": "IDENTITY_MISMATCH", "type": "identity_mismatch", "field": "pack"}})
- PASS list shows 1 live
- PASS get 200
- PASS delete true
- PASS get after delete 404
- PASS state file removed
- PASS [p03-story:p04-list-planets] ref completion 200 (got 200)
- PASS [p03-story:p04-list-planets] flush before create
- PASS [p03-story:p04-list-planets] create 200 (got 200)
- PASS [p03-story:p04-list-planets] flush before fork
- PASS [p03-story:p04-list-planets] fork 200 (got 200)
- PASS [p03-story:p04-list-planets] fork ids == ref ids (32 tokens)
- PASS [p05-math:p06-translate] ref completion 200 (got 200)
- PASS [p05-math:p06-translate] flush before create
- PASS [p05-math:p06-translate] create 200 (got 200)
- PASS [p05-math:p06-translate] flush before fork
- PASS [p05-math:p06-translate] fork 200 (got 200)
- PASS [p05-math:p06-translate] fork ids == ref ids (32 tokens)

ref ids: [262, 413, 307, 606, 220, 15, 25, 198, 285, 460, 220, 15, 198, 262, 413, 307, 606, 220, 16, 25, 198, 285, 460, 220, 16, 198, 262, 264, 11, 292, 283, 220]
fork ids: [262, 413, 307, 606, 220, 15, 25, 198, 285, 460, 220, 15, 198, 262, 413, 307, 606, 220, 16, 25, 198, 285, 460, 220, 16, 198, 262, 264, 11, 292, 283, 220]
create timings: {"prefill_s": 0.055499216, "decode_s": 0.008342929000000006, "tok_s_gen": 0.0, "drafted": null, "accepted": null, "cached": 0, "prefill_rows": 12, "restore_s": 0.012835155, "load_s": 0.0, "load_read_s": 0.0, "load_parse_s": 0.0, "load_pin_s": 0.0, "load_host_s": 0.0, "load_h2d_s": 0.0, "wall_s": 0.077284193, "finish": "length", "fail": 0, "path": "mega", "grammar": "none"}
fork timings: {"prefill_s": 0.062657808, "decode_s": 0.227839978, "tok_s_gen": 136.06040639628222, "drafted": null, "accepted": null, "cached": 12, "prefill_rows": 15, "restore_s": 0.001881321, "load_s": 0.009481745, "load_read_s": 0.00509944, "load_parse_s": 0.003616781, "load_pin_s": 0.0, "load_host_s": 7e-08, "load_h2d_s": 0.000764294, "wall_s": 0.302332445, "finish": "length", "fail": 0, "path": "mega", "grammar": "none", "load_tier": "file"}

extra pairs: [{"pair": "p03-story:p04-list-planets", "ref_ids": [42120, 198, 17, 13, 47334, 198, 18, 13, 8964, 198, 19, 13, 20403, 198, 20, 13, 48017, 198, 21, 13, 47503, 198, 22, 13, 76925, 350, 198, 23, 13, 76879, 198, 24], "fork_ids": [42120, 198, 17, 13, 47334, 198, 18, 13, 8964, 198, 19, 13, 20403, 198, 20, 13, 48017, 198, 21, 13, 47503, 198, 22, 13, 76925, 350, 198, 23, 13, 76879, 198, 24], "match": true}, {"pair": "p05-math:p06-translate", "ref_ids": [328, 31538, 183012, 5810, 46994, 72219, 1962, 15637, 60354, 303, 3262, 5367, 1149, 198, 48, 25, 3437, 369, 279, 6511, 314, 9338, 30, 198, 32, 25, 26583, 310, 5753, 25, 328, 760], "fork_ids": [328, 31538, 183012, 5810, 46994, 72219, 1962, 15637, 60354, 303, 3262, 5367, 1149, 198, 48, 25, 3437, 369, 279, 6511, 314, 9338, 30, 198, 32, 25, 26583, 310, 5753, 25, 328, 760], "match": true}]
engine: state saved: ~/Projects/mojo/mojo-baro-lanes/F0/.work/run-tests-state-gates/ckpt-s4/api-f32/ckpts/lat_state_pos12_d62c627e8a1598ad.baro  pos 12  kv pages 1  format BAROST01
engine: state saved: ~/Projects/mojo/mojo-baro-lanes/F0/.work/run-tests-state-gates/ckpt-s4/api-f32/ckpts/lat_state_pos12_d62c627e8a1598ad.baro  pos 12  in 0.008261549 s
engine: state loaded: ~/Projects/mojo/mojo-baro-lanes/F0/.work/run-tests-state-gates/ckpt-s4/api-f32/ckpts/lat_state_pos12_d62c627e8a1598ad.baro  pos 12  in 0.009481745 s  read 0.00509944  parse 0.003616781  pin 0.0  host 7e-08  h2d 0.000764294
engine: state saved: ~/Projects/mojo/mojo-baro-lanes/F0/.work/run-tests-state-gates/ckpt-s4/api-f32/ckpts/lat_state_pos19_1f2aa08500e35a70.baro  pos 19  kv pages 1  format BAROST01
engine: state saved: ~/Projects/mojo/mojo-baro-lanes/F0/.work/run-tests-state-gates/ckpt-s4/api-f32/ckpts/lat_state_pos19_1f2aa08500e35a70.baro  pos 19  in 0.024291153 s
engine: state saved: ~/Projects/mojo/mojo-baro-lanes/F0/.work/run-tests-state-gates/ckpt-s4/api-f32/ckpts/lat_state_pos14_a71063ee3634ccd7.baro  pos 14  kv pages 1  format BAROST01
engine: state saved: ~/Projects/mojo/mojo-baro-lanes/F0/.work/run-tests-state-gates/ckpt-s4/api-f32/ckpts/lat_state_pos14_a71063ee3634ccd7.baro  pos 14  in 0.010068839 s
engine: state loaded: ~/Projects/mojo/mojo-baro-lanes/F0/.work/run-tests-state-gates/ckpt-s4/api-f32/ckpts/lat_state_pos14_a71063ee3634ccd7.baro  pos 14  in 0.025819812 s  read 0.00855412  parse 0.016507797  pin 0.0  host 1.6e-07  h2d 0.000756425
