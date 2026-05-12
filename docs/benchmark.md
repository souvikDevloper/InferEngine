# Benchmark Notes

Run:

```bash
./scripts/run_server.sh
python scripts/bench.py -n 64 -c 16 --tokens 80
```

Record:

- requests
- concurrency
- generated tokens
- wall time
- token throughput
- p50 / p95 / p99 latency
- average batch size
- max batch observed
- KV-cache pressure events

Do not quote benchmark numbers without recording the environment.
