# Tests

Tests cover biological assumptions (cone fundamentals, center–surround, ON/OFF, LN sigmoid, temporal RC, stimulus shapes, DoG fit, color opponent), config/state/export, and slab layout.

## Run tests

```bash
pip install -r requirements.txt -r requirements-test.txt
PYTHONPATH=. pytest tests/ -v
```

## Performance benchmarks

To see where simulation time is spent (tick, convolution, render, etc.):

```bash
python tests/bench_performance.py
```

With pytest:

```bash
pytest tests/bench_performance.py -v -s
```

Target: tick(256) &lt; 16.67 ms for 60 Hz. Build the Cython extensions in `hot_numerical/` if you haven’t.

Skip the slow 2048 benchmark: `BENCH_SKIP_2048=1 python tests/bench_performance.py`

## CI

Tests run on **push** and **pull_request** via GitHub Actions (`.github/workflows/tests.yml`). See `docs/TESTING.md` for setup.
