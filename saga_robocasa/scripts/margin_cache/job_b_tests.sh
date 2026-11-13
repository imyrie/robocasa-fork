#!/usr/bin/env bash
# Job B (after Job A): planner tests, compatibility tests, path verification and old-demo replay.
source "$(dirname "$0")/common.sh"
fail=0
echo "== offline planner/controller tests"
python -m pytest -q -p no:cacheprovider -W ignore saga_robocasa/benchmark/tests/test_footprint_planner_offline.py \
    saga_robocasa/benchmark/tests/test_nav_v3_offline.py saga_robocasa/benchmark/tests/test_calibration_offline.py 2>&1 | tail -3 || fail=1
echo "== planner tests on the real cache (episode_047 + ten smoke episodes, planner only)"
python -u -W ignore saga_robocasa/benchmark/test_planner_v3.py --output $B/planner_v3 2>&1 | grep --line-buffered -Ev "$FILTER" | tee $B/planner_v3/job_b_planner.log
grep -q "PLANNER TESTS PASS" $B/planner_v3/job_b_planner.log || fail=1
echo "== compatibility tests (old start paths bit-exact, manifest, determinism)"
python -m pytest -q -p no:cacheprovider -W ignore saga_robocasa/benchmark/tests/test_benchmark.py 2>&1 | tail -3 || fail=1
echo "== old demonstration replay (copy of nav_far_001; original untouched)"
mkdir -p $B/compat_replay && cp saga_robocasa/data/nav_demos/nav_far_001.npz saga_robocasa/data/nav_demos/nav_far_001.json $B/compat_replay/
python -u -W ignore saga_robocasa/validate_demo.py $B/compat_replay/nav_far_001 2>&1 | grep -E "^\[4\]|^\[6\]|VALIDATION" | cut -c1-300 | tee $B/compat_replay/replay.log
# read the full validation result (the log line above is truncated for display)
python -c "import json,sys; sys.exit(0 if json.load(open('$B/compat_replay/nav_far_001_validation.json'))['passed'] else 1)" || fail=1
(cd saga_robocasa/benchmark/manifests && sha256sum -c counter_to_sink_v1.sha256) || fail=1
echo "EXIT $fail"
exit $fail
