import subprocess
import sys
from pathlib import Path

import pytest

from abkit.cli import main

EXAMPLE = Path(__file__).parent / "data" / "example.csv"


def write_csv(path, arms):
    """arms: list of (variant, n_users, n_conversions)."""
    lines = ["variant,outcome"]
    for arm, n, x in arms:
        lines += [f"{arm},{1 if i < x else 0}" for i in range(n)]
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def run(argv, capsys):
    code = main(argv)
    out = capsys.readouterr()
    return code, out.out, out.err


@pytest.mark.parametrize("extra, expected", [
    ([], ["14,749 per arm", "29,498 total", "+1.00 pp", "+10.0% relative", "80%"]),
    (["--baseline", "0.05"], ["8,155 per arm", "16,310 total", "+20.0% relative"]),
    (["--power", "0.9"], ["19,744 per arm", "39,488 total", "90%"]),
])
def test_plan(extra, expected, capsys):
    argv = ["plan", "--baseline", "0.10", "--mde", "0.01"] + extra
    # later --baseline/--power override the earlier ones (argparse keeps the last)
    code, out, err = run(argv, capsys)
    assert code == 0 and err == ""
    for text in expected:
        assert text in out


@pytest.mark.parametrize("argv", [
    ["plan", "--baseline", "1.2", "--mde", "0.01"],
    ["plan", "--baseline", "0.10", "--mde", "0"],
    ["plan", "--baseline", "0.10", "--mde", "0.01", "--alpha", "0"],
])
def test_plan_invalid_input(argv, capsys):
    code, out, err = run(argv, capsys)
    assert code == 2 and err.startswith("error:") and out == ""


def test_plan_missing_argument_exits_nonzero():
    with pytest.raises(SystemExit) as e:
        main(["plan", "--baseline", "0.10"])
    assert e.value.code == 2


def test_analyze_example(capsys):
    code, out, err = run(["analyze", str(EXAMPLE), "--delta", "0.005"], capsys)
    assert code == 0 and err == ""
    for text in ["10,000", "1,000", "1,100", "10.00%", "11.00%", "-> OK",
                 "+1.00 pp (+10.0% relative)", "0.0211", "[+0.15 pp, +1.85 pp]",
                 "Delta:    0.50 pp", "INCONCLUSIVE", "collect more data"]:
        assert text in out
    assert "WARNING" not in out


def test_analyze_launch(tmp_path, capsys):
    path = write_csv(tmp_path / "d.csv", [("control", 50000, 5000), ("treatment", 50000, 5550)])
    code, out, _ = run(["analyze", path, "--delta", "0.005"], capsys)
    assert code == 0
    for text in ["+1.10 pp (+11.0% relative)", "1.5e-08", "[+0.72 pp, +1.48 pp]", "LAUNCH"]:
        assert text in out


def test_analyze_srm_mismatch_warns_twice(tmp_path, capsys):
    path = write_csv(tmp_path / "d.csv", [("control", 50000, 5000), ("treatment", 52000, 5200)])
    code, out, _ = run(["analyze", path, "--delta", "0.005"], capsys)
    lines = out.strip().splitlines()
    assert code == 0
    assert "-> MISMATCH" in out and "3.8e-10" in out
    srm_line = next(i for i, l in enumerate(lines) if l.startswith("SRM check"))
    assert lines[srm_line + 1].startswith("WARNING")  # directly after the SRM line
    assert lines[-1].startswith("WARNING")            # and again as the last line
    assert "NO_MEANINGFUL_EFFECT" in out              # results still printed


def test_analyze_expected_ratio(tmp_path, capsys):
    path = write_csv(tmp_path / "d.csv", [("control", 3000, 300), ("treatment", 7000, 700)])
    code, out, _ = run(["analyze", path, "--delta", "0.005", "--expected-ratio", "0.3"], capsys)
    assert code == 0 and "expected 30/70" in out and "-> OK" in out


@pytest.mark.parametrize("content, message", [
    ("variant,converted\ncontrol,1\n", "column 'outcome' not found (found: variant, converted)"),
    ("group,outcome\ncontrol,1\n", "column 'variant' not found"),
    ("variant,outcome\ncontrol,1\ntreatment,0\nother,1\n", "other"),
    ("variant,outcome\ncontrol,1\ntreatment,2\n", "outcome must be 0 or 1"),
    ("variant,outcome\ncontrol,1\ncontrol,0\n", "at least one row"),
    ("", "could not read"),
])
def test_analyze_bad_csv(content, message, tmp_path, capsys):
    path = tmp_path / "bad.csv"
    path.write_text(content)
    code, out, err = run(["analyze", str(path), "--delta", "0.005"], capsys)
    assert code == 2 and out == ""
    assert err.startswith("error:") and message in err


def test_analyze_missing_file(tmp_path, capsys):
    code, _, err = run(["analyze", str(tmp_path / "nope.csv"), "--delta", "0.005"], capsys)
    assert code == 2 and "file not found" in err


def test_analyze_invalid_delta(capsys):
    code, _, err = run(["analyze", str(EXAMPLE), "--delta", "0"], capsys)
    assert code == 2 and "delta" in err


def test_python_dash_m_runs():
    r = subprocess.run([sys.executable, "-m", "abkit", "plan", "--baseline", "0.10", "--mde", "0.01"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "14,749 per arm (29,498 total)" in r.stdout
