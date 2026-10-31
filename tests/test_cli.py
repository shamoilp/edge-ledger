import importlib.util
from pathlib import Path

import pytest

from edge_ledger.cli import main

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def journal(tmp_path_factory):
    spec = importlib.util.spec_from_file_location("make_synthetic",
                                                  ROOT / "examples" / "make_synthetic.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    path = tmp_path_factory.mktemp("data") / "synthetic.csv"
    mod.write(path)
    return str(path)


def test_ic_finds_the_planted_signal_and_not_the_noise(journal, capsys):
    main(["ic", journal, "--factor", "momentum_like", "noise_02", "--returns", "fwd_5"])
    lines = {ln.split()[0]: ln for ln in capsys.readouterr().out.splitlines()
             if ln.strip().startswith(("momentum_like", "noise_02"))}
    assert "CLEARS" in lines["momentum_like"]
    assert "does not clear" in lines["noise_02"]


def test_ledger_sets_n_and_refuses_duplicates(tmp_path, journal, capsys):
    led = str(tmp_path / "l.json")
    for name in ("a", "b", "c"):
        main(["ledger", "--ledger", led, "add", name])
    with pytest.raises(SystemExit, match="already registered"):
        main(["ledger", "--ledger", led, "add", "a"])
    capsys.readouterr()
    main(["ic", journal, "--factor", "momentum_like", "--returns", "fwd_5", "--ledger", led])
    assert "N = 3 (ledger" in capsys.readouterr().out


def test_permute_and_walkforward_run(journal, capsys):
    main(["permute", journal, "--factor", "momentum_like", "--returns", "fwd_20",
          "--iters", "300"])
    main(["walkforward", journal, "--factor", "decays", "--returns", "fwd_5"])
    main(["walkforward", journal, "--select", "momentum_like", "noise_01",
          "--returns", "fwd_5"])
    out = capsys.readouterr().out
    assert "percentile of luck" in out and "selection skill" in out


@pytest.mark.parametrize("argv, msg", [
    (["--factor", "nope", "--returns", "fwd_5"], "no column"),
    (["--factor", "decays", "--returns", "date"], "cannot read a horizon"),
    (["--factor", "decays", "--returns", "fwd_5", "--min-names", "2"], "min-names"),
])
def test_bad_arguments_fail_clearly(journal, argv, msg):
    with pytest.raises(SystemExit, match=msg):
        main(["ic", journal, *argv])


def test_overlapping_oos_windows_are_refused(journal):
    with pytest.raises(SystemExit, match="overlap"):
        main(["walkforward", journal, "--factor", "decays", "--returns", "fwd_5",
              "--step-weeks", "1"])
