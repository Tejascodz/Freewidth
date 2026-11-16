import math

import pytest

from freewidth.costs import DeviceFit, ModelSpec, fit_device_curve, wave_device_cost
from freewidth.grading import answer_key, extract_answer, is_equiv
from freewidth.pareto import kendall_tau, pareto_front
from freewidth.replay import Bank, ModelBank, Sample, Stream, price_trace
from freewidth.strategies import SC, Adaptive, Cascade, HWave, ProxyAlloc, Single, lead_probability, vote


# ---------------------------------------------------------------- grading
@pytest.mark.parametrize("text,ans", [
    ("so \\boxed{\\frac{1}{2}} done", "\\frac{1}{2}"),
    ("nested \\boxed{x^{2}+1}", "x^{2}+1"),
    ("work... #### 1,234", "1,234"),
    ("The answer is 42.", "42"),
    ("we get 3 then 17", "17"),
])
def test_extract(text, ans):
    assert extract_answer(text) == ans


@pytest.mark.parametrize("a,b,eq", [
    ("\\frac{1}{2}", "0.5", True), ("1,000", "1000", True), ("$18", "18", True),
    ("18 dollars", "18", True), ("x=3", "3", True), ("\\text{(B)}", "(B)", True),
    ("17", "18", False), (None, "1", False),
])
def test_equiv(a, b, eq):
    assert is_equiv(a, b) is eq


@pytest.mark.parametrize("a,b,eq", [
    ("\\sqrt{2}", "\\sqrt2", True), ("\\frac{\\pi}{4}", "\\pi/4", True),
    ("(1, 2)", "(1,2)", True), ("2\\sqrt{3}", "\\sqrt{12}", True),
    ("\\frac{\\pi}{3}", "\\frac{\\pi}{4}", False), ("x^2+1", "x^2+2", False),
])
def test_equiv_symbolic(a, b, eq):
    pytest.importorskip("math_verify")
    assert is_equiv(a, b) is eq


def test_answer_key_groups_equivalent_numbers():
    assert answer_key("1,000") == answer_key("1000.0") == answer_key("\\boxed{1000}".strip("\\boxed{}"))


# ------------------------------------------------------------------ costs
def _synthetic_rows(coef):
    fit = DeviceFit("m", coef)
    rows = []
    for P in (64, 512):
        for L in (64, 256):
            for w in (1, 2, 4, 8, 16, 32, 64):
                rows.append({"w": w, "L": L, "P": P, "seconds": fit.seconds(w, L, P), "joules": None})
    return rows


def test_fit_recovers_coefficients():
    true = {"e": 0.05, "d": 2e-5, "a": 2e-3, "b": 5e-5, "g": 1e-8}
    fit = fit_device_curve(_synthetic_rows(true), "m")
    assert fit.r2 > 0.9999
    for k in true:
        assert fit.coef[k] == pytest.approx(true[k], rel=0.05, abs=1e-9)


def test_free_width_and_continuous_batching():
    fit = DeviceFit("m", {"e": 0.0, "d": 0.0, "a": 2e-3, "b": 2e-5, "g": 0.0})
    # T(w)/T(1) = (a + b w)/(a + b) <= 1.25  ->  w <= 26.25  -> widest power of two is 16
    assert fit.free_width(200, 100, eps=0.25) == 16
    assert fit.free_width(200, 100, eps=1.0) == 64
    lens = [50, 100, 400, 80]
    s_static, _ = wave_device_cost(fit, 4, lens, 100, static_batch=True)
    s_cont, _ = wave_device_cost(fit, 4, lens, 100, static_batch=False)
    assert s_cont < s_static
    # a single row: both schedules agree
    a, _ = wave_device_cost(fit, 1, [77], 10, True)
    b, _ = wave_device_cost(fit, 1, [77], 10, False)
    assert a == pytest.approx(b)


def test_spec_from_hybrid_config():
    cfg = {"text_config": {"num_hidden_layers": 24, "num_attention_heads": 8,
                           "num_key_value_heads": 2, "hidden_size": 1024, "head_dim": 256,
                           "layer_types": ["linear_attention"] * 3 * 6 + ["full_attention"] * 6}}
    spec = ModelSpec.from_hf_config("q", cfg, 8e8)
    assert spec.full_layers == 6
    assert spec.kv_bytes_per_token == 2 * 6 * 2 * 256 * 2
    assert spec.linear_state_bytes > 0


# ----------------------------------------------------------------- pareto
def test_pareto_front():
    pts = [(1, 0.5), (2, 0.4), (2, 0.7), (3, 0.7), (4, 0.9), (float("nan"), 1.0)]
    assert pareto_front(pts) == [0, 2, 4]


def test_kendall():
    assert kendall_tau([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert kendall_tau([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


# -------------------------------------------------------------- strategies
def S(key, ok, gen=100, sc=None):
    return Sample(key, ok, gen, sc, None)


def _bank():
    spec_s = ModelSpec("small", 1e9, n_layers=12, n_heads=8, n_kv_heads=2, head_dim=64)
    spec_b = ModelSpec("big", 8e9, n_layers=32, n_heads=32, n_kv_heads=8, head_dim=128)
    fit_s = DeviceFit("small", {"e": 0.01, "d": 1e-6, "a": 1e-3, "b": 1e-5, "g": 1e-9})
    fit_b = DeviceFit("big", {"e": 0.01, "d": 8e-6, "a": 8e-3, "b": 8e-5, "g": 4e-9})
    small = ModelBank("small", spec_s, fit_s, 8)
    big = ModelBank("big", spec_b, fit_b, 4)
    # q_easy: small is unanimous and right. q_hard: small split, big right.
    small.samples = {"q_easy": [S("1", True)] * 32,
                     "q_hard": [S("5", False), S("6", False), S("7", True), S("8", False)] * 8}
    big.samples = {"q_easy": [S("1", True)] * 32, "q_hard": [S("7", True)] * 30 + [S("9", False)] * 2}
    for mb in (small, big):
        mb.prompt_tokens = {"q_easy": 50, "q_hard": 60}
    return Bank({"small": small, "big": big}, ["q_easy", "q_hard"])


def test_vote_and_lead_probability():
    k, c = vote([S("a", 1), S("b", 0), S("a", 1)])
    assert k == "a" and c["a"] == 2
    assert vote([S("x", 0), S("y", 0)])[0] == "x"  # tie -> first seen
    from collections import Counter
    assert lead_probability(Counter(a=8)) > lead_probability(Counter(a=4)) > lead_probability(Counter(a=2, b=2))


def test_cascade_escalates_only_when_needed():
    bank = _bank()
    st = Stream(bank, 0)
    s = Cascade(models=["small", "big"], n_small=8, n_big=4, tau=0.8)
    easy, hard = s.run(bank, st, "q_easy"), s.run(bank, st, "q_hard")
    assert [t for t, _ in easy.waves] == ["small"]
    assert [t for t, _ in hard.waves] == ["small", "big"] and hard.key == "7"


def test_hwave_stops_after_one_wave_when_unanimous():
    bank = _bank()
    st = Stream(bank, 0)
    tr = HWave(models=["small", "big"], tau=0.95, max_waves=2).run(bank, st, "q_easy")
    assert len(tr.waves) == 1 and len(tr.waves[0][1]) == 8 and tr.key == "1"
    tr = HWave(models=["small", "big"], tau=0.95, max_waves=2).run(bank, Stream(bank, 1), "q_hard")
    assert tr.waves[-1][0] == "big" and tr.key == "7"


def test_adaptive_and_sc():
    bank = _bank()
    tr = Adaptive(model="small", tau=0.9, max_n=32).run(bank, Stream(bank, 0), "q_easy")
    assert 2 <= sum(len(s) for _, s in tr.waves) <= 6
    assert all(len(s) == 1 for _, s in tr.waves)
    tr = SC(model="small", n=16).run(bank, Stream(bank, 0), "q_easy")
    assert len(tr.waves) == 1 and tr.key == "1"


def test_proxy_alloc_spends_more_on_disagreement():
    bank = _bank()
    traces = ProxyAlloc(model="small", avg_n=8, probe=4).run_dataset(bank, Stream(bank, 0), bank.qids)
    n = {q: sum(len(s) for _, s in traces[q].waves) for q in bank.qids}
    assert n["q_easy"] == 4 and n["q_hard"] > 4
    assert sum(n.values()) <= 8 * len(bank.qids) + 1


def test_pricing_regimes_disagree():
    """Same samples, parallel vs sequential: equal FLOPs, different device seconds."""
    from freewidth.strategies import SCSeq

    bank = _bank()
    par = price_trace(bank, "q_easy", SC(model="small", n=8).run(bank, Stream(bank, 0), "q_easy"), 295)
    seq = price_trace(bank, "q_easy", SCSeq(model="small", n=8).run(bank, Stream(bank, 0), "q_easy"), 295)
    assert par["flops"] == pytest.approx(seq["flops"])
    assert par["kinetic"] == pytest.approx(seq["kinetic"])
    assert seq["device_s"] > 3 * par["device_s"]
    one = price_trace(bank, "q_easy", Single(model="big").run(bank, Stream(bank, 0), "q_easy"), 295)
    assert math.isfinite(one["device_s"])


# ----------------------------------------------------------- new methods
def test_quorum_wave_cuts_stragglers():
    from freewidth.strategies import quorum_wave

    wave = [S("7", True, 100), S("7", True, 120), S("7", True, 130), S("7", True, 140),
            S("9", False, 900), S("8", False, 1000)]
    key, priced, stopped = quorum_wave(wave, tau=0.9)
    assert stopped and key == "7"
    cut = max(p.gen for p in priced)
    assert cut < 900                                    # long wrong rows never finish
    assert all(p.key is None for p in priced if p.gen == cut and p.correct is False)
    # no quorum possible -> full wave, nothing truncated
    key, priced, stopped = quorum_wave([S("1", 0, 50), S("2", 0, 60), S("3", 1, 70)], tau=0.95)
    assert not stopped and [p.gen for p in priced] == [50, 60, 70]


def test_short_mk_and_conf_filter():
    from freewidth.strategies import ConfFilter, ShortMK

    bank = _bank()
    tr = ShortMK(model="small", k=8, m=1).run(bank, Stream(bank, 0), "q_hard")
    gens = [s.gen for s in tr.waves[0][1]]
    assert len(gens) == 8 and max(gens) == min(gens)  # every row cut at the first finisher
    big = bank.models["big"]
    big.samples["q_hard"] = [S("7", True, 100, sc=2.0)] * 3 + [S("9", False, 100, sc=0.5)] * 5
    tr = ConfFilter(model="big", n=8, keep=0.4).run(bank, Stream(bank, 0), "q_hard")
    assert tr.key == "7"   # majority is wrong, the confident minority wins


def test_pverify_uses_prefill_not_decode():
    from freewidth.strategies import PVerify

    bank = _bank()
    small = bank.models["small"].samples["q_hard"]
    # give sids to the synthetic samples; the verifier separates right from wrong
    bank.models["small"].samples["q_hard"] = [replace_sid(s, i) for i, s in enumerate(small)]
    bank.models["big"].verify["small"] = {"q_hard": {f"0-{i}": ((0.95 if s.correct else 0.05), 400)
                                                      for i, s in enumerate(small)}}
    tr = PVerify(models=["small", "big"], n=8, mode="max").run(bank, Stream(bank, 0), "q_hard")
    assert tr.key == "7"
    assert [t for t, _ in tr.waves] == ["small"]          # big model never decodes
    assert len(tr.prefills) == 1 and tr.prefills[0][0] == "big"
    c = price_trace(bank, "q_hard", tr, 295)
    one_big = price_trace(bank, "q_hard", Single(model="big").run(bank, Stream(bank, 0), "q_hard"), 295)
    assert c["device_s"] < 1.5 * one_big["device_s"]


def replace_sid(s, i):
    from dataclasses import replace

    return replace(s, sid=f"0-{i}")


def test_iso_accuracy_overrun():
    from freewidth.analyze import iso_accuracy_overrun

    rows = [
        {"strategy": "a", "acc": 0.5, "flops": 1.0, "device_s": 10.0},
        {"strategy": "b", "acc": 0.8, "flops": 2.0, "device_s": 50.0},
        {"strategy": "c", "acc": 0.8, "flops": 5.0, "device_s": 5.0},
    ]
    o = iso_accuracy_overrun(rows, "flops", "device_s")
    assert o["max"] == pytest.approx(10.0) and o["worst_strategy"] == "b"
