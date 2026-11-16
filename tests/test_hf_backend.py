"""Runs only where torch + transformers are installed (e.g. Kaggle). Builds a tiny random
model locally, so no download is needed."""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    d = tmp_path_factory.mktemp("tiny")
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel()
    tok.train_from_iterator(["the answer is \\boxed{42} Yes No </think> compute 1 + 2 = 3"] * 50,
                            trainers.BpeTrainer(vocab_size=400, special_tokens=[
                                "<unk>", "<|im_start|>", "<|im_end|>", "<pad>"]))
    ft = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", eos_token="<|im_end|>",
                                 pad_token="<pad>")
    ft.chat_template = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
                        "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    ft.save_pretrained(d)
    cfg = Qwen3Config(vocab_size=len(ft), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      eos_token_id=ft.eos_token_id, pad_token_id=ft.pad_token_id)
    Qwen3ForCausalLM(cfg).save_pretrained(d)
    return str(d)


def test_generate_verify_and_budget_forcing(tiny):
    from freewidth.backends import make_backend

    b = make_backend({"backend": "hf", "tag": "t", "model_id": tiny, "device": "cpu", "dtype": "float32"})
    r = b.generate("s", "Compute 1 + 2.", 3, temperature=0.8, top_p=1.0, max_new_tokens=12, seed=0)
    assert len(r.texts) == 3 and all(0 < L <= 12 for L in r.gen_tokens)
    assert all(x is not None for x in r.self_certainty)
    sc = b.score_yes("s", ["Is 1+2=3? Answer Yes or No.", "Is 1+2=4? Answer Yes or No."])
    assert len(sc) == 2 and all(0.0 <= p <= 1.0 and n > 0 for p, n in sc)
    s, _ = b.timed_decode(2, 8, 16)
    assert s > 0
    b.close()

    t = make_backend({"backend": "hf", "tag": "t", "model_id": tiny, "device": "cpu", "dtype": "float32",
                      "enable_thinking": True, "answer_tokens": 6})
    r = t.generate("s", "Compute 1 + 2.", 2, temperature=0.8, top_p=1.0, max_new_tokens=5, seed=0)
    forced = [i for i, f in enumerate(r.finish) if f == "forced"]
    for i in forced:
        assert "Final answer:" in r.texts[i] and r.gen_tokens[i] > 5


def test_live_quorum_criteria(tiny):
    from transformers import AutoTokenizer

    from freewidth.engine.runtime import QuorumState, _QuorumCriteria

    tok = AutoTokenizer.from_pretrained(tiny)
    eos = tok.eos_token_id
    ans = tok("the answer is \\boxed{42}", add_special_tokens=False)["input_ids"]
    P = 3
    ids = torch.tensor([[5] * P + ans + [eos], [5] * P + ans + [eos], [5] * P + ans + [7]])
    crit = _QuorumCriteria(tok, P, {eos}, QuorumState(0.8))
    assert bool(crit(ids, None).all()) and len(crit.finished) == 2
    assert not bool(_QuorumCriteria(tok, P, {eos}, QuorumState(0.95))(ids, None).any())


def test_live_engine_on_tiny_model(tiny, tmp_path):
    from freewidth.engine import Engine, EngineConfig

    cfg = EngineConfig.model_validate({
        "ladder": [{"tag": "a", "backend": "hf", "model_id": tiny, "device": "cpu", "dtype": "float32"},
                   {"tag": "b", "backend": "hf", "model_id": tiny, "device": "cpu", "dtype": "float32",
                    "replica_devices": ["cpu"]}],
        "sampling": {"max_new_tokens": 12},
        "calibration": {"dir": str(tmp_path / "c"), "widths": [1, 2, 4], "gen_lens": [1, 4, 8],
                        "prompt_lens": [8, 32]},
        "policy": {"max_waves": 1, "verify_accept": 0.0},
    })
    e = Engine(cfg)
    e.load()
    for pol in ("single", "sc", "hwave", "freewidth"):
        r = e.solve("Compute 1 + 2.", overrides={"name": pol}, seed=0)
        assert r.waves and r.tokens > 0
    e.close()


def test_overlap_live_threads_on_tiny_model(tiny, tmp_path):
    """Real transformers path: controller criteria, verifier thread, speculative big wave."""
    from freewidth.engine import Engine, EngineConfig

    cfg = EngineConfig.model_validate({
        "ladder": [{"tag": "a", "backend": "hf", "model_id": tiny, "device": "cpu", "dtype": "float32"},
                   {"tag": "b", "backend": "hf", "model_id": tiny, "device": "cpu", "dtype": "float32"}],
        "sampling": {"max_new_tokens": 16},
        "calibration": {"dir": str(tmp_path / "c"), "widths": [1, 2, 4], "gen_lens": [1, 4, 8],
                        "prompt_lens": [8, 32]},
        "policy": {"max_waves": 1, "runaway_min_finished": 1, "trigger_min_finished": 1,
                   "verify_wait_s": 2.0},
    })
    e = Engine(cfg)
    e.load()
    for seed in range(3):
        r = e.solve("Compute 1 + 2.", overrides={"name": "overlap"}, seed=seed)
        assert r.wall_seconds is not None and r.waves
        assert r.notes.get("accept_reason") in {"quorum", "verified", "quorum_end", "verified_end",
                                                "big_model", "fallback"}
    e.close()


def test_read_locate_fork_on_tiny_model(tiny):
    """One pass gives verdict + per-token flaw; fork continues from the corrected token."""
    from freewidth.backends import make_backend
    from freewidth.engine.repair import hf_fork_wave, hf_read, shared_tokenizer
    from freewidth.engine.runtime import QuorumState

    b = make_backend({"backend": "hf", "tag": "b", "model_id": tiny, "device": "cpu", "dtype": "float32"})
    assert shared_tokenizer(b, b)
    r = hf_read(b, "s", "Compute 1 + 2.", "the answer is 1 + 2 = 3 so \\boxed{3} compute step",
                flaw_p=0.9, flaw_conf=0.0, warmup_tokens=1)   # random model: force a flaw to exercise fork
    assert 0.0 <= r.p_yes <= 1.0 and r.tokens > len(r.solution_ids) > 0
    assert r.flaw is not None and r.fix_token is not None and r.flaw >= 1
    w = hf_fork_wave(b, r, 3, temperature=0.8, top_p=1.0, max_new_tokens=20, seed=0, quorum=QuorumState(0.9))
    assert len(w.rows) == 3
    head = b.tok.decode(r.solution_ids[: r.flaw] + [r.fix_token], skip_special_tokens=True)
    assert all(row.text.startswith(head) for row in w.rows)
    b.close()


def test_repair_and_big_sc_policies_on_tiny_model(tiny, tmp_path):
    from freewidth.engine import Engine, EngineConfig

    cfg = EngineConfig.model_validate({
        "ladder": [{"tag": "a", "backend": "hf", "model_id": tiny, "device": "cpu", "dtype": "float32"},
                   {"tag": "b", "backend": "hf", "model_id": tiny, "device": "cpu", "dtype": "float32",
                    "replica_devices": ["cpu"]}],
        "sampling": {"max_new_tokens": 12},
        "calibration": {"dir": str(tmp_path / "c"), "widths": [1, 2, 4], "gen_lens": [1, 4, 8],
                        "prompt_lens": [8, 32]},
        "policy": {"max_waves": 1, "flaw_p": 0.9, "flaw_conf": 0.001, "free_form_fallback": False},
    })
    e = Engine(cfg)
    e.load()
    for pol in ("repair", "big_sc", "big_sc2"):
        r = e.solve("Compute 1 + 2.", overrides={"name": pol, "sc_n": 2}, seed=1)
        assert r.waves
    assert [w.width for w in r.waves] == [1, 1] and r.wall_seconds is not None
    e.close()
