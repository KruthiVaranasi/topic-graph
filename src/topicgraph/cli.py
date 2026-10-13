"""Command line: build, search, synth, eval, stats."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

from .embed import get_embedder
from .importers import load_exports
from .llm import get_llm
from .pipeline import PipelineConfig, build_graph, search
from .segment import SegmenterConfig


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _config(args) -> PipelineConfig:
    seg = SegmenterConfig()
    for name in ("low", "mid", "high", "merge"):
        if getattr(args, name, None) is not None:
            setattr(seg, name, getattr(args, name))
    cfg = PipelineConfig(segmenter=seg)
    if args.cluster_threshold is not None:
        cfg.cluster_threshold = args.cluster_threshold
    return cfg


def _add_pipeline_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--embedder", default="hashing", help="hashing (default, no downloads), lsa, st, or st:<model>")
    p.add_argument("--llm", default="none", help="none (default), claude, or claude:<model-id> for boundary checks and labels")
    p.add_argument("--no-llm-judge", action="store_true", help="use the LLM for labels only")
    p.add_argument("--no-llm-labels", action="store_true", help="use the LLM for boundary checks only")
    p.add_argument("--cluster-threshold", type=float, help="min average cosine similarity to merge topics")
    p.add_argument("--low", type=float, help="segmenter: below this score always cut")
    p.add_argument("--mid", type=float, help="segmenter: cut point inside the borderline band when no LLM")
    p.add_argument("--high", type=float, help="segmenter: at or above this score never cut")
    p.add_argument("--merge", type=float, help="segmenter: re-join neighbouring segments at least this similar")


def cmd_build(args) -> None:
    chats = load_exports(args.exports)
    if not chats:
        sys.exit("no chat exports found under the given paths")
    _log(f"imported {len(chats)} chats")
    llm = get_llm(args.llm)
    graph = build_graph(chats, get_embedder(args.embedder), _config(args), llm,
                        not args.no_llm_judge, not args.no_llm_labels, progress=_log)
    out = graph.save(args.out)
    summary = graph.summary()
    if args.judge_sample:
        from .evaluation.judge import judge_sample

        judge_llm = get_llm(args.judge_llm or args.llm) or sys.exit("--judge-sample needs --judge-llm or --llm")
        labels = {n.id: n.label for n in graph.nodes}
        result = judge_sample(graph.segments, labels, judge_llm, args.judge_sample)
        (out / "judge.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        summary["judge"] = {k: v for k, v in result.items() if k != "rows"}
    print(json.dumps(summary, indent=1))
    _log(f"open {out / 'index.html'} in a browser")


def cmd_search(args) -> None:
    data = json.loads(Path(args.graph).read_text(encoding="utf-8"))
    for hit in search(data, args.query, k=args.k):
        when = (hit["timestamp"] or "")[:10]
        print(f"{hit['score']:.2f}  [{hit['source']} {when}] {hit['label']}  — in “{hit['chat_title']}”")
        if hit.get("url"):
            print(f"      {hit['url']}")


def cmd_synth(args) -> None:
    from .synth.export import export
    from .synth.generate import LLMGenerator, LLMTransitions, TemplateGenerator, TemplateTransitions
    from .synth.stitch import build_eval_set

    llm = get_llm(args.llm)
    out = Path(args.out)
    if llm:
        gen = LLMGenerator(llm, cache_path=out / "cache" / "segments.jsonl")
        transitions = LLMTransitions(llm) if not args.template_transitions else TemplateTransitions()
    else:
        gen, transitions = TemplateGenerator(), TemplateTransitions()
    chats = build_eval_set(args.n, gen, seed=args.seed, transitions=transitions, progress=_log,
                           workers=args.workers if llm else 1)
    truth = export(chats, out, seed=args.seed)
    (out / "chats.jsonl").write_text(
        "\n".join(json.dumps({"id": c.id, "tool": c.tool, "case": c.case, "title": c.title, "turns": c.turns},
                             ensure_ascii=False) for c in chats), encoding="utf-8")
    _log(f"wrote {len(truth)} chats ({gen.name}) to {out}")


def cmd_batch(args) -> None:
    from .synth import batch

    if args.step == "plan":
        counts = batch.plan(args.out, args.n, args.seed, args.batch_size)
        _log(f"planned segments per tier: {counts}; job files in {Path(args.out) / 'jobs'}")
    elif args.step == "check":
        _, problems = batch.validate_segments(args.out)
        print("\n".join(problems) or "all segments present and valid")
    elif args.step == "transitions":
        _log(f"wrote {batch.transition_jobs(args.out)} transition jobs")
    else:
        chats, truth = batch.assemble(args.out)
        _log(f"assembled {len(chats)} chats into {Path(args.out) / 'exports'}")


def _run_eval(data_dir: Path, args, cfg: PipelineConfig):
    from .evaluation.metrics import score

    truth = json.loads((data_dir / "truth.json").read_text(encoding="utf-8"))
    if getattr(args, "split", "all") != "all":
        # Deterministic halves by synthetic chat number: tune on one, report on the other.
        want = 0 if args.split == "tune" else 1
        truth = {k: v for k, v in truth.items() if int(v["synth_id"].split("-")[1]) % 2 == want}
    chats = [c for c in load_exports([data_dir / "exports"]) if c.key in truth]
    graph = build_graph(chats, get_embedder(args.embedder), cfg, get_llm(args.llm),
                        not args.no_llm_judge, not args.no_llm_labels)
    return graph, score(graph.segments, truth)


def _print_metrics(m: dict) -> None:
    from .evaluation.metrics import check_targets

    ok = check_targets(m)
    b = m["boundary"]
    mark = lambda k: "PASS" if ok[k] else "MISS"  # noqa: E731
    print(f"chats {m['chats']}  true topics {m['true_topics']}  predicted nodes {m['pred_nodes']}")
    print(f"import check: {m['import']}")
    print(f"topic recall      {m['topic_recall']:.3f}  (target >= 0.90) {mark('topic_recall')}")
    print(f"topic precision   {m['topic_precision']:.3f}  (target >= 0.80) {mark('topic_precision')}")
    print(f"boundary F1       {b['f1']:.3f}  (target >= 0.75) {mark('boundary_f1')}  P {b['precision']:.3f} R {b['recall']:.3f}")
    print(f"hidden recall     {m['hidden_recall']:.3f}  (segments not in the chat's title position)")
    print(f"one-turn recall   {m['one_turn_recall']:.3f}")
    print(f"hidden segment share (truth) {m['truth_hidden_segment_share']:.3f}")
    print("\ncase      chats  boundaryF1  pred/true cuts  recall  one-turn")
    for case, c in m["per_case"].items():
        tiny = "" if c["one_turn_recall"] is None else f"{c['one_turn_recall']:.3f}"
        b = c["boundary"]
        cuts = f"{b['n_pred']}/{b['n_true']}"
        print(f"{case:<9} {c['chats']:>5}  {b['f1']:>10.3f}  {cuts:>14}  {c['topic_recall']:.3f}  {tiny:>8}")
    if m.get("recall_by_tier"):
        print("\ngeneration tier  segments  recall")
        for tier, r in m["recall_by_tier"].items():
            print(f"{tier:<16} {r['segments']:>8}  {r['topic_recall']:.3f}")
    worst = sorted(m["recall_by_topic"].items(), key=lambda kv: kv[1])[:5]
    print("\nlowest-recall topics: " + ", ".join(f"{k} {v:.2f}" for k, v in worst))


def cmd_eval(args) -> None:
    data_dir = Path(args.data)
    if args.sweep:
        rows = []
        grid = lambda s: [float(x) for x in s.split(",")]  # noqa: E731
        merges = [None if x == "none" else float(x) for x in args.sweep_merge.split(",")]
        for low, mid, mg, ct in itertools.product(grid(args.sweep_low), grid(args.sweep_mid), merges, grid(args.sweep_cluster)):
            if mid < low:
                continue
            seg = SegmenterConfig(low=low, mid=mid, high=mid + (mid - low) * 3, merge=mg)
            _, m = _run_eval(data_dir, args, PipelineConfig(segmenter=seg, cluster_threshold=ct))
            rows.append((low, mid, mg, ct, m["topic_recall"], m["topic_precision"], m["boundary"]["f1"]))
            _log(f"low {low:.2f} mid {mid:.2f} merge {mg} cluster {ct:.2f}: recall {m['topic_recall']:.3f} "
                 f"precision {m['topic_precision']:.3f} boundaryF1 {m['boundary']['f1']:.3f}")
        # Recall first, subject to the precision floor.
        ok = [r for r in rows if r[5] >= 0.80]
        if ok:
            best = max(ok, key=lambda r: (r[4], r[6]))
        else:
            _log("no setting reaches the 0.80 precision floor; picking the best recall/precision balance (F1)")
            best = max(rows, key=lambda r: 2 * r[4] * r[5] / ((r[4] + r[5]) or 1))
        merge_flag = f" --merge {best[2]}" if best[2] is not None else ""
        print(f"best (max recall with precision >= 0.80): --low {best[0]} --mid {best[1]}{merge_flag} "
              f"--cluster-threshold {best[3]}  recall {best[4]:.3f} precision {best[5]:.3f} boundaryF1 {best[6]:.3f}")
        return
    graph, m = _run_eval(data_dir, args, _config(args))
    _print_metrics(m)
    if args.out:
        Path(args.out).write_text(json.dumps(m, indent=1), encoding="utf-8")
    if args.graph_out:
        graph.save(args.graph_out)


def cmd_stats(args) -> None:
    from .evaluation.stats import chat_stats

    print(json.dumps(chat_stats(load_exports(args.exports)), indent=1))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="topicgraph", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build", help="import exports and build the topic graph")
    p.add_argument("exports", nargs="+", help="export files or folders (ChatGPT, Claude, Gemini Takeout)")
    p.add_argument("--out", default="out", help="output folder for graph.json and index.html")
    p.add_argument("--judge-sample", type=int, default=0, help="rate N random segments with an LLM judge")
    p.add_argument("--judge-llm", help="LLM for the judge; defaults to --llm")
    _add_pipeline_args(p)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("search", help="find earlier segments about a question before re-asking it")
    p.add_argument("query")
    p.add_argument("--graph", default="out/graph.json")
    p.add_argument("-k", type=int, default=10)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("synth", help="generate the synthetic eval set with ground truth")
    p.add_argument("--out", default="data/synth")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--llm", default="none", help="none = offline templates; claude[:model] = LLM-written chats")
    p.add_argument("--template-transitions", action="store_true", help="cue-phrase transitions even with an LLM")
    p.add_argument("--workers", type=int, default=8, help="parallel LLM calls")
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("batch", help="plan / check / transitions / assemble an externally generated eval set")
    p.add_argument("step", choices=["plan", "check", "transitions", "assemble"])
    p.add_argument("--out", default="data/synth-llm")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=25)
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("eval", help="run the pipeline on a synthetic set and score it")
    p.add_argument("--data", default="data/synth")
    p.add_argument("--out", help="write metrics JSON here")
    p.add_argument("--graph-out", help="also save the graph viewer here")
    p.add_argument("--sweep", action="store_true", help="grid-search thresholds (run without --llm: every point re-runs the pipeline)")
    p.add_argument("--sweep-low", default="0.02,0.04,0.06")
    p.add_argument("--sweep-mid", default="0.04,0.06,0.08,0.10")
    p.add_argument("--sweep-cluster", default="0.06,0.08,0.10,0.14")
    p.add_argument("--sweep-merge", default="none", help="comma list of merge thresholds, or none")
    p.add_argument("--split", choices=["all", "tune", "test"], default="all",
                   help="score on all chats, or on the even (tune) / odd (test) half")
    _add_pipeline_args(p)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("stats", help="shape statistics for the distribution check")
    p.add_argument("exports", nargs="+")
    p.set_defaults(func=cmd_stats)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
