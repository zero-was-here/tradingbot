"""Figures for docs/llm-desk.md (page key ``llm-desk``).

Each ``@register``-ed ``draw(pal)`` becomes
docs/_assets/figures/llm-desk/<name>-{light,dark}.png.
Render with:  python tools/figures/build.py --only llm-desk

The policy figure evaluates :class:`aurum.agents.DecisionPolicy` itself on a grid (no
re-implementation of its rules); the timeline reads the committed demo journal in
``runs/desk_demo`` (scripted offline client, synthetic anonymised data), and the cost figure
prices illustrative token counts with :class:`aurum.agents.config.ModelPrice`.
No API call is made anywhere.
"""

from __future__ import annotations

import json

import numpy as np
from _common import (
    FIGSIZE,
    register,
    run_dir,
    signed,
    source_note,
)

from aurum.viz.style import figure, style_axes, zero_line

PAGE = "llm-desk"

#: worked example of the page (q = 0.40): (mode, action kwargs, short label)
_EXAMPLES = {
    "overlay": [({"action": "override", "forecast": 0.9}, "override +0.9"),
                ({"action": "override", "forecast": -0.5}, "override \N{MINUS SIGN}0.5"),
                ({"action": "scale", "scale": 0.5}, "scale 0.5")],
    "advisory": [({"action": "veto"}, "veto")],
    "discretionary": [({"action": "override", "forecast": -0.9}, "override \N{MINUS SIGN}0.9")],
}


def _decision(**kw):
    from aurum.agents import Decision

    base = {"action": "follow_quant", "scale": 1.0, "forecast": 0.0, "confidence": 0.6, "horizon_bars": 24,
            "rationale": "figure"}
    base.update(kw)
    return Decision(**base)


@register(PAGE, "policy-modes")
def policy_modes(pal):
    """Final forecast reachable from each quant forecast, per DecisionPolicy mode."""
    from aurum.agents import DecisionPolicy

    q_grid = np.linspace(-1.0, 1.0, 201)
    r_grid = np.linspace(-1.0, 1.0, 201)
    modes = (("overlay", 1.0), ("advisory", 1.0), ("discretionary", 0.5))
    fig, axes = figure(1, 3, figsize=(8.0, 3.5), sharey=True)
    blue, orange = pal["categorical"][0], pal["categorical"][1]
    for ax, (mode, cap) in zip(axes, modes, strict=True):
        pol = DecisionPolicy(mode=mode, max_abs_forecast=cap)
        lo, hi = [], []
        for q in q_grid:
            finals = [pol.evaluate(q, _decision(action="override", forecast=float(r))).final_forecast for r in r_grid]
            lo.append(min(finals))
            hi.append(max(finals))
        lo, hi = np.array(lo), np.array(hi)
        if mode == "advisory":
            ax.plot(q_grid, q_grid, color=blue, linewidth=2.6)
        else:
            ax.fill_between(q_grid, lo, hi, color=blue, alpha=0.28, linewidth=0)
            ax.plot(q_grid, lo, color=blue, linewidth=1.2)
            ax.plot(q_grid, hi, color=blue, linewidth=1.2)
        ax.plot([-1, 1], [-1, 1], color=pal["text_muted"], linewidth=0.8, linestyle=(0, (4, 3)))
        zero_line(ax)
        zero_line(ax, axis="x")
        q0 = 0.40
        for kw, txt in _EXAMPLES[mode]:
            out = pol.evaluate(q0, _decision(**kw))
            req = out.requested_forecast
            if abs(req - out.final_forecast) > 1e-9:
                ax.annotate("", (q0, out.final_forecast), xytext=(q0, req),
                            arrowprops={"arrowstyle": "->", "color": orange, "linewidth": 1.3})
                ax.plot([q0], [req], marker="o", markerfacecolor="none", markeredgecolor=orange, markersize=6,
                        linestyle="none")
            ax.plot([q0], [out.final_forecast], marker="o", color=orange, markeredgecolor=pal["surface"],
                    markersize=7, linestyle="none", zorder=5)
            left = mode == "discretionary"
            ax.annotate(f"{txt} $\\rightarrow$ {signed(out.final_forecast)}", (q0, out.final_forecast),
                        xytext=(-8, 3) if left else (8, 0), textcoords="offset points", fontsize=7.5,
                        color=pal["text"], ha="right" if left else "left", va="bottom" if left else "center",
                        bbox={"facecolor": pal["surface"], "edgecolor": "none", "pad": 0.4, "alpha": 0.85}, zorder=6)
        title = {"overlay": "overlay (default)", "advisory": "advisory",
                 "discretionary": f"discretionary, max_abs {cap:g}"}[mode]
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.set_xlabel("quant forecast q")
    axes[0].set_ylabel("final forecast")
    fig.suptitle("Where DecisionPolicy lets the final forecast go (shaded), for any requested forecast",
                 color=pal["text"], fontsize=11)
    source_note(fig, "Synthetic illustration: DecisionPolicy.evaluate on a 201 x 201 grid of (q, requested "
                "override); points: the worked example below (q = +0.40)", pal)
    return fig


# =========================================================================================
# demo journal timeline
# =========================================================================================
_DATA_TOOLS = {"get_market_snapshot", "get_quant_signals", "get_risk_status", "get_macro_snapshot",
               "get_calendar", "get_backtest_stats", "get_positions"}


@register(PAGE, "desk-timeline", needs=("runs/desk_demo",))
def desk_timeline(pal):
    """Swimlanes of the 51 events of the committed demo cycle (journal line order)."""
    d = run_dir("desk_demo")
    path = sorted(d.glob("*.jsonl"))[0]
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    lanes = ["desk (harness)", "chief", "macro_strategist", "risk_officer", "adhoc:event_risk_analyst"]
    ypos = {name: len(lanes) - 1 - i for i, name in enumerate(lanes)}
    blue, orange, aqua, violet = (pal["categorical"][i] for i in (0, 1, 2, 6))
    style = {
        "turn": ("o", blue, "model turn (llm_response)"),
        "data": ("s", aqua, "data tool call"),
        "spawn": ("D", orange, "consult_specialist / create_specialist"),
        "terminal": ("^", violet, "terminal tool: submit_memo / submit_decision"),
    }
    fig, ax = figure(figsize=(8.0, 4.6))
    style_axes(ax, grid=None)
    for name in lanes:
        ax.axhline(ypos[name], color=pal["grid"], linewidth=0.8, zorder=0)
    seen = set()
    spawn_x, start_x = [], {}
    q = final = None
    for i, e in enumerate(events, start=1):
        ev = e["event"]
        lane = e.get("agent") or "desk (harness)"
        y = ypos.get(lane, ypos["desk (harness)"])
        kind = None
        if ev == "llm_response":
            kind = "turn"
        elif ev == "tool_call":
            tool = e.get("tool") or e.get("name")
            kind = ("data" if tool in _DATA_TOOLS else "spawn" if tool in ("consult_specialist", "create_specialist")
                    else "terminal")
            if kind == "spawn":
                spawn_x.append(i)
        elif ev == "agent_start" and lane != "chief":
            start_x[lane] = i
        if ev == "cycle_start":
            q = float(e["quant_forecast"])
        if ev == "policy":
            final = float(e["outcome"]["final_forecast"])
        if kind:
            m, c, lbl = style[kind]
            ax.plot([i], [y], marker=m, color=c, markeredgecolor=pal["surface"], markersize=8, linestyle="none",
                    label=lbl if kind not in seen else "_nolegend_", zorder=3)
            seen.add(kind)
        elif ev == "agent_created":
            ax.plot([i], [y], marker="*", color=pal["categorical"][4], markeredgecolor=pal["surface"], markersize=13,
                    linestyle="none", zorder=3, label="agent_created (runtime agent)")
            ax.annotate("agent_created", (i, y), xytext=(0, -11), textcoords="offset points", ha="center",
                        va="top", fontsize=7.5, color=pal["text"])
        else:
            ax.plot([i], [y], marker="|", color=pal["text_muted"], markeredgecolor=pal["text_muted"],
                    markeredgewidth=1.2, markersize=10, linestyle="none", zorder=2,
                    label="other journal event" if "other" not in seen else "_nolegend_")
            seen.add("other")
    # delegation arrows: chief's consult/create turn -> each specialist's first event
    if spawn_x:
        for lane, x in start_x.items():
            ax.annotate("", (x, ypos[lane]), xytext=(spawn_x[0], ypos["chief"]),
                        arrowprops={"arrowstyle": "->", "color": pal["text_muted"], "linewidth": 0.8,
                                    "shrinkA": 4, "shrinkB": 4})
    memo_x = [i for i, e in enumerate(events, start=1) if e["event"] == "memo"]
    back_x = [i for i, e in enumerate(events, start=1)
              if e["event"] == "tool_result" and e.get("agent") == "chief"
              and (e.get("tool") or e.get("name")) in ("consult_specialist", "create_specialist")]
    if memo_x and back_x:
        ax.annotate("3 memos return in one message", (back_x[0], ypos["chief"]), xytext=(memo_x[0] - 6, ypos["chief"] + 0.55),
                    fontsize=8, color=pal["text_secondary"], ha="center",
                    arrowprops={"arrowstyle": "->", "color": pal["text_muted"], "linewidth": 0.8})

    def note(event, text, dy=0.42, ha="center"):
        xs = [i for i, e in enumerate(events, start=1) if e["event"] == event]
        if xs:
            ax.annotate(text, (xs[0], ypos["desk (harness)"]), xytext=(0, dy * 40), textcoords="offset points",
                        ha=ha, fontsize=8, color=pal["text"],
                        arrowprops={"arrowstyle": "-", "color": pal["text_muted"], "linewidth": 0.6})

    note("cycle_start", f"cycle_start\nq = {signed(q, '{:+.3f}')}", ha="left")
    note("decision", "decision\nscale 0.5", dy=-0.55, ha="right")
    note("policy", f"policy: final\n{signed(final, '{:+.3f}')}", ha="right")
    ax.set_yticks([ypos[n] for n in lanes], lanes, fontsize=8.5)
    ax.set_xlim(0, len(events) + 2)
    ax.set_ylim(-0.75, len(lanes) - 0.1)
    ax.set_xlabel("journal line (event order of one cycle)")
    ax.set_title(f"One desk cycle from the demo journal: {len(events)} events, 4 agents, quant "
                 f"{signed(q, '{:+.3f}')} $\\rightarrow$ final {signed(final, '{:+.3f}')}", fontsize=10.5)
    from aurum.viz.style import add_legend

    add_legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8)
    source_note(fig, f"Data: runs/desk_demo/{path.name} (aurum desk demo: scripted offline client, synthetic "
                "anonymised bars)", pal)
    return fig


# =========================================================================================
# cost accounting
# =========================================================================================
@register(PAGE, "cost-caching")
def cost_caching(pal):
    """Per-turn cost of one 8-turn agent with and without prompt caching (illustrative tokens)."""
    from aurum.agents.config import CACHE_WRITE_MULTIPLIER, DEFAULT_MODEL, DEFAULT_PRICES

    price = DEFAULT_PRICES[DEFAULT_MODEL]
    prefix, per_turn, out_tok, turns = 6_000, 1_500, 800, 8      # illustrative token counts
    prompt = [prefix + per_turn * k for k in range(turns)]
    no_cache, cached = [], []
    for k, p in enumerate(prompt):
        no_cache.append(price.cost(input_tokens=p, output_tokens=out_tok))
        read = prompt[k - 1] if k else 0                         # the previous request's prompt is cached
        cached.append(price.cost(cache_read_input_tokens=read, cache_creation_input_tokens=p - read,
                                 output_tokens=out_tok, cache_write_multiplier=CACHE_WRITE_MULTIPLIER["5m"]))
    x = np.arange(1, turns + 1)
    fig, ax = figure(figsize=FIGSIZE)
    w = 0.38
    blue, orange = pal["categorical"][0], pal["categorical"][1]
    ax.bar(x - w / 2, no_cache, width=w, color=blue, label=f"no caching: {sum(no_cache):.3f} USD in total")
    ax.bar(x + w / 2, cached, width=w, color=orange,
           label=f"prompt caching, 5m TTL: {sum(cached):.3f} USD in total")
    from aurum.viz.style import add_legend

    add_legend(ax, loc="upper left", fontsize=8.5)
    ax.set_xticks(x, [str(i) for i in x])
    ax.set_xlabel("turn of one agent loop")
    ax.set_ylabel("estimated cost per turn (USD)")
    ax.set_title(f"How usage.py prices a growing conversation ({DEFAULT_MODEL}: {price.input_per_mtok:g} / "
                 f"{price.output_per_mtok:g} USD per M tokens in / out)", fontsize=10)
    source_note(fig, f"Synthetic illustration: ModelPrice.cost with assumed tokens (prefix {prefix:,}, "
                f"+{per_turn:,} per turn, {out_tok} out), not a measured desk cycle", pal)
    return fig
