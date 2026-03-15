#!/usr/bin/env python3
"""
analyze.py – Compare two result JSONs (from collect.py) and generate a report.
No HTTP calls. Pure analysis.

Usage:
    python analyze.py \\
        --lg  results/langgraph_analyst_20260221_143000.json \\
        --aws results/strands_analyst_20260221_143500.json

    python analyze.py --lg results/lg_*.json --aws results/aws_*.json   # glob

    # Compare both personas at once:
    python analyze.py \\
        --lg  results/langgraph_analyst_*.json results/langgraph_business_*.json \\
        --aws results/strands_analyst_*.json   results/strands_business_*.json

Output:
    reports/report_<timestamp>.html
    reports/summary_<timestamp>.txt    (CI-friendly pass/fail)
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Check logic
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


def check_step(step: dict, persona: str) -> list[Check]:
    checks: list[Check] = []

    # Always: HTTP 200
    checks.append(Check(
        "HTTP 200",
        step.get("http_status") == 200,
        f"Got {step.get('http_status')}" if step.get("http_status") != 200 else "",
    ))

    if step.get("error"):
        checks.append(Check("No connection error", False, step["error"]))
        return checks

    # No error trace
    checks.append(Check(
        "No error trace",
        not step.get("has_error_trace", False),
        "error-type message in response" if step.get("has_error_trace") else "",
    ))

    # Insight / content
    if not step.get("expect_clarification"):
        checks.append(Check(
            "Insight non-empty",
            bool(step.get("has_insight")),
            "insight/content empty" if not step.get("has_insight") else "",
        ))

    # Plot
    expect_plot = step.get("expect_plot")
    if expect_plot is True:
        checks.append(Check(
            "Plot present",
            bool(step.get("has_plot")),
            "Expected plot, got none" if not step.get("has_plot") else "",
        ))
    elif expect_plot is False:
        checks.append(Check(
            "No spurious plot",
            not step.get("has_plot"),
            "Unexpected plot in response" if step.get("has_plot") else "",
        ))

    # Chart type
    if step.get("expect_chart_type") and step.get("has_plot"):
        expected = step["expect_chart_type"].lower()
        actual = (step.get("plot_type") or "").lower()
        checks.append(Check(
            f"Chart type '{expected}'",
            expected in actual,
            f"Expected '{expected}', got '{actual}'" if expected not in actual else "",
        ))

    # SQL (only meaningful for analyst)
    expect_sql = step.get("expect_sql")
    if expect_sql is True and persona == "analyst":
        checks.append(Check(
            "SQL query captured",
            bool(step.get("has_sql_query")),
            "sql_query is null" if not step.get("has_sql_query") else "",
        ))
    elif expect_sql is False:
        checks.append(Check(
            "No SQL (business/non-data)",
            not step.get("has_sql_query"),
            "Unexpected sql_query" if step.get("has_sql_query") else "",
        ))

    # Clarification
    if step.get("expect_clarification"):
        checks.append(Check(
            "Clarification triggered",
            bool(step.get("is_clarification")),
            "Expected is_clarification=true" if not step.get("is_clarification") else "",
        ))
    else:
        checks.append(Check(
            "No spurious clarification",
            not step.get("is_clarification"),
            "Unexpected clarification flag" if step.get("is_clarification") else "",
        ))

    return checks


def step_passed(step: dict, persona: str) -> bool:
    return all(c.passed for c in check_step(step, persona))


def failed_check_strs(step: dict, persona: str) -> list[str]:
    return [
        f"{c.name}: {c.detail}" if c.detail else c.name
        for c in check_step(step, persona)
        if not c.passed
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Load + merge result files
# ─────────────────────────────────────────────────────────────────────────────

def load_results(paths: list[str]) -> list[dict]:
    """Load one or more result JSON files, returning list of scenario-result dicts."""
    all_scenarios: list[dict] = []
    for pattern in paths:
        for p in sorted(glob.glob(pattern)) or [pattern]:
            try:
                data = json.loads(Path(p).read_text(encoding="utf-8"))
                impl    = data.get("impl", "unknown")
                persona = data.get("persona", "unknown")
                url     = data.get("url", "?")
                ts      = data.get("timestamp", "?")
                for sc in data.get("scenarios", []):
                    sc["_impl"]    = impl
                    sc["_persona"] = persona
                    sc["_url"]     = url
                    sc["_ts"]      = ts
                    all_scenarios.append(sc)
                print(f"  Loaded {len(data.get('scenarios',[]))} scenarios from {p}")
            except Exception as e:
                print(f"  ✗ Failed to load {p}: {e}")
    return all_scenarios


def index_by_id(scenarios: list[dict]) -> dict[str, dict]:
    return {s["scenario_id"]: s for s in scenarios}


# ─────────────────────────────────────────────────────────────────────────────
# Comparison
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PairResult:
    scenario_id: str
    scenario_name: str
    persona: str
    lg: dict | None
    aws: dict | None

    @property
    def lg_steps(self) -> list[dict]:
        return (self.lg or {}).get("steps", [])

    @property
    def aws_steps(self) -> list[dict]:
        return (self.aws or {}).get("steps", [])

    def lg_step_passed(self, i: int) -> bool:
        if i >= len(self.lg_steps): return False
        return step_passed(self.lg_steps[i], self.persona)

    def aws_step_passed(self, i: int) -> bool:
        if i >= len(self.aws_steps): return False
        return step_passed(self.aws_steps[i], self.persona)

    @property
    def lg_passed(self) -> bool:
        return bool(self.lg_steps) and all(
            step_passed(s, self.persona) for s in self.lg_steps
        )

    @property
    def aws_passed(self) -> bool:
        return bool(self.aws_steps) and all(
            step_passed(s, self.persona) for s in self.aws_steps
        )

    @property
    def both_passed(self) -> bool:
        return self.lg_passed and self.aws_passed

    @property
    def parity(self) -> bool:
        """Structural parity: plot/sql/clarification match between implementations."""
        for ls, aws in zip(self.lg_steps, self.aws_steps):
            if ls.get("has_plot") != aws.get("has_plot"):
                return False
            if ls.get("is_clarification") != aws.get("is_clarification"):
                return False
        return True


def build_pairs(lg_scenarios: list[dict], aws_scenarios: list[dict]) -> list[PairResult]:
    lg_idx  = index_by_id(lg_scenarios)
    aws_idx = index_by_id(aws_scenarios)
    all_ids = sorted(set(lg_idx) | set(aws_idx))
    pairs: list[PairResult] = []
    for sid in all_ids:
        lg  = lg_idx.get(sid)
        aws = aws_idx.get(sid)
        persona = (lg or aws or {}).get("_persona", "analyst")
        name    = (lg or aws or {}).get("scenario_name", sid)
        pairs.append(PairResult(
            scenario_id=sid,
            scenario_name=name,
            persona=persona,
            lg=lg,
            aws=aws,
        ))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Token / latency stats
# ─────────────────────────────────────────────────────────────────────────────

def latency_stats(steps: list[dict]) -> dict:
    lats = [s["latency_ms"] for s in steps if s.get("latency_ms")]
    if not lats:
        return {"avg": 0, "min": 0, "max": 0, "total": 0, "count": 0}
    return {
        "avg":   sum(lats) / len(lats),
        "min":   min(lats),
        "max":   max(lats),
        "total": sum(lats),
        "count": len(lats),
    }


def token_stats(steps: list[dict]) -> dict:
    return {
        "input":          sum(s.get("tokens_input",       0) for s in steps),
        "output":         sum(s.get("tokens_output",      0) for s in steps),
        "total":          sum(s.get("tokens_total",       0) for s in steps),
        "cache_read":     sum(s.get("tokens_cache_read",  0) for s in steps),
        "cache_create":   sum(s.get("tokens_cache_create",0) for s in steps),
        "llm_calls":      sum(s.get("llm_calls",          0) for s in steps),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Quality scoring
# ─────────────────────────────────────────────────────────────────────────────

def quality_score(step: dict) -> int:
    """0-10 heuristic quality score for one step response."""
    if not step.get("has_insight") or step.get("q_artifact"):
        return 0
    score = 0
    score += 2 if step.get("q_has_numbers")      else 0   # quantitative
    score += 2 if step.get("q_data_claims", 0) >= 2 else (1 if step.get("q_data_claims") == 1 else 0)
    score += 1 if step.get("q_has_bullets")       else 0   # structured
    score += 1 if step.get("q_has_bold")          else 0   # formatted
    score += 1 if step.get("q_word_count", 0) >= 50 else 0 # complete
    score += 1 if step.get("q_reasoning_limits")  else 0   # honest
    score += 1 if step.get("q_reasoning_scope")   else 0   # scoped
    score += 1 if step.get("q_sql_complexity", 0) >= 3 else 0  # good SQL
    return min(score, 10)


def shared_numbers(s1: dict, s2: dict) -> int:
    """Count numeric tokens appearing in both insights (factual agreement)."""
    import re
    def nums(text):
        return set(re.findall(r"[0-9][0-9,%.]*", text or ""))
    return len(nums(s1.get("insight_full","")) & nums(s2.get("insight_full","")))


def sql_agreement(s1: dict, s2: dict) -> str:
    """High/Medium/Low/N-A agreement between LG and AWS SQL queries."""
    q1 = (s1.get("sql_query") or "").upper().split()
    q2 = (s2.get("sql_query") or "").upper().split()
    if not q1 or not q2:
        return "N/A"
    overlap = len(set(q1) & set(q2)) / max(len(set(q1) | set(q2)), 1)
    if overlap > 0.8: return "High"
    if overlap > 0.5: return "Medium"
    return "Low"


def word_overlap(s1: dict, s2: dict) -> int:
    """Word overlap % between two insight texts (0-100)."""
    w1 = set((s1.get("insight_full") or "").lower().split())
    w2 = set((s2.get("insight_full") or "").lower().split())
    if not w1 or not w2: return 0
    return round(len(w1 & w2) / max(len(w1 | w2), 1) * 100)


# ─────────────────────────────────────────────────────────────────────────────
# HTML Report
# ─────────────────────────────────────────────────────────────────────────────

def badge(ok: bool, label_ok: str = "PASS", label_fail: str = "FAIL") -> str:
    color = "#27ae60" if ok else "#e74c3c"
    label = label_ok if ok else label_fail
    return (f'<span style="background:{color};color:#fff;padding:1px 7px;'
            f'border-radius:3px;font-size:11px;font-weight:bold">{label}</span>')


def parity_cell(lg_val: Any, aws_val: Any) -> str:
    match = lg_val == aws_val
    bg = "" if match else ' style="background:#ffeeba"'
    return f'<td{bg}>{lg_val}</td><td{bg}>{aws_val}</td>'


def generate_html(pairs: list[PairResult], run_ts: str, lg_meta: str, aws_meta: str) -> str:
    total   = len(pairs)
    lg_pass = sum(1 for p in pairs if p.lg_passed)
    aw_pass = sum(1 for p in pairs if p.aws_passed)
    both    = sum(1 for p in pairs if p.both_passed)
    parity  = sum(1 for p in pairs if p.parity)

    all_lg_steps  = [s for p in pairs for s in p.lg_steps]
    all_aws_steps = [s for p in pairs for s in p.aws_steps]
    lg_lat  = latency_stats(all_lg_steps)
    aws_lat = latency_stats(all_aws_steps)
    lg_tok  = token_stats(all_lg_steps)
    aws_tok = token_stats(all_aws_steps)
    tokens_available = lg_tok["total"] > 0 or aws_tok["total"] > 0

    # Plot presence stats
    lg_plot_rate  = sum(1 for s in all_lg_steps if s.get("has_plot")) / max(len(all_lg_steps), 1)
    aws_plot_rate = sum(1 for s in all_aws_steps if s.get("has_plot")) / max(len(all_aws_steps), 1)

    # ── Quality comparison rows ──────────────────────────────────────────────
    quality_rows = ""
    side_by_side_rows = ""
    for p in pairs:
        n_steps = max(len(p.lg_steps), len(p.aws_steps))
        for i in range(n_steps):
            ls  = p.lg_steps[i]  if i < len(p.lg_steps)  else {}
            aws = p.aws_steps[i] if i < len(p.aws_steps) else {}

            lg_q  = quality_score(ls)
            aws_q = quality_score(aws)
            sn    = shared_numbers(ls, aws)
            sqla  = sql_agreement(ls, aws)
            wo    = word_overlap(ls, aws)

            q_color = lambda q: "#27ae60" if q >= 7 else ("#e67e22" if q >= 4 else "#e74c3c")
            sqla_color = "#27ae60" if sqla == "High" else ("#e67e22" if sqla == "Medium" else ("#e74c3c" if sqla == "Low" else "#aaa"))

            art_lg  = '<span style="color:#e74c3c">⚠ Yes</span>' if ls.get("q_artifact")  else "–"
            art_aws = '<span style="color:#e74c3c">⚠ Yes</span>' if aws.get("q_artifact") else "–"

            scenario_td = (
                f'<td rowspan="{n_steps}" style="font-weight:bold;vertical-align:top;font-size:12px">'
                f'{p.scenario_name}<br/><small style="color:#888">{p.persona}</small></td>'
            ) if i == 0 else ""

            quality_rows += f"""<tr>
  {scenario_td}
  <td style="font-size:11px;max-width:180px">{(ls or aws).get("query","")[:70]}</td>
  <td style="color:{q_color(lg_q)};font-weight:bold;text-align:center">{lg_q}/10</td>
  <td style="color:{q_color(aws_q)};font-weight:bold;text-align:center">{aws_q}/10</td>
  <td style="text-align:center">{sn}</td>
  <td style="color:{sqla_color};text-align:center">{sqla}</td>
  <td style="text-align:center">{wo}%</td>
  <td style="text-align:center">{art_lg}</td>
  <td style="text-align:center">{art_aws}</td>
</tr>"""

            def _ins_cell(step: dict) -> str:
                txt = (step.get("insight_full") or step.get("insight_snippet") or "").strip()
                if not txt: return '<em style="color:#aaa">—</em>'
                score = quality_score(step)
                border = "#e74c3c" if step.get("q_artifact") else ("#27ae60" if score >= 7 else "#e67e22")
                escaped = txt.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                return f'<div style="font-size:11px;max-height:120px;overflow:auto;border-left:3px solid {border};padding-left:6px;white-space:pre-wrap">{escaped[:600]}{"…" if len(txt)>600 else ""}</div>'

            def _sql_cell(step: dict) -> str:
                q = (step.get("sql_query") or "").strip()
                if not q: return '<em style="color:#aaa">—</em>'
                escaped = q.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                return f'<code style="font-size:10px;white-space:pre-wrap;word-break:break-all">{escaped[:300]}{"…" if len(q)>300 else ""}</code>'

            scenario_td2 = (
                f'<td rowspan="{n_steps}" style="font-size:11px;font-weight:bold;vertical-align:top">'
                f'{p.scenario_name}<br/><small style="color:#888">{p.persona}</small><br/>'
                f'<small style="color:#666">{(ls or aws).get("label","")}</small></td>'
            ) if i == 0 else ""

            side_by_side_rows += f"""<tr style="border-bottom:2px solid #dee2e6">
  {scenario_td2}
  <td style="vertical-align:top;padding:8px">{_ins_cell(ls)}</td>
  <td style="vertical-align:top;padding:8px">{_ins_cell(aws)}</td>
  <td style="vertical-align:top;padding:4px;font-size:10px">
    <b>LG:</b><br/>{_sql_cell(ls)}<br/><br/><b>AWS:</b><br/>{_sql_cell(aws)}
  </td>
</tr>"""

    # Build per-scenario token breakdown rows
    token_rows = ""
    for p in pairs:
        lg_calls  = sum(s.get("llm_calls",    0) for s in p.lg_steps)
        lg_total  = sum(s.get("tokens_total", 0) for s in p.lg_steps)
        aws_calls = sum(s.get("llm_calls",    0) for s in p.aws_steps)
        aws_total = sum(s.get("tokens_total", 0) for s in p.aws_steps)
        d_calls = aws_calls - lg_calls
        d_tok   = aws_total - lg_total
        calls_bg = ' style="background:#ffeeba"' if (lg_calls and aws_calls and d_calls > 3) else ""
        tok_bg   = ' style="background:#ffeeba"' if (lg_total and aws_total and d_tok > 2000) else ""
        def _fmt(v: int, is_tok: bool = False) -> str:
            if v == 0:
                return '<small style="color:#aaa">–</small>'
            return f"{v:,}" if is_tok else str(v)
        d_calls_str = f"{d_calls:+}" if (lg_calls or aws_calls) else "–"
        d_tok_str   = f"{d_tok:+,}" if (lg_total or aws_total) else "–"
        token_rows += (
            f"<tr><td>{p.scenario_name} "
            f"<small style='color:#888'>({p.persona})</small></td>"
            f"<td>{_fmt(lg_calls)}</td><td>{_fmt(lg_total, True)}</td>"
            f"<td>{_fmt(aws_calls)}</td><td>{_fmt(aws_total, True)}</td>"
            f"<td{calls_bg}>{d_calls_str}</td>"
            f"<td{tok_bg}>{d_tok_str}</td></tr>\n"
        )

    # Build detail rows
    detail_rows = ""
    for p in pairs:
        n_steps = max(len(p.lg_steps), len(p.aws_steps))
        for i in range(n_steps):
            ls  = p.lg_steps[i]  if i < len(p.lg_steps)  else {}
            aws = p.aws_steps[i] if i < len(p.aws_steps) else {}

            lg_ok  = step_passed(ls,  p.persona) if ls  else False
            aw_ok  = step_passed(aws, p.persona) if aws else False

            lg_fail_str  = "; ".join(failed_check_strs(ls,  p.persona)) if ls  else "missing"
            aw_fail_str  = "; ".join(failed_check_strs(aws, p.persona)) if aws else "missing"

            scenario_cell = (
                f'<td rowspan="{n_steps}" style="font-weight:bold;vertical-align:top">'
                f'{p.scenario_name}<br/>'
                f'<small style="color:#888;font-weight:normal">{p.persona}</small></td>'
            ) if i == 0 else ""

            plot_match = ls.get("has_plot") == aws.get("has_plot")
            plot_bg    = "" if plot_match else ' style="background:#fff3cd"'

            lg_plot  = f'✓ ({ls.get("plot_type")  or "?"})' if ls.get("has_plot")  else "–"
            aw_plot  = f'✓ ({aws.get("plot_type") or "?"})' if aws.get("has_plot") else "–"
            lg_sql   = "✓" if ls.get("has_sql_query")  else "–"
            aw_sql   = "✓" if aws.get("has_sql_query") else "–"
            lg_ms    = f'{ls.get("latency_ms", 0):.0f}ms'
            aw_ms    = f'{aws.get("latency_ms", 0):.0f}ms'

            def _tok_cell(s: dict) -> str:
                calls = s.get("llm_calls", 0)
                total = s.get("tokens_total", 0)
                inp   = s.get("tokens_input",  0)
                out   = s.get("tokens_output", 0)
                if calls == 0 and total == 0 and inp == 0:
                    return '<small style="color:#aaa">–</small>'
                tok_str = f"{total:,}" if total else f"{inp:,}+{out:,}"
                return f'<small>{calls}c / {tok_str}t</small>'

            detail_rows += f"""
<tr>
  {scenario_cell}
  <td style="font-size:11px">{(ls or aws).get("label","")}</td>
  <td style="font-size:11px;max-width:220px;word-break:break-word">{(ls or aws).get("query","")[:80]}</td>
  <td>{badge(lg_ok)}<br/><small style="color:#c0392b">{lg_fail_str[:90]}</small></td>
  <td>{badge(aw_ok)}<br/><small style="color:#c0392b">{aw_fail_str[:90]}</small></td>
  <td{plot_bg}>{lg_plot}</td><td{plot_bg}>{aw_plot}</td>
  <td>{lg_sql}</td><td>{aw_sql}</td>
  <td>{lg_ms}</td><td>{aw_ms}</td>
  <td>{_tok_cell(ls)}</td><td>{_tok_cell(aws)}</td>
</tr>"""

    # Parity table rows
    parity_rows = ""
    for p in pairs:
        icon_lg  = "✓" if p.lg_passed  else "✗"
        icon_aws = "✓" if p.aws_passed else "✗"
        icon_par = "✓" if p.parity     else "⚠"
        icon_bot = "✓" if p.both_passed else "✗"
        color_par = "#27ae60" if p.parity else "#e67e22"
        color_bot = "#27ae60" if p.both_passed else "#e74c3c"
        plot_delta = ""
        for ls, aws in zip(p.lg_steps, p.aws_steps):
            if ls.get("has_plot") != aws.get("has_plot"):
                plot_delta += f'Step "{ls.get("label","?")}": LG={ls.get("has_plot")} AWS={aws.get("has_plot")}  '
        parity_rows += (
            f'<tr><td>{p.scenario_name}</td><td>{p.persona}</td>'
            f'<td style="color:{"#27ae60" if p.lg_passed else "#e74c3c"}">{icon_lg}</td>'
            f'<td style="color:{"#27ae60" if p.aws_passed else "#e74c3c"}">{icon_aws}</td>'
            f'<td style="color:{color_bot};font-weight:bold">{icon_bot}</td>'
            f'<td style="color:{color_par}">{icon_par}</td>'
            f'<td style="font-size:11px;color:#e67e22">{plot_delta}</td></tr>\n'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Migration Analysis – {run_ts}</title>
<style>
  body{{font-family:Arial,sans-serif;margin:30px;color:#333;font-size:13px}}
  h1{{color:#2c3e50;font-size:22px}} h2{{color:#34495e;border-bottom:2px solid #ecf0f1;padding-bottom:5px}}
  .grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:16px 0}}
  .card{{background:#f8f9fa;border-radius:6px;padding:14px;text-align:center;border:1px solid #dee2e6}}
  .card .v{{font-size:28px;font-weight:bold}} .card .l{{font-size:12px;color:#666;margin-top:3px}}
  .g{{color:#27ae60}} .r{{color:#e74c3c}} .o{{color:#e67e22}} .b{{color:#2980b9}}
  table{{border-collapse:collapse;width:100%;margin-top:12px}}
  th{{background:#2c3e50;color:#fff;padding:7px 9px;text-align:left;font-size:12px}}
  td{{padding:6px 9px;border-bottom:1px solid #ecf0f1;vertical-align:top}}
  tr:hover{{background:#f5f7fa}}
  .stat-table{{width:680px}}
  footer{{color:#888;font-size:11px;margin-top:36px}}
  .note{{background:#fff8e1;border-left:4px solid #ffc107;padding:10px 14px;margin:12px 0;font-size:12px}}
</style>
</head>
<body>
<h1>🧪 LangGraph → AWS Strands Migration Analysis</h1>
<p><b>Generated:</b> {run_ts}</p>
<p><b>LangGraph data:</b> {lg_meta}<br/><b>AWS Strands data:</b> {aws_meta}</p>

<h2>📊 Executive Summary</h2>
<div class="grid">
  <div class="card"><div class="v">{total}</div><div class="l">Scenarios</div></div>
  <div class="card"><div class="v {'g' if lg_pass==total else 'r'}">{lg_pass}/{total}</div><div class="l">LangGraph PASS</div></div>
  <div class="card"><div class="v {'g' if aw_pass==total else 'r'}">{aw_pass}/{total}</div><div class="l">AWS Strands PASS</div></div>
  <div class="card"><div class="v {'g' if both==total else 'r'}">{both}/{total}</div><div class="l">Both Correct</div></div>
  <div class="card"><div class="v {'g' if parity==total else 'o'}">{parity}/{total}</div><div class="l">Output Parity</div></div>
</div>

<div class="note">
  <b>Output Parity</b> = LangGraph and AWS Strands produce the same plot-present/absent result for every step.
  A scenario can score <b>Both Correct</b> but not <b>Parity</b> if one implementation returns an extra plot
  that wasn't strictly required (but isn't wrong either).
</div>

<h2>⏱ Latency Comparison</h2>
<table class="stat-table">
<tr><th>Metric</th><th>LangGraph</th><th>AWS Strands</th><th>Δ</th></tr>
<tr><td>Avg latency</td><td>{lg_lat['avg']:.0f}ms</td><td>{aws_lat['avg']:.0f}ms</td>
    <td>{aws_lat['avg']-lg_lat['avg']:+.0f}ms</td></tr>
<tr><td>Min latency</td><td>{lg_lat['min']:.0f}ms</td><td>{aws_lat['min']:.0f}ms</td>
    <td>{aws_lat['min']-lg_lat['min']:+.0f}ms</td></tr>
<tr><td>Max latency</td><td>{lg_lat['max']:.0f}ms</td><td>{aws_lat['max']:.0f}ms</td>
    <td>{aws_lat['max']-lg_lat['max']:+.0f}ms</td></tr>
<tr><td>Total wall time</td><td>{lg_lat['total']/1000:.1f}s</td><td>{aws_lat['total']/1000:.1f}s</td>
    <td>{(aws_lat['total']-lg_lat['total'])/1000:+.1f}s</td></tr>
<tr><td>Steps measured</td><td>{lg_lat['count']}</td><td>{aws_lat['count']}</td><td>–</td></tr>
</table>

<h2>🪙 Token Usage</h2>
{'<div class="note">Token data not available. Enable <code>usage</code> field in MessageSummaryDTO. collect.py tries both <code>inputTokens</code>/<code>outputTokens</code> (LangGraph) and <code>input_tokens</code>/<code>output_tokens</code> (Strands) automatically.</div>' if not tokens_available else f"""
<table class="stat-table">
<tr><th>Metric</th><th>LangGraph</th><th>AWS Strands</th><th>Δ</th><th>Δ%</th></tr>
<tr><td>Input tokens</td><td>{lg_tok['input']:,}</td><td>{aws_tok['input']:,}</td>
    <td>{aws_tok['input']-lg_tok['input']:+,}</td>
    <td>{((aws_tok['input']-lg_tok['input'])/max(lg_tok['input'],1)*100):+.0f}%</td></tr>
<tr><td>Output tokens</td><td>{lg_tok['output']:,}</td><td>{aws_tok['output']:,}</td>
    <td>{aws_tok['output']-lg_tok['output']:+,}</td>
    <td>{((aws_tok['output']-lg_tok['output'])/max(lg_tok['output'],1)*100):+.0f}%</td></tr>
<tr><td>Effective total (excl. cached)</td><td>{lg_tok['total']:,}</td><td>{aws_tok['total']:,}</td>
    <td>{aws_tok['total']-lg_tok['total']:+,}</td>
    <td>{((aws_tok['total']-lg_tok['total'])/max(lg_tok['total'],1)*100):+.0f}%</td></tr>
<tr><td>Cache reads</td><td>{lg_tok['cache_read']:,}</td><td>{aws_tok['cache_read']:,}</td>
    <td>{aws_tok['cache_read']-lg_tok['cache_read']:+,}</td><td>–</td></tr>
<tr><td>Cache creation</td><td>{lg_tok['cache_create']:,}</td><td>{aws_tok['cache_create']:,}</td>
    <td>{aws_tok['cache_create']-lg_tok['cache_create']:+,}</td><td>–</td></tr>
<tr><td>LLM API calls</td><td>{lg_tok['llm_calls']}</td><td>{aws_tok['llm_calls']}</td>
    <td>{aws_tok['llm_calls']-lg_tok['llm_calls']:+}</td>
    <td>{((aws_tok['llm_calls']-lg_tok['llm_calls'])/max(lg_tok['llm_calls'],1)*100):+.0f}%</td></tr>
<tr><td>Avg LLM calls / step</td>
    <td>{lg_tok['llm_calls']/max(len(all_lg_steps),1):.1f}</td>
    <td>{aws_tok['llm_calls']/max(len(all_aws_steps),1):.1f}</td>
    <td>–</td><td>–</td></tr>
<tr><td>Avg effective tokens / step</td>
    <td>{lg_tok['total']//max(len(all_lg_steps),1):,}</td>
    <td>{aws_tok['total']//max(len(all_aws_steps),1):,}</td>
    <td>–</td><td>–</td></tr>
</table>
<p><i>Effective total = raw total − cache_creation − cache_read (billing-equivalent tokens).</i></p>
<h3 style='font-size:14px;margin-top:16px'>Per-Scenario Token Breakdown</h3>
<table>
<tr><th>Scenario</th><th>LG calls</th><th>LG total tok</th><th>AWS calls</th><th>AWS total tok</th><th>Δ calls</th><th>Δ tokens</th></tr>
{token_rows}
</table>"""}

<h2>📈 Feature Coverage</h2>
<table class="stat-table">
<tr><th>Feature</th><th>LangGraph</th><th>AWS Strands</th><th>Match?</th></tr>
<tr><td>Plot generation rate</td>
    <td>{lg_plot_rate*100:.0f}% ({sum(1 for s in all_lg_steps if s.get("has_plot"))}/{len(all_lg_steps)} steps)</td>
    <td>{aws_plot_rate*100:.0f}% ({sum(1 for s in all_aws_steps if s.get("has_plot"))}/{len(all_aws_steps)} steps)</td>
    <td>{"✓" if abs(lg_plot_rate-aws_plot_rate)<0.1 else "⚠ Differs"}</td></tr>
<tr><td>SQL captured (analyst)</td>
    <td>{sum(1 for s in all_lg_steps if s.get("has_sql_query"))}/{len([s for s in all_lg_steps])}</td>
    <td>{sum(1 for s in all_aws_steps if s.get("has_sql_query"))}/{len([s for s in all_aws_steps])}</td>
    <td>–</td></tr>
<tr><td>Error traces</td>
    <td>{sum(1 for s in all_lg_steps if s.get("has_error_trace"))}</td>
    <td>{sum(1 for s in all_aws_steps if s.get("has_error_trace"))}</td>
    <td>{"✓" if sum(1 for s in all_aws_steps if s.get("has_error_trace"))==0 else "⚠"}</td></tr>
<tr><td>Clarification triggered</td>
    <td>{sum(1 for s in all_lg_steps if s.get("is_clarification"))}</td>
    <td>{sum(1 for s in all_aws_steps if s.get("is_clarification"))}</td>
    <td>–</td></tr>
</table>

<h2>🔍 Scenario Parity</h2>
<table>
<tr><th>Scenario</th><th>Persona</th><th>LangGraph</th><th>AWS Strands</th><th>Both Correct</th><th>Output Parity</th><th>Plot Δ</th></tr>
{parity_rows}
</table>

<h2>📋 Step-by-Step Results</h2>
<table>
<tr><th>Scenario</th><th>Step</th><th>Query</th>
    <th>LangGraph</th><th>AWS Strands</th>
    <th>LG Plot</th><th>AWS Plot</th>
    <th>LG SQL</th><th>AWS SQL</th>
    <th>LG ms</th><th>AWS ms</th>
    <th>LG calls/tok</th><th>AWS calls/tok</th></tr>
{detail_rows}
</table>

<h2>🏆 Response Quality Comparison</h2>
<div class="note">
  Quality score (0–10) is computed from heuristic signals: quantitative content, data claims,
  structure, formatting, SQL complexity, reasoning scope. It does NOT call an LLM — it is a
  fast proxy for response richness and is most useful when comparing the two implementations
  on the same query. <b>Shared numbers</b> = numeric values appearing in both responses (factual agreement).
</div>
<table>
<tr>
  <th>Scenario / Step</th>
  <th>Query</th>
  <th>LG score</th><th>AWS score</th>
  <th>Shared #s</th>
  <th>SQL agree</th>
  <th>Word overlap</th>
  <th>LG artifact?</th><th>AWS artifact?</th>
</tr>
{quality_rows}
</table>

<h2>📝 Side-by-Side Insights</h2>
<div class="note">Full insight text from both implementations for every step. Useful for spot-checking factual accuracy and tone.</div>
<table>
<tr><th style="width:15%">Scenario / Step</th><th style="width:35%">LangGraph Insight</th><th style="width:35%">AWS Strands Insight</th><th style="width:15%">LG SQL / AWS SQL</th></tr>
{side_by_side_rows}
</table>

<footer>
  <p>Generated by analyze.py — sb3 LangGraph → AWS Strands migration validation — {run_ts}</p>
</footer>
</body>
</html>"""
    return html


# ─────────────────────────────────────────────────────────────────────────────
# Text summary (CI)
# ─────────────────────────────────────────────────────────────────────────────

def generate_summary(pairs: list[PairResult], run_ts: str) -> str:
    lines = [
        f"Migration Analysis: {run_ts}",
        "=" * 60,
        f"  Scenarios  : {len(pairs)}",
        f"  LangGraph  : {sum(1 for p in pairs if p.lg_passed)}/{len(pairs)} PASS",
        f"  AWS Strands: {sum(1 for p in pairs if p.aws_passed)}/{len(pairs)} PASS",
        f"  Both OK    : {sum(1 for p in pairs if p.both_passed)}/{len(pairs)}",
        f"  Parity     : {sum(1 for p in pairs if p.parity)}/{len(pairs)}",
        "",
        "─" * 60,
    ]
    for p in pairs:
        icon = "✓" if p.both_passed else "✗"
        par  = "≡" if p.parity else "≠"
        lines.append(f"{icon}{par} {p.scenario_name} [{p.persona}]")
        for i, (ls, aws) in enumerate(zip(p.lg_steps, p.aws_steps)):
            lg_ok  = step_passed(ls,  p.persona)
            aw_ok  = step_passed(aws, p.persona)
            icon2  = "  ✓" if (lg_ok and aw_ok) else "  ✗"
            lines.append(f"{icon2} Step {i+1}: {ls.get('label','')}")
            for f in failed_check_strs(ls,  p.persona): lines.append(f"      LG  ✗ {f}")
            for f in failed_check_strs(aws, p.persona): lines.append(f"      AWS ✗ {f}")
        lines.append("")

    both  = sum(1 for p in pairs if p.both_passed)
    total = len(pairs)
    lines += [
        "─" * 60,
        f"OVERALL: {'PASSED ✓' if both==total else 'FAILED ✗'}  ({both}/{total})",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze and compare collect.py result files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze.py --lg results/langgraph_analyst_*.json --aws results/strands_analyst_*.json
  python analyze.py --lg results/lg_a.json results/lg_b.json --aws results/aws_a.json results/aws_b.json
        """,
    )
    parser.add_argument("--lg",  nargs="+", required=True, help="LangGraph result JSON file(s) or glob")
    parser.add_argument("--aws", nargs="+", required=True, help="AWS Strands result JSON file(s) or glob")
    parser.add_argument("--out-dir", default="reports",    help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\nLoading LangGraph results:")
    lg_scenarios = load_results(args.lg)
    print("Loading AWS Strands results:")
    aws_scenarios = load_results(args.aws)

    if not lg_scenarios and not aws_scenarios:
        print("No scenarios loaded. Check file paths.")
        return 1

    pairs = build_pairs(lg_scenarios, aws_scenarios)
    print(f"\n{len(pairs)} scenario(s) to compare")

    lg_meta  = " | ".join(sorted({f"{s['_impl']} @ {s['_url']} ({s['_ts']})" for s in lg_scenarios}))
    aws_meta = " | ".join(sorted({f"{s['_impl']} @ {s['_url']} ({s['_ts']})" for s in aws_scenarios}))

    html_path = out_dir / f"report_{ts}.html"
    txt_path  = out_dir / f"summary_{ts}.txt"

    html_path.write_text(generate_html(pairs, ts, lg_meta, aws_meta), encoding="utf-8")
    summary = generate_summary(pairs, ts)
    txt_path.write_text(summary, encoding="utf-8")

    print("\n" + summary)
    print(f"\nHTML report : {html_path}")
    print(f"Text summary: {txt_path}")

    both  = sum(1 for p in pairs if p.both_passed)
    total = len(pairs)
    return 0 if both == total else 1


if __name__ == "__main__":
    sys.exit(main())
