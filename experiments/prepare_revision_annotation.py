#!/usr/bin/env python3
"""Build a condition-blinded, matched manual-rating package.

The default sample contains two scenarios for every model x split x game
stratum and includes baseline, higher-index, and lower-index responses for each
selected scenario.  Conditions and model identities are kept in a separate key.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import random
from collections import Counter
from pathlib import Path


CRITERIA = (
    "rule_consistency",
    "semantic_coherence",
    "decision_explanation_consistency",
)
CONDITIONS = ("baseline", "full_+1", "full_-1")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = (
        "review_id",
        "game",
        "prompt",
        "response",
        *CRITERIA,
        "notes",
    )
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stable_rank(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def annotation_id(model: str, split: str, source_id: str) -> str:
    return hashlib.sha256(f"{model}\0{split}\0{source_id}".encode()).hexdigest()[:24]


def build_sample(root: Path, per_stratum: int, seed: str):
    public: list[dict[str, str]] = []
    key: list[dict[str, object]] = []
    models = sorted(
        path.name for path in root.iterdir()
        if path.is_dir() and (path / "manifest.json").exists()
    )
    if not models:
        raise RuntimeError(f"No completed model directories found in {root}")

    for model in models:
        splits = sorted((root / model).glob("split_*"))
        if not splits:
            raise RuntimeError(f"No splits found for {model}")
        for split_path in splits:
            split = split_path.name
            validation = split_path / "validation"
            sheet = read_csv(validation / "blind_review.csv")
            source_key = json.loads((validation / "blind_review_key.json").read_text())
            meta = {row["review_id"]: row for row in source_key}
            by_scenario: dict[str, dict[str, dict[str, str]]] = {}
            for row in sheet:
                item_meta = meta[row["review_id"]]
                condition = item_meta["condition"]
                if condition not in CONDITIONS:
                    continue
                enriched = dict(row, condition=condition,
                                scenario_id=item_meta["scenario_id"])
                by_scenario.setdefault(item_meta["scenario_id"], {})[condition] = enriched

            games: dict[str, list[str]] = {}
            for scenario, condition_rows in by_scenario.items():
                if set(condition_rows) != set(CONDITIONS):
                    continue
                game_values = {row["game"] for row in condition_rows.values()}
                if len(game_values) != 1:
                    raise RuntimeError(f"Game mismatch for {model}/{split}/{scenario}")
                games.setdefault(next(iter(game_values)), []).append(scenario)

            for game, scenarios in sorted(games.items()):
                if len(scenarios) < per_stratum:
                    raise RuntimeError(
                        f"Need {per_stratum} matched scenarios in {model}/{split}/{game}; "
                        f"found {len(scenarios)}"
                    )
                chosen = sorted(
                    scenarios,
                    key=lambda value: stable_rank(f"{seed}:{model}:{split}:{game}", value),
                )[:per_stratum]
                for scenario in chosen:
                    cluster_id = f"{model}:{split}:{scenario}"
                    for condition in CONDITIONS:
                        source = by_scenario[scenario][condition]
                        rid = annotation_id(model, split, source["review_id"])
                        public.append({
                            "review_id": rid,
                            "game": source["game"],
                            "prompt": source["prompt"],
                            "response": source["response"],
                            **{criterion: "" for criterion in CRITERIA},
                            "notes": "",
                        })
                        key.append({
                            "review_id": rid,
                            "model": model,
                            "split": split,
                            "condition": condition,
                            "scenario_id": cluster_id,
                            "source_scenario_id": scenario,
                            "source_review_id": source["review_id"],
                        })
    return public, key


def shuffled(rows: list[dict[str, str]], seed: str) -> list[dict[str, str]]:
    result = [dict(row) for row in rows]
    random.Random(seed).shuffle(result)
    return result


def html_document(rows: list[dict[str, str]], rater: str) -> str:
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    title = f"BiRD blinded explanation review — Rater {rater}"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root{{--bg:#f4f6fb;--card:#fff;--ink:#172033;--muted:#64708a;--line:#dce2ee;--accent:#3659d9;--yes:#19764a;--no:#b33a3a}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,sans-serif}}
header{{position:sticky;top:0;z-index:2;background:#172033;color:white;padding:12px 20px;display:flex;gap:18px;align-items:center;flex-wrap:wrap}}
header strong{{font-size:17px}} .progress{{color:#dce5ff}} button,.button{{border:1px solid var(--line);background:white;border-radius:8px;padding:8px 12px;cursor:pointer;font-weight:600}}
header button{{border-color:#7181a4}} main{{max-width:1050px;margin:22px auto;padding:0 18px 40px}} details{{background:#eef2ff;border:1px solid #ccd6ff;border-radius:10px;padding:10px 14px;margin-bottom:16px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;box-shadow:0 5px 20px #24345a12;padding:20px}}
.meta{{display:flex;justify-content:space-between;color:var(--muted);margin-bottom:12px}} h2{{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:18px 0 6px}}
.text{{white-space:pre-wrap;border-left:4px solid #aebce9;padding:10px 14px;background:#f8f9fd;border-radius:4px}} .response{{border-left-color:#65a985;background:#f5faf7}}
.criterion{{display:grid;grid-template-columns:1fr auto;gap:14px;align-items:center;border-top:1px solid var(--line);padding:14px 0}} .criterion:first-of-type{{margin-top:18px}}
.criterion small{{display:block;color:var(--muted)}} .choice{{display:flex;gap:8px}} .choice button{{min-width:58px}} .choice button.selected.yes{{background:var(--yes);color:white;border-color:var(--yes)}} .choice button.selected.no{{background:var(--no);color:white;border-color:var(--no)}}
textarea{{width:100%;min-height:80px;border:1px solid var(--line);border-radius:8px;padding:10px;font:inherit}} nav{{display:flex;justify-content:space-between;gap:10px;margin-top:16px}} .danger{{color:#ffdddd}}
@media(max-width:700px){{.criterion{{grid-template-columns:1fr}}}}
</style></head><body>
<header><strong>{html.escape(title)}</strong><span class="progress" id="progress"></span><button onclick="goUnanswered()">Next unanswered</button><button onclick="exportCsv()">Export CSV</button><button class="danger" onclick="clearSaved()">Clear saved ratings</button></header>
<main><details><summary><b>Rating instructions / Инструкция</b></summary>
<p>Rate every criterion as 1 (passes) or 0 (fails). Judge compliance with the stated game rules, not whether the choice is generous or selfish. Include invalid responses. A missing explanation receives 0 on all three criteria.</p>
<ul><li><b>Rule consistency:</b> the explanation does not contradict payoffs, rounds, constraints, or the partner's possible actions.</li><li><b>Semantic coherence:</b> the response has clear, internally consistent meaning.</li><li><b>Decision–explanation consistency:</b> the explanation supports the decision shown on the first line.</li></ul>
<p>Оценивайте соблюдение правил игры, связность текста и соответствие объяснения указанному решению. Щедрость или эгоистичность решения сама по себе не является ошибкой. Пустое объяснение получает 0 по всем трём критериям.</p></details>
<section class="card"><div class="meta"><span id="itemNo"></span><span id="game"></span></div><h2>Prompt</h2><div class="text" id="prompt"></div><h2>Model response</h2><div class="text response" id="response"></div>
<div id="criteria"></div><h2>Optional notes</h2><textarea id="notes" placeholder="Explain unusual cases or the reason for a 0 rating"></textarea>
<nav><button onclick="move(-1)">← Previous</button><button onclick="move(1)">Next →</button></nav></section></main>
<script>
const rows={payload}; const criteria={json.dumps(list(CRITERIA))}; const labels={{rule_consistency:['Rule consistency','No contradiction with the stated game rules.'],semantic_coherence:['Semantic coherence','Clear and internally consistent meaning.'],decision_explanation_consistency:['Decision–explanation consistency','Explanation supports the first-line decision.']}};
const storageKey='bird-annotation-{rater}-v1'; let saved=JSON.parse(localStorage.getItem(storageKey)||'{{}}'); let index=0;
function persist(){{localStorage.setItem(storageKey,JSON.stringify(saved)); renderProgress()}}
function current(){{return rows[index]}} function state(){{return saved[current().review_id]||(saved[current().review_id]={{}})}}
function setRating(c,v){{state()[c]=String(v);persist();renderCriteria()}}
function renderCriteria(){{const s=state();document.getElementById('criteria').innerHTML=criteria.map(c=>`<div class="criterion"><div><b>${{labels[c][0]}}</b><small>${{labels[c][1]}}</small></div><div class="choice"><button class="no ${{s[c]==='0'?'selected':''}}" onclick="setRating('${{c}}',0)">0 · Fail</button><button class="yes ${{s[c]==='1'?'selected':''}}" onclick="setRating('${{c}}',1)">1 · Pass</button></div></div>`).join('')}}
function complete(row){{const s=saved[row.review_id]||{{}};return criteria.every(c=>s[c]==='0'||s[c]==='1')}}
function renderProgress(){{const done=rows.filter(complete).length;document.getElementById('progress').textContent=`${{done}} / ${{rows.length}} complete`}}
function render(){{const r=current(),s=state();document.getElementById('itemNo').textContent=`Item ${{index+1}} of ${{rows.length}}`;document.getElementById('game').textContent=r.game.replaceAll('_',' ');document.getElementById('prompt').textContent=r.prompt;document.getElementById('response').textContent=r.response||'[EMPTY RESPONSE]';document.getElementById('notes').value=s.notes||'';renderCriteria();renderProgress();window.scrollTo(0,0)}}
function move(delta){{index=Math.max(0,Math.min(rows.length-1,index+delta));render()}}
function goUnanswered(){{const next=rows.findIndex((r,i)=>i>index&&!complete(r));const any=rows.findIndex(r=>!complete(r));index=next>=0?next:(any>=0?any:index);render()}}
document.getElementById('notes').addEventListener('input',e=>{{state().notes=e.target.value;persist()}});
function csvCell(v){{return '"'+String(v??'').replaceAll('"','""')+'"'}}
function exportCsv(){{const fields=['review_id','game','prompt','response',...criteria,'notes'];const lines=[fields.map(csvCell).join(',')];for(const r of rows){{const s=saved[r.review_id]||{{}};lines.push(fields.map(f=>csvCell(criteria.includes(f)||f==='notes'?s[f]:r[f])).join(','))}}const blob=new Blob(['\\ufeff'+lines.join('\\r\\n')],{{type:'text/csv;charset=utf-8'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='ratings_{rater.lower()}.csv';a.click();URL.revokeObjectURL(a.href)}}
function clearSaved(){{if(confirm('Delete all locally saved ratings for this rater?')){{saved={{}};persist();render()}}}} render();
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path,
                        default=Path("runs/revision_measured_20260909"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--per-stratum", type=int, default=2)
    parser.add_argument("--seed", default="bird-reviewer1-manual-v1")
    args = parser.parse_args()
    if args.per_stratum < 1:
        parser.error("--per-stratum must be positive")
    out = args.out or args.run_root / "manual_annotation"
    out.mkdir(parents=True, exist_ok=True)
    public, key = build_sample(args.run_root, args.per_stratum, args.seed)
    orders = {
        "A": shuffled(public, args.seed + ":rater-a"),
        "B": shuffled(public, args.seed + ":rater-b"),
    }
    for rater, rows in orders.items():
        write_csv(out / f"rater_{rater.lower()}.csv", rows)
        (out / f"rater_{rater.lower()}.html").write_text(
            html_document(rows, rater), encoding="utf-8"
        )
    (out / "private_blinding_key.json").write_text(
        json.dumps(key, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    counts = {
        "items": len(public),
        "matched_scenario_clusters": len(public) // len(CONDITIONS),
        "models": dict(Counter(row["model"] for row in key)),
        "splits": dict(Counter(row["split"] for row in key)),
        "conditions": dict(Counter(row["condition"] for row in key)),
        "games": dict(Counter(row["game"] for row in public)),
        "per_model_split_game_scenarios": args.per_stratum,
        "sampling_seed": args.seed,
    }
    (out / "sample_manifest.json").write_text(
        json.dumps(counts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
