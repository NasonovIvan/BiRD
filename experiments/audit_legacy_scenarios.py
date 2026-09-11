#!/usr/bin/env python3
"""Read-only provenance audit, no inference or statistical re-estimation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def audit(root):
    report={}
    for model in sorted(root.iterdir()):
        final=model/'baseline_paired_records.jsonl'
        extraction=model/'aggression_extraction_decisions.jsonl'
        if not final.exists() or not extraction.exists(): continue
        test,train=rows(final),rows(extraction)
        train_keys={(r['game'],' '.join(r['prompt'].split())) for r in train}
        games={}
        for g in sorted({r['game'] for r in test}):
            rs=[r for r in test if r['game']==g]
            keys={(g,' '.join(r['prompt'].split())) for r in rs}
            games[g]=dict(attempts=len(rs),unique_prompts=len(keys),
                prompts_also_in_extraction=len(keys&train_keys),
                repeats_per_prompt=sorted(Counter(k[1] for k in [(g,' '.join(r['prompt'].split())) for r in rs]).values()))
        report[model.name]=dict(games=games,total_unique_prompts=sum(v['unique_prompts'] for v in games.values()),
          source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (final,extraction)})
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',type=Path,default=Path('plotting/_results'))
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(audit(a.data_root),indent=2)+'\n')
