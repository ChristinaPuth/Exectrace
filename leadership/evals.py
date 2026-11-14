"""`leadership eval`: measured quality on saved, hand-labelled material. SPEC section 7.2.

M3 measures extraction: precision and recall of (person, title) pairs on five
real pages in fixtures/labelled/. The real model is called; answers are
cached, so a repeat costs nothing unless the prompt, schema or model changed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .cache import FileCache
from .config import ROOT, Config
from .extract import Target, extract_document
from .llm import LLM
from .normalize import canonical_role, normalize_name
from .runlog import RunLog

LABELLED = ROOT / "fixtures" / "labelled"


def _words(title: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", title.casefold())) - {"of", "the", "and", "our", "s"}


def same_title(expected: str, found: str, company: str) -> bool:
    """The label names this title: the same single-holder role, or all of the label's words."""
    a, b = canonical_role(expected, company), canonical_role(found, company)
    if a or b:
        return a == b
    return _words(expected) <= _words(found)


@dataclass
class PageScore:
    file: str
    expected: int = 0
    found: int = 0                  # extracted pairs, not counting optional ones
    matched: int = 0
    missed: list[str] = field(default_factory=list)
    wrong: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    detail_errors: list[str] = field(default_factory=list)
    calls: int = 0
    rejected: int = 0


def score_page(page: dict, accepted: list, company: str) -> PageScore:
    """Compare what extraction kept with the hand labels of one page."""
    score = PageScore(page["file"], expected=len(page["expected"]))
    pairs: dict[tuple[str, str], list] = {}
    for a in accepted:
        if a.claim.claim_type in ("role", "appointment", "departure") and a.claim.raw_title:
            pairs.setdefault((normalize_name(a.claim.person), a.claim.raw_title), []).append(a.claim)
    used: set[tuple[str, str]] = set()

    def find(label: dict):
        for (person, title), claims in pairs.items():
            if person == normalize_name(label["person"]) and same_title(label["title"], title, company):
                used.add((person, title))
                yield from claims

    for label in page["expected"]:
        claims = list(find(label))
        name = f"{label['person']} / {label['title']}"
        if not claims:
            score.missed.append(name)
            continue
        score.matched += 1
        for key in ("claim_type", "departure_scope", "effective_date"):
            if key in label and not any(getattr(c, key) == label[key] for c in claims):
                got = sorted({str(getattr(c, key)) for c in claims})
                score.detail_errors.append(f"{name}: {key} should be {label[key]}, got {', '.join(got)}")
    for label in page.get("optional", []):
        list(find(label))                               # correct if present; never counted
    for person, title in pairs:
        if (person, title) not in used:
            score.wrong.append(f"{person} / {title}")
    forbidden = {normalize_name(n) for n in page.get("forbidden_people", [])}
    score.forbidden = sorted({p for p, _ in pairs if p in forbidden})
    score.found = score.matched + len(score.wrong)
    return score


def run_extraction_eval(cfg: Config, root: Path = ROOT) -> int:
    labels = json.loads((LABELLED / "labels.json").read_text(encoding="utf-8"))
    llm = LLM(cfg, "extract")
    log = RunLog(root / "logs" / "eval-extraction.jsonl")
    cache = FileCache(root / "cache" / "llm")
    scores: list[PageScore] = []
    print(f"Extraction accuracy on {len(labels['pages'])} hand-labelled pages (labelled {labels['labelled_on']})")
    print(f"Models: {', '.join(model for _, model in llm.chain)}\n")
    for page in labels["pages"]:
        c = labels["companies"][page["company"]]
        target = Target(c["name"], set(c["domains"]), c["description"])
        text = (LABELLED / page["file"]).read_text(encoding="utf-8")
        found = extract_document(llm, target, text, source_type=page["source_type"],
                                 doc_date=page["published"], url=page["url"],
                                 max_chunks=cfg.limits.max_chunks_per_page, cache=cache, log=log)
        s = score_page(page, found.accepted, c["name"])
        s.calls, s.rejected = found.calls, len(found.rejected)
        scores.append(s)
        print(f"{page['file']}\n    expected {s.expected}, found {s.found}, matched {s.matched}; "
              f"{s.rejected} refused by the checks; {s.calls} model call(s)")
        for kind, items in (("MISSED", s.missed), ("NOT IN THE LABELS", s.wrong),
                            ("PERSON FROM ANOTHER ORGANISATION", s.forbidden),
                            ("WRONG DETAIL", s.detail_errors)):
            for item in items:
                print(f"    {kind}: {item}")
    expected, found, matched = (sum(getattr(s, k) for s in scores) for k in ("expected", "found", "matched"))
    precision = matched / found if found else 0.0
    recall = matched / expected if expected else 0.0
    forbidden = sum(len(s.forbidden) for s in scores)
    details = sum(len(s.detail_errors) for s in scores)
    print(f"\nPairs: {expected} expected, {found} found, {matched} matched")
    print(f"Precision {precision:.0%}   Recall {recall:.0%}")
    print(f"People from another organisation stored as leaders: {forbidden}")
    print(f"Labelled claim types and effective dates that came out wrong: {details}")
    print("\nThis measures five pages, not the whole pipeline. Recall is against the labels only.")
    return 0


# ------------------------------------------------------------------ gold sets
def run_gold_eval(cfg: Config, root: Path = ROOT) -> int:
    """Measure the exported fixtures against the hand-verified gold sets. No model, no network.

    Reads fixtures/leadership.db (written by `leadership export`), or the
    working database when no fixture exists yet. For each company it also
    writes fixtures/gold/<domain>_review.json: one row per role the pipeline
    marks current, for a person to judge the ones the gold set does not cover.
    """
    from . import gold as gold_module
    from .db import connect

    fixture, working = root / "fixtures" / "leadership.db", root / "data" / "leadership.db"
    source = fixture if fixture.exists() else working
    print(f"\nGold-set measurement, from {source.relative_to(root)}")
    conn = connect(source)
    for path in sorted((root / "fixtures" / "gold").glob("*.json")):
        if path.name.endswith("_review.json"):
            continue
        gold = gold_module.load(path)
        review_path = path.with_name(path.stem + "_review.json")
        verdicts: dict[str, str] = {}
        notes: dict[str, str] = {}                      # who judged a role, and on what basis
        if review_path.exists():
            for row in json.loads(review_path.read_text(encoding="utf-8"))["roles"]:
                key = f"{row['person']} | {row['title']}"
                if row.get("hand_verdict") in ("correct", "wrong"):
                    verdicts[key] = row["hand_verdict"]
                if row.get("hand_note"):
                    notes[key] = row["hand_note"]
        report = gold_module.measure(conn, gold, verdicts)
        gold_module.print_report(report, cfg.eval.min_precision_current_roles, cfg.eval.min_recall_gold_c_level)
        review_path.write_text(json.dumps({
            "about": "One row per role the pipeline marks current (C-level, VP, Head). 'verdict' is decided from "
                     "the gold set where it can be. For rows marked 'to review', a person sets 'hand_verdict' to "
                     "'correct' or 'wrong' and runs `leadership eval` again.",
            "company": gold["company"],
            "roles": [{**row, "hand_verdict": verdicts.get(f"{row['person']} | {row['title']}"),
                       **({"hand_note": notes[k]} if (k := f"{row['person']} | {row['title']}") in notes else {})}
                      for row in report.review]}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8", newline="\n")
    conn.close()
    print("\nRecall is measured against the gold sets only, not against everyone who may exist.")
    print("A target that is not met is reported as it is; the gold sets are not changed to pass.")
    return 0
