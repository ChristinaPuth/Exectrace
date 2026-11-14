"""Measure the stored leadership of a company against its hand-verified gold set. SPEC 7.2 and 7.4.

No model and no network. Reads the database and fixtures/gold/<domain>.json.

- Recall: of the people the gold set lists as current, how many the pipeline
  has as current, per group (C-level, VP, Head). Measured against the gold
  set only, never against everyone who may exist.
- Reachable-source recall, always shown beside it: the same, leaving out a
  person only when there is clear evidence the pipeline could not read any
  source the gold set cites for them. The evidence is a stored record that
  the pipeline asked for that exact page and the site refused it or its
  robots.txt forbade it. A source never attempted is not evidence, and a
  person found as current is reachable by definition.
- Former check: no person the gold set lists as former may be current.
- Precision: each role the pipeline marks current is correct if the gold set
  lists it as current, wrong if the gold set says otherwise, and otherwise
  left for a person to judge. Unjudged roles are counted apart, never as
  correct.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from .extract import states_title
from .normalize import canonical_role, compare_names, normalize_name

GROUPS = ("c_level", "vp", "head")


def _norm_url(url: str) -> str:
    url = url.strip().lower().rstrip("/")
    for prefix in ("https://", "http://"):
        url = url.removeprefix(prefix)
    return url.removeprefix("www.")


def blocked_pages(conn: sqlite3.Connection, domain: str) -> dict[str, str]:
    """Pages the pipeline asked for and was not allowed to read, with the recorded reason."""
    rows = conn.execute(
        """SELECT s.url, s.last_error FROM sources s JOIN companies c ON c.id = s.company_id
            WHERE c.domain = ? AND s.stage = 'skipped'
              AND (s.last_error LIKE '%robots.txt%' OR s.last_error LIKE '%refused automated access%')""",
        (domain,)).fetchall()
    return {_norm_url(r["url"]): r["last_error"] for r in rows}


def unreachable_because(gold: dict, row: dict, blocked: dict[str, str]) -> str | None:
    """Why no source cited for this gold row could be read, or None when one could (or might)."""
    keys = [row.get("source")] + ([row["also_supported_by"]["source"]] if row.get("also_supported_by") else [])
    urls = [gold.get("sources", {}).get(k, {}).get("url") for k in keys if k]
    if not urls or not all(u and _norm_url(u) in blocked for u in urls):
        return None
    return "; ".join(f"{u}: {blocked[_norm_url(u)]}" for u in urls)


def _same_person(a: str, names: list[str]) -> bool:
    return any(compare_names(normalize_name(a), normalize_name(n)) in ("same", "nickname") for n in names)


def _same_office(gold: dict, role: sqlite3.Row, company: str) -> bool:
    """The pipeline role is the office the gold row names, at the same scope."""
    if gold.get("role"):                                # a company-wide single-holder office
        return role["canonical_role"] == gold["role"] or canonical_role(role["title"], company) == gold["role"]
    return states_title(gold["title"], role["title"]) or states_title(role["title"], gold["title"])


@dataclass
class GoldReport:
    company: str
    recall: dict = field(default_factory=dict)          # group -> {"gold": n, "current": n, "other": n, "missing": [..]}
    former_as_current: list[str] = field(default_factory=list)
    review: list[dict] = field(default_factory=list)    # one row per pipeline-current leader role
    scope_errors: list[str] = field(default_factory=list)

    @property
    def judged(self) -> tuple[int, int, int]:
        correct = sum(r["verdict"] == "correct" for r in self.review)
        wrong = sum(r["verdict"] == "wrong" for r in self.review)
        return correct, wrong, len(self.review) - correct - wrong

    @property
    def precision(self) -> float | None:
        correct, wrong, _ = self.judged
        return correct / (correct + wrong) if correct + wrong else None

    def recall_of(self, group: str) -> float | None:
        r = self.recall.get(group)
        return r["current"] / r["gold"] if r and r["gold"] else None

    def reachable_recall_of(self, group: str) -> float | None:
        r = self.recall.get(group)
        return r["current"] / r["reachable"] if r and r["reachable"] else None


def measure(conn: sqlite3.Connection, gold: dict, verdicts: dict[str, str] | None = None) -> GoldReport:
    """Compare one company's stored roles with its gold set.

    `verdicts` are hand judgments of roles the gold set does not cover, keyed
    by "person | title", each "correct" or "wrong".
    """
    verdicts = verdicts or {}
    company = gold["company"]
    roles = conn.execute(
        """SELECT r.*, p.full_name FROM roles r JOIN people p ON p.id = r.person_id
             JOIN companies c ON c.id = r.company_id WHERE c.domain = ? ORDER BY p.full_name, r.title""",
        (gold["domain"],)).fetchall()
    report = GoldReport(company)

    def names(row: dict) -> list[str]:
        return [row["name"], *row.get("also_written", [])]

    current_gold = [g for g in gold["leaders"] if g["status"] == "current"]
    blocked = blocked_pages(conn, gold["domain"])
    for group in GROUPS:
        people: dict[str, list[dict]] = {}
        for g in current_gold:
            if group in g["counts_toward"]:
                people.setdefault(g["name"], []).append(g)
        found, other, missing, unreachable = 0, [], [], []
        for name, rows in sorted(people.items()):
            mine = [r for r in roles if _same_person(r["full_name"], names(rows[0]))]
            hit = [r for r in mine for g in rows if _same_office(g, r, company)]
            if any(r["status"] == "current" for r in hit):
                found += 1
                continue
            if hit:
                other.append(f"{name} ({', '.join(sorted({r['status'] for r in hit}))})")
            else:
                missing.append(name)
            why = [unreachable_because(gold, g, blocked) for g in rows]
            if all(why):                                # every cited source of every row was refused
                unreachable.append(f"{name}: {why[0]}")
        report.recall[group] = {"gold": len(people), "current": found, "other_status": other, "missing": missing,
                                "unreachable": unreachable, "reachable": len(people) - len(unreachable)}

    for g in gold["leaders"]:
        if g["status"] != "former":
            continue
        for r in roles:
            if r["status"] == "current" and _same_person(r["full_name"], names(g)) and _same_office(g, r, company):
                report.former_as_current.append(f"{g['name']} / {g['title']}")

    secondary = [s["name"] for s in gold.get("secondary_candidates", [])]
    for r in roles:
        if r["status"] != "current" or r["level"] not in GROUPS:
            continue
        key = f"{r['full_name']} | {r['title']}"
        mine = [g for g in gold["leaders"] if _same_person(r["full_name"], names(g))]
        office = [g for g in mine if _same_office(g, r, company)]
        if any(g["status"] == "current" for g in office):
            verdict, basis = "correct", "the gold set lists this office as current"
            g = next(g for g in office if g["status"] == "current")
            if g["scope"] == "subsidiary_or_unit" and r["canonical_role"]:
                report.scope_errors.append(f"{key}: stored as the company-wide {r['canonical_role']}, "
                                           f"but the gold set scopes it to {g['unit']}")
        elif any(g["status"] == "former" for g in office):
            verdict, basis = "wrong", "the gold set lists this office as former"
        elif key in verdicts:
            verdict, basis = verdicts[key], "judged by hand"
        elif _same_person(r["full_name"], secondary):
            verdict, basis = "not counted", "a secondary-source candidate, kept apart from the gold set"
        else:
            verdict, basis = "to review", "not in the gold set: a person must judge it"
        report.review.append({"person": r["full_name"], "title": r["title"], "level": r["level"],
                              "decided_by": r["status_reason"], "verdict": verdict, "basis": basis})
    return report


def load(path: Path) -> dict:
    gold = json.loads(path.read_text(encoding="utf-8"))
    unverified = [g["name"] for g in gold["leaders"] if not g.get("verified")]
    if unverified:
        raise ValueError(f"{path.name}: rows not yet verified by a person: {', '.join(unverified)}")
    return gold


def print_report(report: GoldReport, min_precision: float, min_recall_c_level: float) -> None:
    print(f"\n{report.company}")
    print("  Coverage of the gold set (people the gold set lists as current)")
    print("    Both figures are always shown. 'Reachable' leaves out only people whose every cited source the")
    print("    pipeline asked for and was refused; the full figure is the one the target applies to.")
    print(f"    {'group':<9}{'in gold':>8}{'found current':>15}{'full recall':>13}"
          f"{'reachable':>11}{'reachable recall':>18}{'found, other status':>21}{'not found':>11}")

    def pct(n: int, d: int) -> str:
        return "n/a" if not d else f"{n / d:.0%}"

    for group in GROUPS:
        r = report.recall[group]
        print(f"    {group:<9}{r['gold']:>8}{r['current']:>15}{pct(r['current'], r['gold']):>13}"
              f"{r['reachable']:>11}{pct(r['current'], r['reachable']):>18}"
              f"{len(r['other_status']):>21}{len(r['missing']):>11}")
    for group in GROUPS:
        r = report.recall[group]
        if r["other_status"]:
            print(f"    {group}: found but not current: {'; '.join(r['other_status'])}")
        if r["missing"]:
            print(f"    {group}: NOT FOUND: {', '.join(r['missing'])}")
        if r["gold"] and not r["current"]:
            print(f"    GAP: none of the {r['gold']} {group} people in the gold set was found as current")
        for line in r["unreachable"]:
            print(f"    {group}: unreachable: {line}")
    correct, wrong, open_ = report.judged
    precision = report.precision
    print("  Roles the pipeline marks current (C-level, VP, Head)")
    for row in report.review:
        print(f"    {row['verdict']:<12}{row['person']:<22}{row['title'][:52]:<54}{row['basis']}")
    shown = "n/a" if precision is None else f"{precision:.0%}"
    print(f"  Precision of current roles: {shown}  ({correct} correct, {wrong} wrong, {open_} not judged or not counted)")
    print(f"  Gold-set former leaders reported as current: {len(report.former_as_current)}"
          + (f"  -> {'; '.join(report.former_as_current)}" if report.former_as_current else ""))
    if report.scope_errors:
        print("  Scope errors: " + "; ".join(report.scope_errors))
    c_level = report.recall_of("c_level")
    met_p = precision is not None and precision >= min_precision
    met_r = c_level is not None and c_level >= min_recall_c_level
    print(f"  Target precision >= {min_precision:.0%}: {'met' if met_p else 'NOT MET'}"
          + ("" if not open_ or not met_p else "  (provisional: some roles are not judged)"))
    print(f"  Target C-level recall = {min_recall_c_level:.0%}: {'met' if met_r else 'NOT MET'}")
    print(f"  Target no former leader as current: {'met' if not report.former_as_current else 'NOT MET'}")
