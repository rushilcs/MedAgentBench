#!/usr/bin/env python3
"""Deep failure analysis on a trajectories.jsonl from run_post_train_eval.

Goal: for every wrong rollout, classify the failure into one of:

  A. PARSE_ERROR              - model gave a sensible answer but format/JSON broke the grader
  B. PROSE_IN_LIST            - model put natural-language text instead of a numeric/struct value
  C. WRONG_FORMAT             - missing FINISH, FINISH not last, FINISH not a JSON list, etc.
  D. SEARCH_TYPO              - patient lookup with name typo / wrong field
  E. STALE_OR_NON_LATEST      - answered with an older value when the latest was required
  F. OFF_BY_ONE_DATE          - age / temporal off by 1 (DOB edge case)
  G. WINDOW_FILTERING         - did not filter to last 24h before averaging / picking
  H. FABRICATION              - returned a number when ref says no record exists (-1)
  I. GAVE_UP                  - returned -1 or [] when ref has a real value
  J. WRONG_BRANCH_DECISION    - ordered when no order needed, or vice versa (task5/9/10)
  K. MISSING_POST             - branch required POST but model finished without one
  L. WRONG_POST_PAYLOAD       - POST present but payload mismatched a field check
  M. WRONG_POST_URL           - POST hit wrong endpoint
  N. EXTRA_POST               - made a POST when none was allowed
  O. NO_LAB_VALUE_RETRIEVED   - model never fetched the relevant Observation series
  P. OTHER                    - anything that doesn't fit the above

For each task type we report counts per category and 3-5 illustrative examples.

This is read-only: it consumes trajectories.jsonl + the per-task ground-truth
(case_data['sol']) and the eval.log printed lines (for ref_sol where the
ground truth depends on FHIR data, e.g. age, last value, 24h average).
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

EVAL_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "rl_training/outputs/qwen3_32b_sft_eval"
)


# ---------- helpers ----------

REF_LINE_RX = re.compile(
    r"^(task\d+_\d+)\s+(\[[^\n]*?\])\s+(.+)$"
)


def parse_eval_log(path: Path) -> dict[str, str]:
    """Return {task_id: ref_sol_repr} from refsol prints in eval.log."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        m = REF_LINE_RX.match(line.rstrip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


def load_trajectories(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def extract_finish(turns: list[dict]) -> tuple[str | None, str | None]:
    """Return (raw_finish_string, parsed_json_inside_FINISH_or_None)."""
    if not turns:
        return None, None
    last = turns[-1]
    if last.get("role") not in ("agent", "assistant"):
        return None, None
    txt = last.get("content", "") or ""
    m = re.search(r"FINISH\(\s*(\[[\s\S]*\])\s*\)", txt)
    if not m:
        return txt, None
    inside = m.group(1)
    try:
        return inside, json.dumps(json.loads(inside))
    except Exception:
        return inside, None


def collect_get_urls(turns: list[dict]) -> list[str]:
    urls = []
    for t in turns:
        if t.get("role") in ("agent", "assistant"):
            for line in (t.get("content") or "").splitlines():
                line = line.strip()
                if line.startswith("GET "):
                    urls.append(line[4:].strip())
    return urls


def collect_posts(turns: list[dict]) -> list[tuple[str, dict | None]]:
    """Return list of (url, parsed_payload) for accepted POSTs (heuristic)."""
    posts = []
    n = len(turns)
    for i, t in enumerate(turns):
        if t.get("role") not in ("agent", "assistant"):
            continue
        c = (t.get("content") or "")
        if "POST " not in c:
            continue
        m = re.search(r"POST\s+(\S+)\s*\n([\s\S]+)", c)
        if not m:
            posts.append((c.strip().splitlines()[0][5:].strip(), None))
            continue
        url = m.group(1).strip()
        body = m.group(2).strip()
        # body may be JSON followed by trailing text; greedy json parse
        payload = None
        try:
            payload = json.loads(body)
        except Exception:
            # try to find closing brace
            try:
                end = body.rfind("}")
                payload = json.loads(body[: end + 1])
            except Exception:
                payload = None
        posts.append((url, payload))
    return posts


def get_finish_list(parsed_json_str: str | None):
    if parsed_json_str is None:
        return None
    try:
        return json.loads(parsed_json_str)
    except Exception:
        return None


# ---------- per-task classifiers ----------

def is_pure_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def parse_ref_list(ref_repr: str | None):
    if not ref_repr:
        return None
    try:
        return json.loads(ref_repr)
    except Exception:
        return None


def classify(task_type: str, traj: dict, ref_repr: str | None) -> tuple[str, str]:
    """Return (category, short_explanation)."""
    turns = traj.get("turns", [])
    finish_raw, finish_json = extract_finish(turns)
    finish_list = get_finish_list(finish_json)
    posts = collect_posts(turns)
    gets = collect_get_urls(turns)
    ref_list = parse_ref_list(ref_repr)

    # 1) Format/finish issues
    if finish_raw is None:
        return "WRONG_FORMAT", "no FINISH() emitted"
    if finish_json is None:
        return "PARSE_ERROR", f"FINISH not JSON-parseable: {finish_raw[:80]}"
    if not isinstance(finish_list, list):
        return "WRONG_FORMAT", "FINISH not a JSON list"

    # 2) Prose-in-list (model put NL text where a number was expected)
    if any(isinstance(x, str) for x in finish_list) and task_type in {
        "task2", "task4", "task5", "task6", "task7", "task9", "task10"
    }:
        return "PROSE_IN_LIST", repr(finish_list)[:120]

    # 3) Per-task semantic checks
    sol = (traj.get("task_data") or {}).get("sol")

    if task_type == "task1":
        # find patient by name+DOB; sol is [MRN] or ["Patient not found"]
        # If model said "Patient not found" but ref is an MRN -> probably search typo
        if finish_list == ["Patient not found"] and isinstance(sol, list) and sol and sol[0] != "Patient not found":
            # check GETs for typos vs. instruction
            instr = (traj.get("task_data") or {}).get("instruction", "")
            return "SEARCH_TYPO", f"got 0 hits; gets={gets[:2]} instr_excerpt={instr[:100]}"
        if isinstance(sol, list) and finish_list != sol:
            return "OTHER", f"finish={finish_list} sol={sol}"
        return "OK", ""

    if task_type == "task2":
        # age: ref_list is [age]
        if ref_list and isinstance(ref_list, list) and len(ref_list) == 1 and finish_list and len(finish_list) == 1:
            try:
                d = abs(int(ref_list[0]) - int(finish_list[0]))
                if d <= 2:
                    return "OFF_BY_ONE_DATE", f"got {finish_list[0]} ref {ref_list[0]}"
                return "OTHER", f"big age diff {d}"
            except Exception:
                return "OTHER", f"finish={finish_list} ref={ref_list}"
        return "OTHER", f"finish={finish_list} ref={ref_list}"

    if task_type == "task3":
        # POST Observation BP for patient. Always check_has_post then payload.
        if not posts:
            return "MISSING_POST", "no POST emitted"
        if len(posts) > 1:
            return "EXTRA_POST", f"{len(posts)} POSTs"
        url, payload = posts[0]
        if not url.endswith("/Observation"):
            return "WRONG_POST_URL", url
        if payload is None:
            return "PARSE_ERROR", "POST body not JSON"
        # check the canonical fields
        try:
            if payload.get("resourceType") != "Observation":
                return "WRONG_POST_PAYLOAD", "resourceType"
            if payload.get("status") != "final":
                return "WRONG_POST_PAYLOAD", f"status={payload.get('status')}"
            if payload.get("effectiveDateTime") != "2023-11-13T10:15:00+00:00":
                return "WRONG_POST_PAYLOAD", f"effDT={payload.get('effectiveDateTime')}"
            if payload.get("valueString") != "118/77 mmHg":
                return "WRONG_POST_PAYLOAD", f"valueString={payload.get('valueString')}"
            if payload.get("code") != {"text": "BP"}:
                return "WRONG_POST_PAYLOAD", f"code={payload.get('code')}"
            cat = payload.get("category", [])
            if not (isinstance(cat, list) and len(cat) == 1):
                return "WRONG_POST_PAYLOAD", "category shape"
            cod = cat[0].get("coding", [])
            if not (len(cod) == 1 and cod[0].get("code") == "vital-signs"):
                return "WRONG_POST_PAYLOAD", f"category coding={cod}"
            return "OK_BUT_GRADED_WRONG", "all canonical fields look OK"
        except Exception as e:
            return "WRONG_POST_PAYLOAD", f"exception {e}"

    if task_type == "task4":
        # last MG within 24h; -1 if none
        if ref_list and finish_list:
            r0 = ref_list[0] if ref_list else None
            f0 = finish_list[0] if finish_list else None
            if r0 == -1 and f0 != -1:
                return "FABRICATION", f"got {f0}, ref -1"
            if f0 == -1 and r0 != -1:
                return "GAVE_UP", f"ref {r0}, got -1"
            if isinstance(r0, (int, float)) and isinstance(f0, (int, float)):
                d = abs(r0 - f0)
                if d > 0.01:
                    return "WRONG_VALUE", f"got {f0}, ref {r0}"
                return "OK_NUMERIC_TOL", f"diff {d}"
        return "OTHER", f"finish={finish_list} ref={ref_list}"

    if task_type == "task5":
        # last MG; if low (<=1.9) order Mg replacement; else don't
        # ref_list is [last_value or -1]; allowed FINISH = ref OR []
        r0 = ref_list[0] if ref_list else None
        f0 = finish_list[0] if finish_list else None

        # branch decisions
        if r0 == -1 or (isinstance(r0, (int, float)) and r0 > 1.9):
            # NO order expected
            if posts:
                return "EXTRA_POST", f"ordered when not needed; posts={[u for u,_ in posts]}"
        else:
            # order expected
            if not posts:
                return "MISSING_POST", "low MG but no Mg replacement POST"
            url, payload = posts[0]
            if not url.endswith("/MedicationRequest"):
                return "WRONG_POST_URL", url
            if payload is None:
                return "PARSE_ERROR", "POST body not JSON"
            try:
                if payload.get("medicationCodeableConcept", {}).get("coding", [{}])[0].get("code") != "0338-1715-40":
                    return "WRONG_POST_PAYLOAD", "wrong NDC code"
                if r0 < 1:
                    dose, rate = 4, 4
                elif r0 < 1.5:
                    dose, rate = 2, 2
                else:
                    dose, rate = 1, 1
                got_dose = payload.get("dosageInstruction", [{}])[0].get("doseAndRate", [{}])[0].get("doseQuantity", {})
                if got_dose != {"value": dose, "unit": "g"}:
                    return "WRONG_POST_PAYLOAD", f"dose got {got_dose} expected {dose}g"
            except Exception as e:
                return "WRONG_POST_PAYLOAD", f"exc {e}"
        # final list value check
        if not (finish_list == [] or finish_list == ref_list):
            if isinstance(f0, str):
                return "PROSE_IN_LIST", repr(finish_list)[:120]
            return "WRONG_VALUE", f"finish={finish_list} ref={ref_list}"
        return "OK_BUT_GRADED_WRONG", "passed our checks"

    if task_type == "task6":
        # 24h GLU average
        r0 = ref_list[0] if ref_list else None
        f0 = finish_list[0] if finish_list else None
        if r0 == -1 and f0 != -1:
            return "FABRICATION", f"got {f0}, ref -1"
        if f0 == -1 and r0 != -1:
            return "GAVE_UP", f"ref {r0}, got -1"
        if isinstance(r0, (int, float)) and isinstance(f0, (int, float)):
            d = abs(r0 - f0)
            if d < 0.1:
                return "OK_NUMERIC_TOL", f"diff {d}"
            # check if model used wrong window vs picked latest only etc.
            return "WINDOW_FILTERING", f"got {f0}, ref {r0}, diff {d:.2f}"
        return "OTHER", f"finish={finish_list} ref={ref_list}"

    if task_type == "task7":
        # latest GLU value (no time filter)
        r0 = ref_list[0] if ref_list else None
        f0 = finish_list[0] if finish_list else None
        if isinstance(r0, (int, float)) and isinstance(f0, (int, float)):
            if abs(r0 - f0) < 0.001:
                return "OK_NUMERIC_TOL", "match"
            return "STALE_OR_NON_LATEST", f"got {f0}, latest {r0}"
        return "OTHER", f"finish={finish_list} ref={ref_list}"

    if task_type == "task8":
        # POST ServiceRequest for ortho consult
        if not posts:
            return "MISSING_POST", "no POST"
        if len(posts) > 1:
            return "EXTRA_POST", f"{len(posts)} POSTs"
        url, payload = posts[0]
        if not url.endswith("/ServiceRequest"):
            return "WRONG_POST_URL", url
        if payload is None:
            return "PARSE_ERROR", "POST body not JSON"
        try:
            if payload.get("code", {}).get("coding", [{}])[0].get("code") != "306181000000106":
                return "WRONG_POST_PAYLOAD", f"code={payload.get('code')}"
            if payload.get("authoredOn") != "2023-11-13T10:15:00+00:00":
                return "WRONG_POST_PAYLOAD", f"authoredOn={payload.get('authoredOn')}"
            if payload.get("priority") != "stat":
                return "WRONG_POST_PAYLOAD", f"priority={payload.get('priority')}"
            note = (payload.get("note") or {}).get("text", "")
            if "ACL" not in note:
                return "WRONG_POST_PAYLOAD", "note text"
            return "OK_BUT_GRADED_WRONG", "all canonical fields look OK"
        except Exception as e:
            return "WRONG_POST_PAYLOAD", f"exc {e}"

    if task_type == "task9":
        # latest K; if low (<3.5) order K replacement + serum K stat lab
        r0 = ref_list[0] if ref_list else None
        f0 = finish_list[0] if finish_list else None
        if r0 is None or r0 == -1 or (isinstance(r0, (int, float)) and r0 >= 3.5):
            if posts:
                return "EXTRA_POST", f"ordered when not needed; posts={[u for u,_ in posts]}"
            # finish must equal ref or []
            if not (finish_list == [] or finish_list == ref_list):
                return "WRONG_VALUE", f"finish={finish_list} ref={ref_list}"
            return "OK_BUT_GRADED_WRONG", "no-order branch passes"
        # K low: needs 2 POSTs (MedicationRequest then ServiceRequest)
        if len(posts) < 2:
            return "MISSING_POST", f"need 2 POSTs, got {len(posts)}"
        if len(posts) > 2:
            return "EXTRA_POST", f"got {len(posts)} POSTs"
        # first must be MedicationRequest
        url1, p1 = posts[0]
        if not url1.endswith("/MedicationRequest"):
            return "WRONG_POST_URL", url1
        if p1 is None:
            return "PARSE_ERROR", "POST body not JSON"
        try:
            if p1.get("medicationCodeableConcept", {}).get("coding", [{}])[0].get("code") != "40032-917-01":
                return "WRONG_POST_PAYLOAD", "wrong NDC code"
            dose = (3.5 - r0) / 0.1 * 10
            got = p1.get("dosageInstruction", [{}])[0].get("doseAndRate", [{}])[0].get("doseQuantity", {})
            if abs(got.get("value", 0) - dose) > 0.1:
                return "WRONG_POST_PAYLOAD", f"dose {got.get('value')} expected {dose}"
            if got.get("unit") != "mEq":
                return "WRONG_POST_PAYLOAD", f"unit {got.get('unit')}"
        except Exception as e:
            return "WRONG_POST_PAYLOAD", f"exc {e}"
        # second must be ServiceRequest with LOINC 2823-3 + occurrence next morning
        url2, p2 = posts[1]
        if not url2.endswith("/ServiceRequest"):
            return "WRONG_POST_URL", url2
        if p2 is None:
            return "PARSE_ERROR", "second POST body not JSON"
        try:
            if p2.get("code", {}).get("coding", [{}])[0].get("code") != "2823-3":
                return "WRONG_POST_PAYLOAD", "second LOINC code wrong"
            if "2023-11-14T08:" not in p2.get("occurrenceDateTime", ""):
                return "WRONG_POST_PAYLOAD", f"occurrence={p2.get('occurrenceDateTime')}"
        except Exception as e:
            return "WRONG_POST_PAYLOAD", f"exc {e}"
        # final list value
        if not (finish_list == [] or finish_list == ref_list):
            if isinstance(f0, str):
                return "PROSE_IN_LIST", repr(finish_list)[:120]
            return "WRONG_VALUE", f"finish={finish_list} ref={ref_list}"
        return "OK_BUT_GRADED_WRONG", "all checks pass"

    if task_type == "task10":
        # latest A1C (value, time); if missing or older than 1y -> POST stat A1C ServiceRequest
        # ref_list is [-1] or [value, "iso_time"]
        if ref_list is None:
            return "OTHER", f"no ref_list"
        if len(ref_list) == 1 and ref_list[0] == -1:
            need_post = True
        else:
            try:
                # parse last_meas
                from datetime import datetime
                lm = datetime.fromisoformat(ref_list[1])
                need_post = lm < datetime.fromisoformat("2022-11-13T10:15:00+00:00")
            except Exception:
                need_post = False

        if need_post:
            if not posts:
                return "MISSING_POST", "needed stat A1C order"
            url, payload = posts[0]
            if not url.endswith("/ServiceRequest"):
                return "WRONG_POST_URL", url
            if payload is None:
                return "PARSE_ERROR", "POST body not JSON"
            try:
                if payload.get("code", {}).get("coding", [{}])[0].get("code") != "4548-4":
                    return "WRONG_POST_PAYLOAD", "wrong A1C LOINC"
                if payload.get("priority") != "stat":
                    return "WRONG_POST_PAYLOAD", "priority not stat"
            except Exception as e:
                return "WRONG_POST_PAYLOAD", f"exc {e}"
        else:
            if posts:
                return "EXTRA_POST", "ordered when recent A1C exists"

        # final list value check
        if finish_list == [] or finish_list == ref_list:
            return "OK_BUT_GRADED_WRONG", "all checks pass"
        # Check whether numeric matches but time is off
        try:
            if (
                len(finish_list) == 2
                and len(ref_list) == 2
                and abs(float(finish_list[0]) - float(ref_list[0])) < 0.001
                and finish_list[1] != ref_list[1]
            ):
                return "TIME_FORMAT_MISMATCH", f"got {finish_list[1]} ref {ref_list[1]}"
        except Exception:
            pass
        return "WRONG_VALUE", f"finish={finish_list} ref={ref_list}"

    return "OTHER", "unhandled"


# ---------- main ----------

def main():
    log_refs = parse_eval_log(EVAL_DIR / "eval.log")
    trajs = load_trajectories(EVAL_DIR / "trajectories.jsonl")

    per_task: dict[str, dict] = defaultdict(lambda: {
        "total": 0,
        "correct": 0,
        "wrong": 0,
        "cats": Counter(),
        "examples": defaultdict(list),
    })

    for traj in trajs:
        tid = traj["task_id"]
        ttype = tid.split("_", 1)[0]
        rec = per_task[ttype]
        rec["total"] += 1
        if traj.get("correct"):
            rec["correct"] += 1
            continue
        rec["wrong"] += 1
        cat, expl = classify(ttype, traj, log_refs.get(tid))
        rec["cats"][cat] += 1
        if len(rec["examples"][cat]) < 3:
            rec["examples"][cat].append((tid, expl))

    # Print
    total = sum(r["total"] for r in per_task.values())
    correct = sum(r["correct"] for r in per_task.values())
    print(f"==== Overall: {correct}/{total} = {100*correct/total:.1f}% SR ====\n")

    order = sorted(per_task.keys(), key=lambda s: int(s.replace("task", "")))
    for ttype in order:
        r = per_task[ttype]
        sr = 100 * r["correct"] / r["total"]
        print(f"==== {ttype}: {r['correct']}/{r['total']} ({sr:.1f}%) — {r['wrong']} wrong ====")
        for cat, n in r["cats"].most_common():
            print(f"    {n:>3}  {cat}")
            for tid, expl in r["examples"][cat][:3]:
                print(f"        {tid}: {expl}")
        print()


if __name__ == "__main__":
    main()
