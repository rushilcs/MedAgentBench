"""``score_model`` (LLM-as-judge) grader for OpenAI RFT on MedAgentBench.

This is the fallback for organizations that have RFT enabled but DO NOT have
access to the ``python`` grader sandbox (which currently returns
``python_grader_server_error_type='unauthorized_error'`` for our org).

The judge sees the model's full text rollout plus the offline reference fields
already present in each ``train.jsonl`` / ``val.jsonl`` row (``task_type``,
``reference_sol``, ``task_params``, ``accepts_empty_finish``) and emits a single
float in ``[0, 1]`` per the rubric below.

Rubric mirrors ``rl_training/rft/medagent_grader.py`` and
``src/server/tasks/medagentbench/refsol.py``:

* ``1.0`` = fully correct (FINISH matches reference AND any required POST(s)
  are correctly structured)
* ``0.4`` = FINISH matches reference but POST(s) missing/wrong on a
  POST-required task
* ``0.0`` = anything else
"""

from __future__ import annotations

from typing import Any

# Score-model graders only accept pinned snapshots (validate endpoint will
# reject "gpt-4o-mini" with grader_error). Allowed list as of 2025-05:
#   gpt-4o-2024-08-06, gpt-4o-mini-2024-07-18,
#   gpt-4.1-2025-04-14, gpt-4.1-mini-2025-04-14, gpt-4.1-nano-2025-04-14,
#   o1-2024-12-17, o3-mini-2025-01-31, o3-2025-04-16, o4-mini-2025-04-16.
#
# We default to gpt-4.1-mini because gpt-4o-mini consistently misreads task 8
# (orthopedic consult) and a few partial-credit cases on this rubric, even at
# temperature=0 with a fixed seed. gpt-4.1-mini hits 93/93 on the smoke suite
# (rl_training/rft/test_score_grader.py) at ~10x lower cost than gpt-4.1.
JUDGE_DEFAULT_MODEL = "gpt-4.1-mini-2025-04-14"

# The judge prompt is intentionally long and explicit. Reasoning models will
# happily wing it on under-specified rubrics; fast cheap judges like
# gpt-4o-mini need every rule spelled out.
_JUDGE_SYSTEM_PROMPT = """You are a strict, deterministic grader for MedAgentBench FHIR-agent rollouts.

You receive the model's full text output and the offline reference data for a
single task. Apply the rules for the given task_type EXACTLY. Do not be
lenient. Do not invent fields the model did not output. Do not give partial
credit for partial POST correctness.

THINK BEFORE SCORING. First walk through the rubric step-by-step in plain text
(parse posts, parse FINISH, check each required field for the task_type). Be
brief and methodical. Then on the FINAL line emit ONLY a single tag of the
form `<score>X</score>` where X is one of `0.0`, `0.4`, or `1.0`.

The only valid scores are:

  1.0   fully correct (FINISH matches reference AND every required POST is
        correctly structured)
  0.4   FINISH matches reference but a required POST is missing or wrong
        (only applicable to tasks 5, 9, 10 when must_post / needs_order is true;
        for tasks 3 and 8 there is NO partial credit, score 0.0 or 1.0 only)
  0.0   anything else (FINISH wrong, extra POSTs on lookup tasks, malformed
        output, etc.)

The very last line of your response MUST match the regex `<score>(0\\.0|0\\.4|1\\.0)</score>`.
Do not put anything after the closing tag.

PARSING THE MODEL OUTPUT
========================
1. Strip surrounding markdown code fences (```...```) if present.
2. Find every `POST <url>\\n<json body>` block (URL on its own line, JSON body
   on the following lines, body ends at the next POST or FINISH).
3. Find the final `FINISH(<json list>)` call. Parse the JSON list inside.
4. Treat the model output as INVALID (score 0.0) if no parseable
   `FINISH([...])` exists.
5. Ignore any narrative text the model added before/after — the model is
   supposed to output only POST blocks and a single final FINISH.

FINISH MATCHING
===============
- "FINISH matches reference" iff the parsed JSON list equals `reference_sol`
  element-for-element (numbers compared with the per-task tolerance noted
  below; strings compared exactly).
- If `reference_sol` itself is the empty list `[]`, then the model output
  `FINISH([])` MATCHES the reference. This is the normal case for tasks 3
  and 8.
- The flag `accepts_empty_finish` is ADDITIVE ONLY. When true, `FINISH([])`
  is ALSO accepted as a valid match in addition to the reference. When
  false, it adds no extra constraint — a FINISH that exactly equals the
  reference is ALWAYS accepted regardless of this flag.
- Concretely: an output of `FINISH([])` is accepted iff EITHER
  `reference_sol == []` OR `accepts_empty_finish` is true.
- For task 6 (numeric average): the FINISH list must be a single number within
  tolerance (default 0.1) of `reference_sol[0]`.
- For task 10 with `last_value` present: reference_sol = [last_value, last_time];
  FINISH must contain the same numeric value (exact) and the same ISO timestamp
  string. `FINISH([])` is also acceptable iff `accepts_empty_finish` is true.

PER-TASK RULES
==============
The variable `cutoff` below is the fixed clinical "now" `2023-11-13T10:15:00+00:00`.

TASK 1  (find MRN / "Patient not found")
  - No POSTs allowed.
  - FINISH list must equal reference_sol exactly.

TASK 2  (count of resources / age)
  - No POSTs allowed.
  - FINISH list must equal reference_sol exactly.

TASK 3  (record vital sign — BP)
  - Exactly one POST whose URL ends with `/Observation`.
  - Body must include ALL of:
      resourceType="Observation"
      category[0].coding[0] == {system:"http://hl7.org/fhir/observation-category",
                                code:"vital-signs", display:"Vital Signs"}
      code == {text:"BP"}
      effectiveDateTime == "2023-11-13T10:15:00+00:00"
      status=="final"
      valueString=="118/77 mmHg"
      subject == {reference: "Patient/<task_params.mrn>"}
  - Then FINISH that matches reference_sol. For task 3 reference_sol is `[]`,
    so the expected FINISH is `FINISH([])`.
  - Score 1.0 iff POST and FINISH are both correct; else 0.0.

TASK 4  (lookup last value within window — magnesium)
  - No POSTs allowed.
  - FINISH list must equal reference_sol exactly.

TASK 5  (conditional magnesium order)
  - If task_params.must_post is FALSE:
      - No POSTs allowed.
      - FINISH must match reference_sol (or `FINISH([])` since
        accepts_empty_finish=true).
      - Score 1.0 if FINISH ok, else 0.0.
  - If task_params.must_post is TRUE:
      - Exactly one POST whose URL ends with `/MedicationRequest` with body:
          resourceType="MedicationRequest"
          medicationCodeableConcept.coding[0] == {system:"http://hl7.org/fhir/sid/ndc",
                                                  code:"0338-1715-40"}
          authoredOn contains the substring "2023-11-13T10:15"
          dosageInstruction[0].route == "IV"
          dosageInstruction[0].doseAndRate[0].doseQuantity == {value:DOSE, unit:"g"}
          dosageInstruction[0].doseAndRate[0].rateQuantity == {value:RATE, unit:"h"}
            where (DOSE, RATE) are determined by task_params.last_value:
              last_value < 1.0   -> (4, 4)
              last_value < 1.5   -> (2, 2)
              else               -> (1, 1)
          status=="active", intent=="order"
          subject == {reference: "Patient/<task_params.mrn>"}
      - Then FINISH that matches reference_sol (or `FINISH([])`).
      - Score 1.0 iff POST AND FINISH both correct.
      - Score 0.4 iff FINISH correct but POST missing/wrong.
      - Otherwise 0.0.

TASK 6  (numeric mean within window — glucose)
  - No POSTs allowed.
  - FINISH must be a single-element list whose number is within tolerance
    (default 0.1) of reference_sol[0].

TASK 7  (lookup last glucose value, no window)
  - No POSTs allowed.
  - FINISH list must equal reference_sol exactly.

TASK 8  (place orthopedic consult)
  - Exactly one POST whose URL ends with `/ServiceRequest` with body:
      resourceType="ServiceRequest"
      code.coding[0] == {system:"http://snomed.info/sct", code:"306181000000106"}
      authoredOn == "2023-11-13T10:15:00+00:00"
      status=="active", intent=="order", priority=="stat"
      subject == {reference: "Patient/<task_params.mrn>"}
      note.text contains the substring:
        "Situation: acute left knee injury, Background: radiology report indicates ACL tear. Assessment: ACL tear grade II. Recommendation: request for Orthopedic service to evaluate and provide management recommendations."
  - Then FINISH that matches reference_sol. For task 8 reference_sol is `[]`,
    so the expected FINISH is `FINISH([])`.
  - Score 1.0 iff POST and FINISH both correct; else 0.0.

TASK 9  (potassium replacement order)
  - If task_params.must_post is FALSE:
      - No POSTs allowed.
      - FINISH must match reference_sol (or `FINISH([])`).
      - Score 1.0 if FINISH ok, else 0.0.
  - If task_params.must_post is TRUE:
      - EXACTLY TWO POSTs IN THIS ORDER:
        POST 1 — URL ends with `/MedicationRequest`, body:
          resourceType="MedicationRequest"
          medicationCodeableConcept.coding[0] == {system:"http://hl7.org/fhir/sid/ndc",
                                                  code:"40032-917-01"}
          authoredOn contains "2023-11-13T10:15"
          dosageInstruction[0].route lower-stripped == "oral"
          dosageInstruction[0].doseAndRate[0].doseQuantity ==
              {value: round((3.5 - last_value) / 0.1 * 10, 1) (within 0.1), unit: "mEq"}
          status=="active", intent=="order"
          subject == {reference: "Patient/<task_params.mrn>"}
        POST 2 — URL ends with `/ServiceRequest`, body:
          resourceType="ServiceRequest"
          code.coding[0] == {system:"http://loinc.org", code:"2823-3"}
          authoredOn == "2023-11-13T10:15:00+00:00"
          status=="active", intent=="order", priority=="stat"
          subject == {reference: "Patient/<task_params.mrn>"}
          occurrenceDateTime contains "2023-11-14T08:"
      - Then FINISH that matches reference_sol (or `FINISH([])`).
      - Score 1.0 iff BOTH POSTs and FINISH all correct.
      - Score 0.4 iff FINISH correct but POSTs missing/wrong.
      - Otherwise 0.0.

TASK 10 (HbA1C order if stale or missing)
  - If task_params.needs_order is FALSE:
      - No POSTs allowed.
      - FINISH must match reference_sol (or `FINISH([])`).
      - Score 1.0 if FINISH ok, else 0.0.
  - If task_params.needs_order is TRUE:
      - Exactly one POST whose URL ends with `/ServiceRequest` with body:
          resourceType="ServiceRequest"
          code.coding[0] == {system:"http://loinc.org", code:"4548-4"}
          authoredOn == "2023-11-13T10:15:00+00:00"
          status=="active", intent=="order", priority=="stat"
          subject == {reference: "Patient/<task_params.mrn>"}
      - Then FINISH that matches reference_sol (or `FINISH([])`).
      - Score 1.0 iff POST and FINISH both correct.
      - Score 0.4 iff FINISH correct but POST missing/wrong.
      - Otherwise 0.0.

ABSOLUTE RULES
==============
- An extra POST on a lookup-only task (1, 2, 4, 6, 7) → score 0.0.
- Wrong number of POSTs on a POST task → POST is "wrong" (so score 0.4 if
  FINISH ok and partial credit applies, else 0.0).
- A missing or unparseable FINISH → score 0.0 regardless of any POSTs.
- Reply with one of `0.0`, `0.4`, `1.0`. Nothing else."""


_JUDGE_USER_TEMPLATE = """task_type: {{ item.task_type }}
reference_sol: {{ item.reference_sol }}
task_params: {{ item.task_params }}
accepts_empty_finish: {{ item.accepts_empty_finish }}

MODEL OUTPUT (verbatim, between the markers):
<<<MODEL_OUTPUT_BEGIN>>>
{{ sample.output_text }}
<<<MODEL_OUTPUT_END>>>"""


def build_score_model_grader(
    *,
    name: str = "medagent_judge",
    judge_model: str = JUDGE_DEFAULT_MODEL,
    judge_temperature: float = 0.0,
    judge_seed: int | None = 0,
    max_completion_tokens: int = 512,
) -> dict[str, Any]:
    """Build the RFT ``score_model`` grader config.

    The judge model is prompted with the rubric above and must emit ONE float
    in {0.0, 0.4, 1.0}. RFT clamps to [0, 1] via the default ``range``.

    ``gpt-4o-mini`` is the cheapest competent judge for this task; switch to a
    stronger model (gpt-4.1-mini, gpt-4o) if reward hacking shows up.
    """
    sampling_params: dict[str, Any] = {
        "temperature": judge_temperature,
        "max_completions_tokens": max_completion_tokens,
    }
    if judge_seed is not None:
        sampling_params["seed"] = judge_seed

    return {
        "type": "score_model",
        "name": name,
        "model": judge_model,
        "range": [0.0, 1.0],
        "input": [
            {
                "role": "system",
                "type": "message",
                "content": _JUDGE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "type": "message",
                "content": _JUDGE_USER_TEMPLATE,
            },
        ],
        "sampling_params": sampling_params,
    }
