---
name: cpu-only-ml-solutions
description: Plan, implement, validate, and manage ML challenge solutions and challenge-creation packages. Use when working on challenge pages, datasets under dataset/public, solution.ipynb notebooks, submission.csv validators, leaderboard/credit strategy, or creating datasets, prepare.py, grade.py, rubrics, and reference solutions.
---

# Competition workflow

Operate as a competition engineer: produce valid, compliant, competitive ML submissions quickly, with git-backed project isolation and strict quality checks.

## Project Rules

- Use git for every project. Keep work in a private GitHub repository unless the user explicitly says otherwise.
- Give each challenge its own directory at the repository root, e.g. natural-product-evidence-matching/.
- Use one Codex thread per challenge project. Stay inside the assigned challenge directory unless reading shared repo tooling or the user asks to coordinate across projects.
- Treat official solver notebooks as Kaggle-style submissions: read from ./dataset/public/, write final output to ./working/submission.csv, and run end-to-end without manual intervention.
- Use only libraries available in the expected Kaggle Docker/runtime environment for official solution.py or solution.ipynb submissions unless the challenge explicitly permits installing or packaging extras.
- For upload handoffs, match the platform's field names exactly: provide solution.py or solution.ipynb plus submission.csv. If also creating a local folder, mirror the runtime-generated output as working/submission.csv. Do not present solution.csv as the final upload file.
- Keep final code and final CSV consistent. If a post-processing change creates a new submission.csv (thresholding, calibration, blending, quantiles, clipping, etc.), update solution.py/solution.ipynb to reproduce that same decision rule before handing files to the user.
- Never make official solution.py or solution.ipynb read, discover, validate, reorder, copy, mirror, or fall back to an uploaded/root/sibling submission.csv, previous working/submission.csv, or other precomputed test prediction file. The official script must generate ./working/submission.csv from ./dataset/public/ through the declared modeling pipeline when run in isolation.
- Never ship a decoy or fallback pipeline that produces different predictions from the submitted CSV. If a neural ensemble, blend, calibration, or postprocessing path produced the upload, the official code must execute that same path or the submission is not review-safe.
- After public-score feedback, update only the canonical upload files (solution.py/solution.ipynb, submission.csv, optional working/submission.csv) and log the previous public score plus the exact calibration change. Treat public scores as coarse diagnostics only; do not use repeated submissions to infer row-level labels, and avoid large distribution swings unless there is local validation evidence for them.
- Assume official runtime is constrained, usually A10G-class GPU, 24GB VRAM, 64GB RAM, and about 30 minutes unless the challenge page says otherwise.
- Use AWS connected CPU for sweeps and recipe discovery only; final official notebooks must reproduce without cloud CPU artifacts unless the challenge explicitly allows uploaded artifacts.
- Optimize for expected payout per hour: valid submission first, compliance second, local validation third, public score fourth, private robustness fifth.
- Show a clear progression from baseline to optimized submissions when using multiple credits; record what changed and why.

## Long-Run / H100 Safety

- Never run CPU training, remote final-training, AWS provisioning, scp, or long CV commands as an unbounded foreground command. Every long command must have a tool timeout and a plan for recovery.
- For expected runs over 5 minutes, start the work detached on the remote host with nohup, tmux, or an equivalent background job. Always write a PID file, stdout/stderr log, and summary/artifact path, then return control to the user-facing thread quickly.
- Poll detached jobs in short bounded intervals, usually 30-120 seconds, by checking PID/process status and tailing logs. Do not keep an SSH session open waiting for a full training run.
- Every remote command should include SSH connection safeguards such as BatchMode=yes, ConnectTimeout, and ServerAliveInterval/ServerAliveCountMax when practical. Prefer one short SSH command per poll.
- If a command appears stuck, first inspect local ssh/scp processes and remote ps output before assuming training is still active. Kill only the known PID from the recorded PID file, not broad Python processes.
- Record run metadata in project notes or a run summary: command, host/IP, PID, log path, start time, expected duration, artifacts, and stop/deallocate status.
- Deallocate or stop expensive cloud VMs when research is complete or when a run is abandoned, after preserving needed artifacts.
- If the platform lock/deadline is near, prefer producing a valid current-best handoff over launching another long sweep.
- RUNTIME CPU-MEMORY PARITY (mandatory): the grader runs the official notebook on the constrained runtime CPU (assume ~62 GB RAM unless the page says otherwise), NOT the cloud CPU. A model/batch/resolution that trains fine on the cloud CPU can OOM on the grader. Before shipping any solution, VERIFY it fits the target memory, and run the official script end-to-end, or test on a <=24 GB device. If it OOMs, reduce footprint without changing outputs first: gradient checkpointing (model.set_grad_checkpointing() / torch.utils.checkpoint, the biggest, math-identical lever), then smaller batch, lower resolution, smaller backbone, CPU-resident image cache, and set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True inside the script. Never ship a solution validated only at full H100 memory.

## Hard Guardrails

Never generate or recommend solutions that:

- Read ./dataset/private/, hidden labels, grader answers, or files outside allowed dataset/work directories.
- Use internet, external APIs, hosted LLMs, external datasets, or cloud model calls inside the official notebook unless explicitly allowed by the challenge.
- Manually label test examples, hardcode row-level test predictions, probe the public leaderboard into answer maps, exploit platform bugs, or reverse-engineer hidden generators when prohibited.
- Use hardcoded physics-simulation constants, parameter dictionaries, or generator settings such as BEST_PARAMS, SIM_PARAMS, or DT_PARAMS as the predictive mechanism. If a physics task permits modeling, parameters must be learned/fitted from allowed public training data inside the official pipeline, not copied in as magic simulator constants.
- Use dataset fingerprinting, file ordering, hidden IDs, image metadata, sample IDs, filenames, row positions, or other construction artifacts as the main prediction signal unless the challenge explicitly makes them part of the intended task.
- Use regex, heuristic pattern-based text matching, or rule-based-only shortcuts as the predictive solution when the challenge expects a learned ML approach. Regex is only acceptable for non-predictive parsing/validation when required by the file format.
- For challenges that ban symbolic simulation or explicit structure extraction, hide a simulator inside a neural-looking pipeline. Differentiable, soft, or learned detectors are still disallowed if their outputs represent the prohibited structure and are consumed by hand-written algorithmic code to compute the answer.
- Override submission rows manually, exploit leaderboard feedback, or make changes that cannot be reproduced by the official solution script.
- Copy, mirror, reorder, or validate an uploaded submission.csv as the official output instead of computing predictions in the official runtime.
- Include a visible ML pipeline that is bypassed whenever a precomputed submission file is present.
- Use private, role-gated, paid, obscure, inaccessible, API-backed, or non-reproducible pretrained models, weights, embeddings, or endpoints.
- Upload magic weights or externally trained artifacts that bypass the runtime/training limits or cannot be fairly reproduced in the expected environment.
- Ignore challenge-specific method rules, such as open-weight LLM requirements or bans on metadata/source IDs.
- Submit invalid files: missing rows, duplicate IDs, wrong columns/order, invalid labels, NaN/Inf, out-of-range values, or nondeterministic outputs.

If the user asks for a violating path, redirect to a compliant modeling or validation alternative.

## Pretrained Model Policy

- Public, commonly available pretrained models are usually allowed when appropriate for the challenge and compliant with its rules.
- Do not rely on private credentials, hidden services, personal API keys, non-public artifacts, private endpoints, or paid/role-gated model access.
- Do not use niche or domain-specific pretrained models that effectively bypass the intended difficulty or make the solution unfairly hard to reproduce.
- Do not use external API inference, including LLM APIs or private model endpoints, unless the challenge explicitly permits it.
- The final solution should be reproducible inside the expected environment and should not use pretrained models as a way to evade runtime limits.

## Workflow

1. Create or enter the challenge directory at repo root.
2. Capture the challenge statement, rules, metric, AI baseline, leaderboard state, close/lock status, and dataset layout in the project notes.
3. Inspect ./dataset/public/ and load sample_submission.csv before modeling.
4. Choose a task recipe from references/task-recipes.md and build a serious baseline that writes a valid submission. When the target is a geometric/motion quantity (displacement, offset, disparity, pose, or a binning of one), apply the Geometric / Motion / Displacement recipe (learned correlation/cost-volume between paired inputs plus label-preserving geometric augmentation) rather than defaulting to a scalar-feature-bag plus tree. Consult this recipe whenever authoring a solve plan, including injected per-challenge strategies, not only during sequential solves.
5. Add local scoring/proxy validation aligned with the official metric; use group/time splits when leakage risk exists.
6. Improve in clear increments: features, stronger model, calibration, thresholding, ensembling, or compact neural training.
7. Run a strict final validator before any official submission.
8. Track each credit: credit number, public score, local score, change made, and next action.
9. Write approach.md with submission-form-ready text covering time spent, model architecture, preprocessing, key design decisions, what worked, what did not work, local validation, and compliance notes. Store the numeric time value separately when helpful, e.g. time_spent.txt. In any paste-ready submission text (the short reviewer paragraph / simple-approach.md, and the approach text pasted into the the platform form), avoid typographic AI tells: no em or en dashes, no two words joined by a hyphen into a compound (write "multiscale" not "multi-scale", "per image" not "per-image"; proper model names such as Faster R-CNN and U-Net keep their real spelling), and write small whole numbers as digits (4 not four).
10. Commit meaningful milestones. Push to the private GitHub repo when work is stable or the user asks.

For detailed solver workflow, read references/solve-workflow.md.

## Final Solver Checklist

- solution.py or solution.ipynb runs end-to-end without errors.
- Official code reads only from ./dataset/public/ and writes ./working/submission.csv.
- Official code uses only expected Kaggle Docker/runtime libraries unless the challenge explicitly permits extras.
- The generated submission.csv passes strict local validation.
- An isolated smoke test from a temporary directory containing only solution.py or solution.ipynb and dataset/public/ still generates working/submission.csv; it must not require or read any sibling/root uploaded submission.csv.
- The official code path that generates working/submission.csv matches the model family and postprocessing described in approach.md.
- Local validation or another honest proxy shows a reasonable score and confirms the task is solvable.
- Reasoning is explained at each major step through comments when allowed, or through notebook markdown/approach.md when project instructions ban code comments.
- Multiple submissions show clear progression from baseline to optimized and record exact changes.
- The method solves the intended ML problem in good faith rather than relying on loopholes, metadata, hidden artifacts, hardcoded rows, or leaderboard exploitation.
- The official script has been verified to fit the runtime GPU memory (~A10G/24 GB), not just the H100 — by capping the research-GPU memory fraction or testing on a <=24 GB device — and to finish within the runtime wall-clock budget.

## Outputs To Prefer

For solving:

- solution.ipynb as the official, self-contained notebook.
- validate_submission.py for strict local schema/value checks when useful.
- run_experiment.py or research/ scripts for H100-only experiments, clearly separated from official notebooks.
- notes.md for challenge facts, local CV, public submissions, and next actions.
- approach.md with concise text the user can paste into the the platform "Your Approach" submission field, plus a recommended time-spent value.
- Final upload handoff files named exactly solution.py/solution.ipynb and submission.csv; optional working/submission.csv should be an exact mirror of the final CSV.

For challenge creation:

- dataset_description.md
- problem_description.md
- prepare.py
- grade.py
- rubrics.yaml or rubrics.json
- reference_solution.ipynb

For creation workflow, read references/creation-workflow.md.

## Challenge Triage

Prioritize challenges with high expected value:

- Closing soon but still open or not locked.
- Few solvers above AI baseline.
- Leaderboard gap is plausible and not already near the metric ceiling.
- Task type supports fast iteration: tabular, NLP candidate matching, tokenization, small-data CV, or structured sequence tasks.
- Low disqualification risk.

Spend credits conservatively:

- Credit 1: valid serious baseline.
- Credit 2: main engineered model.
- Credit 3: ensemble, tuned model, or corrected bug.
- Credits 4-5: only when near payout/top 3 or above AI baseline with a clear improvement.
- Credit 6: reserve for final-hour blend or format bug.

MAJOR NOTE: Never Hardcode weights or units to optimize the solution, always keep it honest. do not overfit or underfit. keep it perfect. never break any rule listed under the project description, keep that as the masterguide. follow and adhere everything written there.
