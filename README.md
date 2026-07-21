# CPU Only ML Solutions

Seven machine learning problems I solved without a GPU. Each had to train and run
inside a fixed grader budget, roughly 10 cores and 62 GB with a wall clock limit,
reading only its own dataset folder and writing one submission file.

That constraint is the whole point. Most of these have an obvious deep learning
answer that I simply could not afford here, so each folder is really a record of
what I chose to give up and what I kept.

| Folder | What the problem is |
|---|---|
| `sparse frame/` | Forecast the bounding box and category of a drifting object at frame t4, given four sparse motion history frames |
| `parser attatchment disagreement/` | For each token, predict whether 25 independent dependency parsers would disagree about what it attaches to |
| `docstring gap/` | Fill a removed span inside a docstring by selecting the best candidate from a learned pool |
| `ins/` | Recover an institutional edit ledger by tagging which whitespace token spans were edited |
| `LPBF/` | Localize square alert regions in transformed laser powder bed fusion inspection images |
| `art-auction-sale-reconstruction/` | Given a pool that mixes the lots of six auction sales, recover which lot belongs to which sale |
| `swadesh phenome/` | Invert a global substitution cipher over IPA segments and decode the lexicon of a hidden Uralic language |

Each folder carries its own `approach.md`, working notes, and solution code.
Datasets are not committed. `GUARDRAILS.md` is the checklist I hold myself to
while working: no id hardcoding, no leakage, no external answer sources.
