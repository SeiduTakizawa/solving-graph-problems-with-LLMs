"""Skills (M4): a short playbook per task, added to the system prompt when the task is known (routed, or the
oracle's). One Markdown file per task, named after it; tasks without a file get no skill. Off unless asked for
(eval.runner --skills), so it can be measured as an ablation.

Written for the tasks where the traces showed the model improvising and failing (2026-10-09, ms_* runs): long
code that failed a third of the time, invented imports, the node itself as its own best match, evidence lists
copied by hand. The old pseudocodes (data/pseudocodes/) cover the ten simple tasks, which one tool call answers;
porting them is still open.
"""
from pathlib import Path

SKILLS_DIR = Path(__file__).parent


def load_skill(task: str) -> str | None:
    path = SKILLS_DIR / f"{task}.md"
    return path.read_text().strip() if path.exists() else None
