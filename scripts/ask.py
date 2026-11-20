"""Run relocation answer generation. Pass a question as the first argument,
 or leave it unset to use the default question below.
"""

import sys
from pathlib import Path

if __name__ == "__main__" and not __package__:
    # Direct script launches put scripts/ on the import path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_QUESTION = (
    "My approved move was completed 45 days ago. "
    "Under the current policy, am I still within the filing window?"
)


def main() -> None:
    from relocation_rag.generate_answer import main as generate_answer_main

    if len(sys.argv) == 1:
        sys.argv.append(DEFAULT_QUESTION)
    generate_answer_main()


if __name__ == "__main__":
    main()
