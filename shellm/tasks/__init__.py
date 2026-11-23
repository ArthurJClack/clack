"""pc tasks: a tiny, sandboxed action language the model can learn to write.

    task: make a folder called reports and put a file notes.txt in it that says hello world
    > mkdir reports
    > write reports/notes.txt hello world
    .

actions.py  the sandbox that runs those lines (python, cross-platform, cannot leave its folder)
suite.py    task families that generate (instruction, setup, reference program)
learn.py    build training data, train the skill, and score it by really running the output
"""
from .actions import ActionError, Workspace, parse_program
from .suite import FAMILIES, Task, make_tasks
