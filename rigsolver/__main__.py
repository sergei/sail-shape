"""python -m rigsolver [state.json]  -> solve a tuning state (default: EXAMPLE_TUNE) and print a summary."""
import json
import sys

from .rig import EXAMPLE_TUNE, Rig, RigState

state = RigState.from_dict(json.load(open(sys.argv[1]))) if len(sys.argv) > 1 else RigState(EXAMPLE_TUNE)
print(Rig.load().solve(state).summary())
