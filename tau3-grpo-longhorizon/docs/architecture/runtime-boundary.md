# τ² training and τ³ final boundary

`tau2-bench v1.0.1` is the official τ³ code release despite retaining `tau2`
as its import and CLI name. Two adapters use that same runtime:

- `ArealAirlineTaskLoader` converts synthetic AReaL τ²-style records and loads
  the record-specific DB snapshot.
- `OfficialTau3TaskLoader` reads the untouched 50-task Airline `base` split
  shipped by Sierra.

Separate manifest types and default configs prevent the official final split
from being passed to the training data builder. The verifier bridge always uses
Sierra's evaluator against the matching task/runtime revision.

