# Developer guide

This section explains *why* the codebase is structured the way it is, not just what each
function does (the docstrings, and {doc}`../api/index`, already cover that). Read this
before making a structural change — adding a rewrite rule, changing a container's data
model, or touching the graph conversion — since several of these decisions exist to avoid
bugs that were hit and fixed once already (the module docstrings and commit history are full
of "this used to be wrong because ..." notes; this guide collects the ones worth knowing
before you go looking for them).

```{toctree}
:maxdepth: 1

architecture
rewrite_engine
normalization
benchmarks
```
