# Tutorials

Both notebooks run when the documentation is built, on the `pbmc68k_reduced` dataset that ships with
scanpy, so they double as integration tests.

* **Which of my cell-type labels can I trust?** scores every label in a dataset and shows how to read
  the result.
* **Choosing methods** compares clustering algorithms and classifiers against known labels, checks
  whether the differences survive a change of random seed, and runs a differential expression test with
  effect sizes.

```{toctree}
:maxdepth: 1

label_quality_workflow
method_benchmarking
```
