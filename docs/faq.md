# FAQ

## Q: Is TGEFD another AutoML or symbolic regression tool?

**A:** No. TGEFD does not search for the "best" model.
It validates whether discovered models are **structurally stable** or should be rejected.

## Q: What does "reject" mean?

**A:** It means no stable structure was detected in the specified hypothesis space.
This is a **valid, auditable outcome**, not a failure.

## Q: Can TGEFD say that no relationship exists?

**A:** Yes - within the defined hypothesis class.
Negative results are a first-class output.

## Q: Does this replace cross-validation or domain expertise?

**A:** No.
TGEFD complements existing validation by testing **structural robustness**, not accuracy alone.

## Q: Is topology reliable for real data?

**A:** Topology captures invariants that persist under perturbations.
If a structure is real, it remains. If it is noise, it collapses.

## Q: Are results reproducible?

**A:** Yes.
Each run produces immutable artifacts: config, results, and evidence - fully auditable.

## Q: Is this suitable for regulated or enterprise environments?

**A:** Yes.
TGEFD is designed around reproducibility, explainability, and negative certification.

## Q: What does TGEFD not claim?

**A:**

- No claims about physical truth
- No universal laws
- No guarantees outside the chosen hypothesis space

Only **empirical stability**.
