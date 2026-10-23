# Independent cycle 3 audit and required dispositions

Reviewer: independent coordinating agent, 2026-10-01, before third pilot.

The scientific design is ready for the third pilot. Required safeguards:
1. Final analysis and receipt auditor must automatically enforce the exact locked
   roster, arms, resolved parameters, source/runtime/import digests, dose and
   revisions; reject partial, duplicate, missing or extra final data
2. Record and verify actual probe times as well as training clocks; pin pandas too
3. Call the report a receipt-only reconstruction, not independent simulation
   replay. Preselect a fresh-process replay before final launch: all 7 arms in
   world 300001, compared excluding resource timing only
4. Behavioral Holm correction does not jointly confirm all nominal95% mechanism
   component gates. Preserve them as prespecified supportive signatures and state
   that they lack a joint familywise mechanistic confirmation guarantee
5. Run corruption tests for world, parameters, source, runtime and reductions;
   verify summary/render roundtrip
6. Bound all elapsed wall time from first launch, including publication waits and
   resumes, with no reset loophole. Conservatively charge an unobserved in-flight
   job on recovery; stop for budget review if that reaches the cap

Dispositions are implemented in the third-pilot source. Final source and lock
must receive one more independent approval before the first final world runs.

Additional pre-pilot code review: final/pilot runtime import manifests could have
differed if only final imported verification modules. Both now import the same
verification modules before any life. A disposable sealed-final-path smoke test
uses technical world 290099, never any final-world label, and compares pilot/final
runtime plus two distinct-process final receipts excluding resources. Publication
ACKs are bound to world+lock digest+remotely verified commit and retained.
