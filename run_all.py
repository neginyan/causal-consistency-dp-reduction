"""Run every check and print a summary.  Exit code 0 if all checks pass."""

from __future__ import annotations

import sys
import time

import sec2a_static_field
import sec2b_finite_protocol
import supp_retarded_gaussian
import sec4b_self_crossing_time
import sec4b_arrival_lag
import sec4_propositions
import sec5_what_happens_at_mH


def main() -> int:
    t0 = time.time()
    reports = []
    for mod in (sec2a_static_field, sec2b_finite_protocol, supp_retarded_gaussian, sec4b_self_crossing_time,
                sec4b_arrival_lag, sec4_propositions,
                sec5_what_happens_at_mH):
        rep = mod.run()
        rep.print()
        if hasattr(mod, "table"):
            mod.table()
        reports.append(rep)
    total = sum(len(r.rows) for r in reports)
    passed = sum(r.n_pass for r in reports)
    print()
    print("#" * 78)
    for r in reports:
        print(f"  {'OK  ' if r.n_pass == len(r.rows) else 'FAIL'}  {r.n_pass:2d}/{len(r.rows):<2d}  {r.title}")
    print(f"\n  {passed}/{total} checks passed in {time.time() - t0:.1f} s")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
