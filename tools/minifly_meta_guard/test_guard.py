"""Regression tests for the partial executable MiniFly meta-governance guard."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from guard import scope,solver,EXIT_SCOPE_FAIL,EXIT_ADMISSION_FAIL,EXIT_RSS_KILLED,EXIT_TIME_KILLED
CLI=Path(__file__).resolve().parent/"guard.py"

class ScopeTest(unittest.TestCase):
    def test_reject_version(self):
        x=scope(["PROJECT_DOCUMENTS/MINIFLY_VERSION_INDEX.json","minifly_v187_x_design_20261103/DESIGN.md"])
        self.assertEqual(x["status"],"FAIL_SCOPE")
        self.assertEqual(len(x["prohibited"]),2)
    def test_allow_real_meta(self):
        self.assertEqual(scope([".claude/skills/minifly-recursive-self-improvement/SKILL.md","tools/minifly_meta_guard/guard.py"])["status"],"PASS_SCOPE_ONLY")
    def test_disallow_ordinary_changes(self):
        self.assertEqual(scope(["README.md"])["status"],"FAIL_SCOPE")
    def test_empty_is_not_pass(self):
        self.assertEqual(scope([])["status"],"FAIL_SCOPE")
    def test_cli_scope_failure(self):
        x=subprocess.run([sys.executable,str(CLI),"verify-scope","--mode","meta","--changed-file","minifly_v188_test/REPORT.md"],capture_output=True)
        self.assertEqual(x.returncode,EXIT_SCOPE_FAIL)

class SolverTest(unittest.TestCase):
    def dense(self):
        return {"rss_cap_bytes":1073741824,"strategy":"dense-multiclass","execution_guard":"run-capped","reserve_factor":1.5,
        "sites":240,"competitors":256,"classes":257,"features_plus_bias":65,"bytes_per_scalar":8}
    def sparse(self):
        return {"rss_cap_bytes":1073741824,"strategy":"bounded-sparse","execution_guard":"run-capped",
         "max_nnz":12000,"bytes_per_nnz":16,"index_pointer_bytes":4096,
         "max_solver_workspace_bytes":50000000,"baseline_rss_bytes":100000000}
    def source(self):
        return "from scipy.optimize import linprog\nfrom scipy.sparse import hstack\nmatrices=[X.multiply(A[:,c].toarray()) for c in range(12)]\nB=hstack(matrices)\nsol=linprog(obj,A_ub=B)\n"
    def test_v185_bad_dense(self):
        x=solver(self.dense());self.assertEqual(x["status"],"FAIL_RESOURCE_BOUND")
        self.assertGreater(x["estimated_peak_bytes"],8_000_000_000)
    def test_bounded_sparse_good(self):
        x=solver(self.sparse())
        self.assertEqual(x["status"],"ADMIT_FOR_TECHNICAL_ONLY")
    def test_v185_sparse_source_flag(self):
        p=self.sparse();p.update(sites=240,competitors=256,classes=257,features_plus_bias=65,bytes_per_scalar=8)
        self.assertEqual(solver(p,self.source())["status"],"FAIL_SOURCE_BOUNDS_MISMATCH")
    def test_small_source_not_false_blocked(self):
        p=self.sparse();p.update(sites=10,competitors=4,classes=5,features_plus_bias=3,bytes_per_scalar=8)
        self.assertEqual(solver(p,self.source())["status"],"ADMIT_FOR_TECHNICAL_ONLY")
    def test_no_guard_rejected(self):
        p=self.sparse();p.pop("execution_guard")
        self.assertEqual(solver(p)["status"],"FAIL_NO_RUNTIME_GUARD")
    def test_invalid_shape_rejected(self):
        p=self.dense();p["sites"]=-1
        with self.assertRaises(ValueError):solver(p)
    def test_cli_bad(self):
        with tempfile.TemporaryDirectory() as td:
            q=Path(td)/"plan.json";q.write_text(json.dumps(self.dense()))
            p=subprocess.run([sys.executable,str(CLI),"admit-solver","--plan",str(q)],capture_output=True)
            self.assertEqual(p.returncode,EXIT_ADMISSION_FAIL)

@unittest.skipUnless(os.name=="posix","POSIX only")
class RuntimeTest(unittest.TestCase):
    def run_child(self,seconds,program,ram=128):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/"report.json"
            args=[sys.executable,str(CLI),"run-capped","--rss-mib",str(ram),
            "--wall-seconds",str(seconds),"--report",str(out),"--",sys.executable,"-c",program]
            p=subprocess.run(args,capture_output=True,text=True,timeout=15)
            self.assertTrue(out.exists(),p.stderr)
            return p.returncode,json.loads(out.read_text())
    def test_clean_child(self):
        rc,x=self.run_child(3,"print(5)")
        self.assertEqual(rc,0);self.assertEqual(x["status"],"EXITED")
    def test_time_guard(self):
        rc,x=self.run_child(.25,"import time;time.sleep(3)")
        self.assertEqual(rc,EXIT_TIME_KILLED);self.assertEqual(x["status"],"WALL_CAP_EXCEEDED")
    def test_rss_guard(self):
        rc,x=self.run_child(4,"import time;x=bytearray(96*1024*1024);x[::4096]=b'X'*len(x[::4096]);time.sleep(2)",40)
        self.assertEqual(rc,EXIT_RSS_KILLED);self.assertEqual(x["status"],"RSS_CAP_EXCEEDED")

if __name__=="__main__":unittest.main()
