import sys
sys.path.insert(0, "/home/user/digitalbaby/MOD97_LOCAL_RULE_20261106")
sys.argv = ["exp55_reserve.py"] + sys.argv[1:]
import exp55_reserve as E
E.T.set_num_threads(1)
task, rule = sys.argv[1], sys.argv[2]
E.run(task, "blind", rule, 73129, E.OUT55 / "checks" / "dry", f"python3 exp55_reserve.py check 3 73129 (its dry run {task} {rule}, started by a helper)", updates=64)
