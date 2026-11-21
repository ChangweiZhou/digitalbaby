import sys, numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
bad = [k for k in a.files if k != "cpu" and not (a[k].shape == b[k].shape and np.array_equal(a[k], b[k]))]
print("IDENTICAL" if not bad else "DIFFER in " + ", ".join(bad), "| speed-up %.2fx" % (float(a["cpu"]) / float(b["cpu"])))
