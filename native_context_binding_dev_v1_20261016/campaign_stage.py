"""Analysis subprocess so dispatcher and idle monitor never load numerical kernels."""
import argparse
import campaign_common as c


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=("DEV","CONFIRM"),required=True)
    args=p.parse_args()
    import campaign_analysis as analysis
    import campaign_finish as finalizer
    if args.stage=="DEV":
        selection,worlds=analysis.analyze_dev()
        if selection["selected"] is None:finalizer.finish("NO_ADVANCE")
        print(selection["verdict"],flush=True)
    else:
        lock=c.read(c.RESULTS/"DEV/SELECTION_LOCK.json")
        verdict,worlds=analysis.analyze_confirm(lock["selected"],lock["confirmation_CPU_budget"])
        finalizer.finish(verdict["verdict"])
        print(verdict["verdict"],flush=True)
