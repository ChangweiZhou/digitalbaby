import argparse
import traceback
from pathlib import Path
import bootstrap
from artifacts import write,now


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--track',choices=('A','B'),required=True)
    p.add_argument('--world',type=int);p.add_argument('--model')
    args=p.parse_args()
    try:
        if args.track=='A':
            from native_track import run
            print(run(),flush=True)
        else:
            from addition_track import run_job
            print(run_job(args.world,args.model,bootstrap.HERE/'results/track_b/receipts'/f'{args.world}_{args.model}.json.gz')['record_applications'],flush=True)
    except BaseException as exc:
        write(bootstrap.HERE/'results'/('track_a' if args.track=='A' else 'track_b')/'FAILURE.json',
              dict(utc=now(),error=repr(exc),traceback=traceback.format_exc()))
        raise
