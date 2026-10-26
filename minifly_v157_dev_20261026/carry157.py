"""Carry all actual attempt costs/counters across the explicit technical-cycle boundary."""
import json,time
import support157 as s
def load_prior(cycle):
 prior=s.load(s.H/f'cycles/cycle{cycle}/COUNTS.json');old_cpu=s.cpu
 s.COUNTS.update(prior['counts']);s.TIMES.update(prior['timings']);s.STAGE_USED.update(prior['stage_CPU']);s.STAGE_START=prior['CPU'];s.cpu=lambda:prior['CPU']+old_cpu()
 start=s.load(s.H/'START.json');first=json.loads((s.H/'LEDGER.jsonl').read_text().splitlines()[0]);origin=start['unix']-first['wall'];s.START=time.monotonic()-(time.time()-origin)
 return prior
