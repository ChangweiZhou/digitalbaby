"""Durable accounting: reserve before any technical record application."""
import json
import os
import config
from artifacts import now,write,read
PATH=config.HERE/'results/TECH_EXPOSURES.json'


def apply(cycle,label,call):
    ledger=read(PATH) if PATH.exists() else dict(cap=config.TECH_RECORD_CAP,records=[])
    if len(ledger['records'])>=ledger['cap']:raise RuntimeError('technical record cap exhausted')
    row=dict(index=len(ledger['records'])+1,cycle=cycle,label=label,status='STARTED',utc=now())
    ledger['records'].append(row);write(PATH,ledger)
    try:
        result=call()
        row.update(status='COMPLETED',finished=now());write(PATH,ledger)
        return result
    except BaseException as exc:
        row.update(status='FAILED_ATTEMPT',error=repr(exc),finished=now());write(PATH,ledger);raise
