"""Publish the approved project snapshot in bounded normal Git pushes."""
import json
import os
from pathlib import Path
import subprocess
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PLAN = json.loads((OUT / 'BATCH_PLAN.json').read_text())
BRANCH = PLAN['branch']


def run(args, capture=False):
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0')
    result = subprocess.run(args, cwd=ROOT, env=env, text=True,
                            stdout=subprocess.PIPE if capture else None, check=True)
    return result.stdout.strip() if capture else ''


def state(stage, **extra):
    payload = dict(stage=stage, epoch=time.time(), pid=os.getpid(), branch=BRANCH,
                   native_records_executed=0, **extra)
    (OUT / 'RUN_STATE.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload), flush=True)


def push():
    run(['git', '-c', 'http.version=HTTP/1.1', '-c', 'http.postBuffer=1073741824', '-c', 'pack.window=0',
         '-c', 'pack.depth=0', '-c', 'pack.threads=2', '-c', 'core.compression=1',
         'push', '--porcelain', 'origin',
         'HEAD:refs/heads/' + BRANCH])
    head = run(['git', 'rev-parse', 'HEAD'], True)
    remote = run(['git', 'ls-remote', '--heads', 'origin', 'refs/heads/' + BRANCH], True).split()[0]
    if head != remote:
        raise RuntimeError('Remote tip differs after normal push; no force push attempted')
    return head


def main():
    if run(['git', 'branch', '--show-current'], True) != BRANCH:
        raise RuntimeError('Unexpected current branch')
    titles = [
        'Publish gated-feedback design and archived arithmetic preparations',
        'Preserve remaining experiment evidence and recovery artifacts (2/3)',
        'Preserve remaining experiment evidence and recovery artifacts (3/3)',
    ]
    head = run(['git', 'rev-parse', 'HEAD'], True)
    commits = []
    if head != PLAN['starting_commit']:
        run(['git', 'merge-base', '--is-ancestor', PLAN['starting_commit'], 'HEAD'])
        commits = run(['git', 'rev-list', '--reverse', PLAN['starting_commit'] + '..HEAD'], True).splitlines()
        if len(commits) > PLAN['batch_count']:
            raise RuntimeError('Unexpected commits since plan; inspect before publishing')
        for i, commit in enumerate(commits):
            if run(['git', 'show', '-s', '--format=%s', commit], True) != titles[i]:
                raise RuntimeError('Unrecognized commit; preserve and inspect')
    staged = set(run(['git', 'diff', '--cached', '--name-only'], True).splitlines())
    next_index = min(len(commits), len(PLAN['batches']) - 1)
    if staged and not staged.issubset(set(PLAN['batches'][next_index]['files'])):
        raise RuntimeError('Index contains changes outside the approved batch; preserve it')
    try:
        if commits:
            state('RETRY_PUSHING_EXISTING_COMMITS', published_commits=commits)
            push()
        already_committed = len(commits)
        for i, batch in enumerate(PLAN['batches'], 1):
            if i <= already_committed:
                continue
            state('STAGING', batch=i, batches=PLAN['batch_count'], bytes=batch['bytes'],
                  files=len(batch['files']), published_commits=commits)
            spec = OUT / ('batch' + str(i).zfill(3) + '.nul')
            run(['git', 'add', '--sparse', '--pathspec-from-file=' + str(spec), '--pathspec-file-nul'])
            run(['git', '-c', 'gc.auto=0', 'commit', '-m', titles[i - 1]])
            tip = run(['git', 'rev-parse', 'HEAD'], True)
            state('PUSHING', batch=i, batches=PLAN['batch_count'], commit=tip,
                  published_commits=commits)
            push()
            commits.append(tip)
            state('BATCH_PUBLISHED', batch=i, batches=PLAN['batch_count'],
                  published_commits=commits)
        published = dict(status='PUBLISHED', date_client='2026-10-14',
                         repository='https://github.com/ChangweiZhou/digitalbaby',
                         branch=BRANCH, payload_commits=commits,
                         payload_tip=commits[-1], scope='All pending nonignored project work, with archive-covered duplicate binary payloads represented by their published archive members.',
                         archive_duplicate_copies=PLAN['archive_dedup_covered_files'],
                         native_training_records_executed=0, scientific_validation_rerun=False,
                         new_design_status='design only; three-cycle qualification not executed',
                         normal_push_only=True)
        (OUT / 'PUBLICATION.json').write_text(json.dumps(published, indent=2) + '\n')
        files = [p for p in OUT.iterdir() if p.is_file()
                 and p.suffix != '.nul' and p.name not in ('RUN_STATE.json', 'push.log')]
        run(['git', 'add', '--sparse', '--', *[str(p.relative_to(ROOT)) for p in files]])
        run(['git', '-c', 'gc.auto=0', 'commit', '-m', 'Record complete project publication through 14 October 2026'])
        final = push()
        state('COMPLETED', final_commit=final, published_commits=commits,
              url='https://github.com/ChangweiZhou/digitalbaby/tree/' + BRANCH)
    except BaseException as e:
        state('FAILED', error=repr(e), published_commits=commits)
        raise


if __name__ == '__main__':
    main()
