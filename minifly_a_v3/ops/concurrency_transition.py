"""Operational-only guards for the approved four-to-six-worker transition."""


def check_drain(status, ledger, observation, receipt_hashes, jobs, hard):
    assert status.get('state') == 'stopped_on_failure', 'Old driver is not terminal'
    assert status['running'] == [], 'Old driver still has running jobs'
    failure = ledger.get('failure')
    assert failure == status.get('failure'), 'Status/ledger failure differs'
    assert failure and failure['type'] == 'infrastructure', 'Only intentional durability stop may resume'
    assert failure['detail']['stage'] in ('persist', 'final_persist')
    assert 'Durability backlog reached8 receipts without either verified GitHub or private Library backup' in failure['detail']['error'], 'Unexpected infrastructure failure'
    assert observation.get('first_failure_seen_unix_s'), 'No observed stop boundary'
    assert len(receipt_hashes) == status['validated_receipts'], 'Receipt count differs from terminal status'
    assert ledger['active_wall_s'] <= hard['active_wall_hours'] * 3600, 'Cumulative wall budget exhausted'
    assert ledger['worker_s'] <= hard['core_hours'] * 3600, 'Cumulative worker budget exhausted'
    for arm, world in observation['observed_running_pairs']:
        key = (arm, world)
        assert key in receipt_hashes, f'Missing drained receipt: {key}'
        job = jobs.get(key, {})
        assert job.get('ok') is True, f'Missing/failed drained job: {key}'
        assert (job.get('arm'), job.get('world')) == key, f'Drained job identity mismatch: {key}'
        assert 0 <= job['life_s'] <= hard['per_world_arm_life_s_max'], f'Drained job runtime breach: {key}'
        assert 0 <= job['peak_rss_bytes'] <= hard['peak_rss_bytes_per_worker'], f'Drained job RSS breach: {key}'
    return True
