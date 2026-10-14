"""Operations-only regression; no teaching trajectories or efficacy scoring."""
import errno
import json
from pathlib import Path
import stat
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dispatcher import disk_bytes
from runtime import atomic_bytes, atomic_json
from qualification_tests import Cycle1


class Entry:
    def __init__(self, size=1, error=None, directory=False, parts=('root', 'file')):
        self.size, self.error, self.directory, self.parts = size, error, directory, parts
        self.calls = 0

    def is_file(self):
        return True

    def stat(self):
        self.calls += 1
        if self.error:
            raise self.error
        return types.SimpleNamespace(st_size=self.size,
                                     st_mode=stat.S_IFDIR if self.directory else stat.S_IFREG)


class Inventory:
    def __init__(self, entries):
        self.entries = entries

    def rglob(self, pattern):
        return iter(self.entries)


class OperationsRegression(unittest.TestCase):
    def test_original_race_reproduced(self):
        p = Entry(error=FileNotFoundError(errno.ENOENT, 'atomic replacement'))
        with self.assertRaises(FileNotFoundError):
            sum(x.stat().st_size for x in (p,) if x.is_file())

    def test_disappeared_entry_is_benign(self):
        p = Entry(error=FileNotFoundError(errno.ENOENT, 'gone'))
        self.assertEqual(disk_bytes(Inventory([p, Entry(size=17)])), 17)

    def test_single_stat_per_entry(self):
        p = Entry(size=19)
        self.assertEqual(disk_bytes(Inventory([p])), 19)
        self.assertEqual(p.calls, 1)

    def test_permission_error_still_fails(self):
        with self.assertRaises(PermissionError):
            disk_bytes(Inventory([Entry(error=PermissionError(errno.EACCES, 'denied'))]))

    def test_io_error_still_fails(self):
        with self.assertRaises(OSError):
            disk_bytes(Inventory([Entry(error=OSError(errno.EIO, 'disk I/O'))]))

    def test_directory_not_counted(self):
        self.assertEqual(disk_bytes(Inventory([Entry(size=99, directory=True)])), 0)

    def test_scratch_policy_preserved(self):
        e = Entry(size=23, parts=('root', 'scratch', 'cache'))
        self.assertEqual(disk_bytes(Inventory([e]), skip_scratch=True), 0)
        self.assertEqual(disk_bytes(Inventory([e])), 23)

    def test_archive_exclusion(self):
        excluded, kept = Entry(size=100), Entry(size=29)
        self.assertEqual(disk_bytes(Inventory([excluded, kept]), exclude=(excluded,)), 29)

    def test_existing_atomic_temporary_is_counted(self):
        p = Entry(size=31, parts=('root', 'x.json.pending-token'))
        self.assertEqual(disk_bytes(Inventory([p])), 31)

    def test_eight_atomic_writers(self):
        errors = []
        with tempfile.TemporaryDirectory(prefix='minifly_disk_race_') as td:
            folder = Path(td)
            (folder / 'stable.bin').write_bytes(b's' * 4096)
            gate = threading.Barrier(9)
            def writer(i):
                try:
                    gate.wait()
                    for j in range(128):
                        atomic_bytes(folder / f'heartbeat{i}.json', b'h' * (64 + j % 64))
                        if j % 3 == 0:
                            (folder / f'heartbeat{i}.json').unlink(missing_ok=True)
                except BaseException as exc:
                    errors.append(repr(exc))
            threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
            for thread in threads:
                thread.start()
            gate.wait()
            scans = 0
            while any(thread.is_alive() for thread in threads) or scans < 500:
                self.assertGreaterEqual(disk_bytes(folder), 4096)
                scans += 1
            for thread in threads:
                thread.join()
            self.assertEqual(errors, [])
            expected = sum(p.stat().st_size for p in folder.iterdir())
            self.assertEqual(disk_bytes(folder), expected)
            self.stress_scans = scans


def main():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(OperationsRegression)
    # The prelaunch-only no-authorization test is exercised against an absent
    # authorization path; the real sealed authorization is left intact.
    with patch.object(Path, 'exists', autospec=True, side_effect=lambda p: False
                      if p == ROOT / 'LAUNCH_AUTHORIZATION.json.gz'
                      else original_exists(p)):
        # Its require_authorized call already rejects the intentionally stale
        # source lock until the operations amendment is qualified and sealed.
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(Cycle1))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    atomic_json(Path(__file__).resolve().parent / 'REGRESSION.json', {
        'verdict': 'PASS' if result.wasSuccessful() else 'FAIL',
        'tests': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors),
        'new_scientific_trajectories': 0, 'atomic_writers': 8,
        'atomic_writes': 1024, 'minimum_inventory_scans': 500,
    })
    return 0 if result.wasSuccessful() else 1


original_exists = Path.exists
if __name__ == '__main__':
    sys.exit(main())
