import tempfile
import unittest
from pathlib import Path

import numpy as np

from generic_byte_io import ByteCore, CoreConfig, EOS


class GenericByteIOTests(unittest.TestCase):
    def core(self, **kwargs):
        return ByteCore(CoreConfig(**kwargs))

    def test_all_256_payload_bytes_and_no_reserved_punctuation(self):
        core = self.core()
        core.begin_sequence()
        learned = core.learned_digest()
        payload = bytes(range(256)) + b"\x00\n=+\xff"
        core.observe_bytes(payload, learn=False)
        self.assertEqual(core.observed_payload_bytes, len(payload))
        self.assertEqual(core.learning_events, 0)
        self.assertEqual(core.learned_digest(), learned)
        self.assertTrue(core._open)
        self.assertEqual(core._ordered_context().tolist(), [257] + list(payload))
        self.assertEqual(len(core.prediction().probabilities), 257)

    def test_prediction_is_read_only_and_before_observation(self):
        core = self.core()
        core.begin_sequence()
        learned, transient = core.learned_digest(), core.transient_digest()
        before = core.prediction()
        again = core.prediction()
        self.assertTrue(np.array_equal(before.probabilities, again.probabilities))
        self.assertEqual((learned, transient), (core.learned_digest(), core.transient_digest()))
        returned = core.observe(123)
        self.assertTrue(np.array_equal(before.probabilities, returned.probabilities))
        self.assertNotEqual(learned, core.learned_digest())

    def test_chunking_parity_including_binary_and_utf8(self):
        data = bytes(range(256)) + "你好🪰".encode("utf8")
        whole, chunks = self.core(max_context_events=80), self.core(max_context_events=80)
        for core in (whole, chunks):
            core.begin_sequence()
        whole.observe_bytes(data)
        for offset in range(0, len(data), 7):
            chunks.observe_bytes(memoryview(data)[offset:offset + 7])
        self.assertEqual(whole.learned_digest(), chunks.learned_digest())
        self.assertEqual(whole.transient_digest(), chunks.transient_digest())

    def test_strided_and_signed_byte_memoryviews_do_not_require_chunk_copies(self):
        raw, view = self.core(), self.core()
        for core in (raw, view):
            core.begin_sequence()
        raw.observe_bytes(b"ace\xff\x80\x00")
        view.observe_bytes(memoryview(b"abcdef")[::2])
        view.observe_bytes(memoryview(np.asarray([-1, -128, 0], dtype=np.int8)))
        self.assertEqual(raw.learned_digest(), view.learned_digest())
        self.assertEqual(raw.transient_digest(), view.transient_digest())

    def test_context_is_ordered_and_not_only_four_byte_suffix(self):
        a, b, c = self.core(), self.core(), self.core()
        for core, payload in ((a, b"AAshared=tail"), (b, b"BBshared=tail"), (c, b"ABshared=tail")):
            core.begin_sequence()
            core.observe_bytes(payload, learn=False)
        self.assertFalse(np.array_equal(a._address(), b._address()))
        self.assertFalse(np.array_equal(a._address(), c._address()))

    def test_fixed_storage_and_explicit_rolling_overflow(self):
        core = self.core(max_context_events=9)
        initial = core.storage_accounting()["persistent_array_bytes"]
        core.begin_sequence()
        core.observe_bytes(bytes(range(256)) * 3, learn=False)
        self.assertEqual(core._context_len, 9)
        self.assertEqual(core.context_overflow_events, 1 + 768 - 9)
        self.assertEqual(core.storage_accounting()["persistent_array_bytes"], initial)
        self.assertEqual(core._ordered_context().tolist(), list((bytes(range(256)) * 3)[-9:]))

    def test_generation_binary_multibyte_utf8_and_learned_eos(self):
        core = self.core()
        prompt = b"raw\x00\n=prompt"
        answer = bytes(range(256)) + "答复🪰".encode("utf8")
        core.begin_sequence()
        core.observe_bytes(prompt, learn=False)
        core.start_reply()
        core.observe_bytes(answer)
        core.observe_end()
        core.begin_sequence()
        core.observe_bytes(prompt, learn=False)
        core.start_reply()
        learned = core.learned_digest()
        output = core.generate(max_bytes=1024)
        self.assertEqual(output.payload, answer)
        self.assertEqual(output.status, "ENDED")
        self.assertEqual(output.selected_events, len(answer) + 1)
        self.assertEqual(core.learned_digest(), learned)
        self.assertFalse(core._open)

    def test_empty_reply_and_truncation_are_distinct(self):
        core = self.core()
        core.begin_sequence()
        core.start_reply()
        core.observe_end()
        core.begin_sequence()
        core.start_reply()
        self.assertEqual(core.generate(0).status, "ENDED")
        untrained = self.core()
        untrained.begin_sequence()
        old = untrained.learned_digest()
        output = untrained.generate(2)
        self.assertEqual(output.payload, b"\x00\x00")
        self.assertEqual(output.status, "TRUNCATED")
        self.assertEqual(untrained.learned_digest(), old)
        self.assertTrue(untrained._open)

    def test_observed_and_generated_eos_share_the_same_context_transition(self):
        trained = self.core()
        trained.begin_sequence()
        trained.observe_bytes(b"prompt", learn=False)
        trained.start_reply()
        trained.observe_end()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "core.npz"
            trained.begin_sequence()
            trained.observe_bytes(b"prompt", learn=False)
            trained.start_reply()
            trained.save_checkpoint(checkpoint)
            observed = ByteCore.load_checkpoint(checkpoint)
            generated = ByteCore.load_checkpoint(checkpoint)
            learned = trained.learned_digest()
            self.assertEqual(observed.observe_end(learn=False).event, EOS)
            self.assertEqual(generated.generate(50).status, "ENDED")
            self.assertTrue(np.array_equal(observed._ordered_context(), generated._ordered_context()))
            self.assertEqual(observed.learned_digest(), learned)
            self.assertEqual(generated.learned_digest(), learned)

    def test_invalid_checkpoint_is_rejected(self):
        core = self.core()
        core.begin_sequence()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "bad.npz"
            core.save_checkpoint(checkpoint)
            with np.load(checkpoint, allow_pickle=False) as archive:
                metadata = archive["metadata"]
                weights = archive["weights"].copy()
                context = archive["context"].copy()
            weights[0, 0] = np.nan
            np.savez(checkpoint, metadata=metadata, weights=weights, context=context)
            with self.assertRaises(ValueError):
                ByteCore.load_checkpoint(checkpoint)

    def test_declared_time_is_persisted_without_claiming_native_decay(self):
        core = self.core()
        core.begin_sequence()
        core.observe_bytes(b"actual-byte-experience")
        learned = core.learned_digest()
        core.rest(86400)
        self.assertEqual(core.elapsed_seconds, 86400.0)
        self.assertEqual(core.learned_digest(), learned)
        for invalid in (True, -1, float("inf"), float("nan"), "24h"):
            with self.assertRaises((TypeError, ValueError)):
                core.rest(invalid)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clock.npz"
            core.save_checkpoint(path)
            loaded = ByteCore.load_checkpoint(path)
            self.assertEqual(loaded.elapsed_seconds, 86400.0)
            self.assertEqual(loaded.transient_digest(), core.transient_digest())
            self.assertEqual(loaded.learned_digest(), learned)

    def test_observed_eos_not_byte_or_implicit_begin_label(self):
        core = self.core()
        core.begin_sequence()
        core.observe(10)
        learning_events = core.learning_events
        core.begin_sequence()
        self.assertEqual(core.learning_events, learning_events)
        with self.assertRaises(ValueError):
            core.observe(EOS)
        core.observe_end()
        with self.assertRaises(RuntimeError):
            core.observe(2)

    def test_checkpoint_midstream_and_generation_resume_parity(self):
        core = self.core(max_context_events=90)
        core.begin_sequence()
        core.observe_bytes(b"prefix\x00\n")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "core.npz"
            core.save_checkpoint(checkpoint)
            loaded = ByteCore.load_checkpoint(checkpoint)
            self.assertEqual(core.learned_digest(), loaded.learned_digest())
            self.assertEqual(core.transient_digest(), loaded.transient_digest())
            for item in (core, loaded):
                item.observe_bytes(bytes(range(100)))
                item.start_reply()
                item.observe_bytes(b"response\x00\xff")
                item.observe_end()
            self.assertEqual(core.learned_digest(), loaded.learned_digest())
            self.assertEqual(core.transient_digest(), loaded.transient_digest())
            for item in (core, loaded):
                item.begin_sequence()
                item.observe_bytes(b"prefix\x00\n" + bytes(range(100)), learn=False)
                item.start_reply()
            first = core.generate(3)
            self.assertEqual(first.status, "TRUNCATED")
            core.save_checkpoint(checkpoint)
            resumed = ByteCore.load_checkpoint(checkpoint)
            continuation = core.generate(100)
            restored = resumed.generate(100)
            self.assertEqual(continuation, restored)
            self.assertEqual(core.transient_digest(), resumed.transient_digest())

    def test_no_write_control_cannot_recite_taught_reply(self):
        trained, disabled = self.core(), self.core()
        for core, learn in ((trained, True), (disabled, False)):
            core.begin_sequence()
            core.observe_bytes(b"opaque-context", learn=False)
            core.start_reply()
            core.observe_bytes(b"\xffhello\x00", learn=learn)
            core.observe_end(learn=learn)
            core.begin_sequence()
            core.observe_bytes(b"opaque-context", learn=False)
            core.start_reply()
        self.assertEqual(trained.generate(20).payload, b"\xffhello\x00")
        self.assertNotEqual(disabled.generate(20).payload, b"\xffhello\x00")
        self.assertEqual(disabled.learning_events, 0)

    def test_invalid_types_and_flags_do_not_mutate(self):
        core = self.core()
        core.begin_sequence()
        prior = core.transient_digest()
        for invalid in (True, False, b"a", 1.0, -1, 256):
            with self.assertRaises((TypeError, ValueError)):
                core.observe(invalid)
        with self.assertRaises(TypeError):
            core.observe(2, learn=1)
        with self.assertRaises(TypeError):
            core.generate(True)
        with self.assertRaises(ValueError):
            core.generate(-1)
        self.assertEqual(prior, core.transient_digest())


if __name__ == "__main__":
    unittest.main()
