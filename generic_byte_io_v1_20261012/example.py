"""Tiny DEV example: ordinary continuation and framed, variable-length reply."""

from generic_byte_io import ByteCore

core = ByteCore()

# Ordinary observed stream: actual arriving bytes supply prediction targets.
core.begin_sequence()
core.observe_bytes(b"an observed stream\x00\n\xff")
core.observe_end()

# Framing is external and explicit; no calculator or expected length exists.
core.begin_sequence()
core.observe_bytes("给出字节示例".encode("utf8"), learn=False)
core.start_reply()
core.observe_bytes("答复".encode("utf8") + b"\x00\xff")
core.observe_end()

core.begin_sequence()
core.observe_bytes("给出字节示例".encode("utf8"), learn=False)
core.start_reply()
result = core.generate(max_bytes=100)
print(result)
