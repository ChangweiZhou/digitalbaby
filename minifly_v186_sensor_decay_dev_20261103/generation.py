"""Greedy native continuation with the arm's own FE timescale."""

def generate(next_class, sensor, prefix, decay, cap=64):
    state = sensor.R[0].copy()
    state.fill(0)
    for byte in prefix:
        state = (state * decay)[sensor.perm] + sensor.R[byte]
    output = bytearray()
    for _ in range(cap):
        faded = state * decay
        choice = next_class(sensor.encode(faded))
        if choice == 256:
            return bytes(output), True
        output.append(choice)
        state = faded[sensor.perm] + sensor.R[choice]
    return bytes(output), False
