"""Exact event simulation on numeric arrays; no Python object per request."""
import numpy as np
from numba import njit


@njit(cache=True)
def advance(start, end, arrival_limit, arrivals, service, starts, resolved,
            outcomes, server_log, enabled, jobs, finishes, queue, head, size,
            cursor, timeout, sla):
    counts = np.zeros(4, dtype=np.int64)
    latencies = np.empty(len(arrivals) - cursor + size + 5, dtype=np.float64)
    nlat = 0
    busy_seconds = 0.0
    now = start
    while True:
        # Dispatch FIFO to idle servers; queued deadlines are processed below.
        for s in range(5):
            if enabled[s] and jobs[s] < 0 and size > 0:
                rid = queue[head]
                if arrivals[rid] + timeout > now:
                    head = (head + 1) % len(queue)
                    size -= 1
                    jobs[s] = rid
                    starts[rid] = now
                    server_log[rid] = s
                    finishes[s] = min(now + service[rid], arrivals[rid] + timeout)
        next_arrival = arrivals[cursor] if cursor < len(arrivals) and arrivals[cursor] < arrival_limit else np.inf
        next_queue_timeout = arrivals[queue[head]] + timeout if size else np.inf
        next_finish = np.inf
        sid = -1
        busy = 0
        for s in range(5):
            if enabled[s] and jobs[s] >= 0:
                busy += 1
                if finishes[s] < next_finish:
                    next_finish = finishes[s]
                    sid = s
        event = min(next_arrival, next_queue_timeout, next_finish)
        busy_seconds += busy * (min(event, end) - now)
        if event > end or event == np.inf:
            break
        now = event
        # Ties: service resolution, waiting timeout, arrival. Boundary events
        # resolve before the next decision; arrivals belong to [start, end).
        if next_finish <= next_queue_timeout and next_finish <= next_arrival:
            rid = jobs[sid]
            elapsed = now - arrivals[rid]
            if starts[rid] + service[rid] <= arrivals[rid] + timeout:
                code = 1 if elapsed <= sla + 1e-12 else 2
                latencies[nlat] = elapsed
                nlat += 1
            else:
                code = 4
            resolved[rid] = now
            outcomes[rid] = code
            counts[code - 1] += 1
            jobs[sid] = -1
            finishes[sid] = np.inf
        elif next_queue_timeout <= next_arrival:
            rid = queue[head]
            head = (head + 1) % len(queue)
            size -= 1
            outcomes[rid] = 4
            resolved[rid] = now
            counts[3] += 1
        else:
            if now >= end:
                break
            rid = cursor
            cursor += 1
            idle = -1
            for s in range(5):
                if enabled[s] and jobs[s] < 0:
                    idle = s
                    break
            if idle >= 0 and size == 0:
                jobs[idle] = rid
                starts[rid] = now
                server_log[rid] = idle
                finishes[idle] = min(now + service[rid], now + timeout)
            elif size < len(queue):
                queue[(head + size) % len(queue)] = rid
                size += 1
            else:
                outcomes[rid] = 3
                resolved[rid] = now
                counts[2] += 1
    return head, size, cursor, counts, latencies[:nlat], busy_seconds

