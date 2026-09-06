"""Independently audit saved evaluation logs and their common random inputs."""
import argparse
from pathlib import Path
import numpy as np
from evaluation.metrics import write_json


def audit(output):
    logs = sorted((output/"request_logs").glob("*.npz"))
    if not logs:
        raise ValueError("No request logs found; enable evaluation.save_request_logs")
    reference = {}
    total = 0
    for path in logs:
        trace = path.stem.rsplit("_", 1)[-1]
        with np.load(path) as saved:
            log = dict(saved)
        arrival, service = log["arrival_seconds"], log["service_seconds"]
        start, end, outcome = log["service_start_seconds"], log["resolution_seconds"], log["outcome"]
        assert len(arrival) > 0
        assert all(len(value) == len(arrival) and np.isfinite(value).all() for value in log.values())
        assert np.isin(outcome, [1, 2, 3, 4]).all()
        assert (arrival >= 0).all() and (np.diff(arrival) >= 0).all()
        assert (end >= arrival).all()
        completed = (outcome == 1) | (outcome == 2)
        rejected, expired = outcome == 3, outcome == 4
        served = start >= 0
        assert (start[served] >= arrival[served]).all()
        assert (np.diff(start[served]) >= -1e-12).all(), "FIFO service start order broken"
        np.testing.assert_allclose(end[completed]-start[completed], service[completed], atol=1e-10, rtol=0)
        assert (start[rejected] == -1).all()
        np.testing.assert_array_equal(end[rejected], arrival[rejected])
        assert np.isin(log["server"][served], np.arange(5)).all()
        assert (log["server"][~served] == -1).all()
        # Configuration values are read here to support sensitivity experiments.
        import json
        config = json.loads((output/"config.json").read_text())
        timeout, sla = config["environment"]["timeout"], config["environment"]["sla_seconds"]
        assert (end-arrival <= timeout+1e-10).all()
        np.testing.assert_allclose(end[expired]-arrival[expired], timeout, atol=1e-10, rtol=0)
        assert (end[outcome == 1]-arrival[outcome == 1] <= sla+1e-10).all()
        assert (end[outcome == 2]-arrival[outcome == 2] > sla).all()
        if trace not in reference:
            reference[trace] = (arrival.copy(), service.copy())
        else:
            np.testing.assert_array_equal(arrival, reference[trace][0])
            np.testing.assert_array_equal(service, reference[trace][1])
        total += len(arrival)
    result = dict(status="passed", request_log_files=len(logs), audited_request_records=total,
                  held_out_traces=len(reference), checks=["finite complete logs", "FIFO starts", "exact service accounting",
                  "deadline accounting", "outcome partition", "identical arrivals and service requirements across policies"])
    write_json(output/"request_log_validation.json", result)
    print(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent/"results"/"quick")
    audit(parser.parse_args().results)
