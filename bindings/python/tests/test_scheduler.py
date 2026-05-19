"""Python tests for the v0.7.0 distributed-scheduler binding."""

from __future__ import annotations

import json

import pytest

from moonlab.scheduler import Job, SchedulerError
from moonlab.qgtl import GateType


def test_single_worker_bell():
    j = (Job(num_qubits=2)
         .add_gate(GateType.H, target=0)
         .add_gate(GateType.CNOT, target=1, control=0)
         .set_num_shots(1024)
         .set_num_workers(1)
         .set_rng_seed(0xdeadbeef))
    r = j.execute()
    assert r.total_shots == 1024
    assert r.num_workers_used == 1
    assert all(o in (0, 3) for o in r.outcomes)


def test_four_worker_bell():
    j = (Job(num_qubits=2)
         .add_gate(GateType.H, target=0)
         .add_gate(GateType.CNOT, target=1, control=0)
         .set_num_shots(1024)
         .set_num_workers(4)
         .set_rng_seed(0xdeadbeef))
    r = j.execute()
    assert r.num_workers_used == 4
    n00 = sum(1 for o in r.outcomes if o == 0)
    n11 = sum(1 for o in r.outcomes if o == 3)
    nother = sum(1 for o in r.outcomes if o not in (0, 3))
    assert nother == 0
    assert n00 + n11 == 1024
    # Bell parity preserved across workers.
    assert abs(n00 - 512) < 80
    # worker_seconds vector populated.
    assert len(r.worker_seconds) == 4
    assert all(s >= 0 for s in r.worker_seconds)


def test_three_worker_ghz():
    j = Job(num_qubits=3)
    j.add_gate(GateType.H, target=0)
    j.add_gate(GateType.CNOT, target=1, control=0)
    j.add_gate(GateType.CNOT, target=2, control=1)
    j.set_num_shots(3000).set_num_workers(3).set_rng_seed(0xcafe)
    r = j.execute()
    assert all(o in (0, 7) for o in r.outcomes)
    assert r.num_workers_used == 3


def test_to_json_roundtrip_through_dict():
    """The JSON spec is a contract; verify the schema string + fields."""
    j = (Job(num_qubits=2)
         .add_gate(GateType.H, target=0)
         .add_gate(GateType.CNOT, target=1, control=0)
         .set_num_shots(256)
         .set_num_workers(2)
         .set_rng_seed(0xbeef))
    s = j.to_json()
    parsed = json.loads(s)
    assert parsed["schema"] == "moonlab/job/v0.7.0"
    assert parsed["num_qubits"] == 2
    assert parsed["num_shots"] == 256
    assert parsed["num_workers"] == 2
    assert len(parsed["gates"]) == 2
    assert parsed["gates"][0]["type"] == int(GateType.H)
    assert parsed["gates"][1]["type"] == int(GateType.CNOT)
    assert parsed["gates"][1]["control"] == 0


def test_introspection():
    j = Job(num_qubits=4)
    assert j.num_qubits == 4
    assert j.num_gates == 0
    j.add_gate(GateType.H, target=0)
    assert j.num_gates == 1
    j.set_num_shots(100).set_num_workers(4)
    assert j.num_shots == 100
    assert j.num_workers == 4


def test_rejects_zero_qubits():
    with pytest.raises(SchedulerError):
        Job(num_qubits=0)


def test_rejects_negative_shots():
    j = Job(num_qubits=2)
    with pytest.raises(SchedulerError):
        j.set_num_shots(-1)


def test_rejects_zero_workers():
    j = Job(num_qubits=2)
    with pytest.raises(SchedulerError):
        j.set_num_workers(0)


def test_rejects_zero_shot_execute():
    j = Job(num_qubits=2).add_gate(GateType.H, target=0)
    with pytest.raises(SchedulerError):
        j.execute()
