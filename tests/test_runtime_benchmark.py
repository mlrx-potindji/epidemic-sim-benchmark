import random

from experiments.profiling.benchmark_models import (
    advance_fixed_steps,
    balanced_framework_orders,
    summarize_measurements,
)


class StoppingFakeModel:
    def __init__(self):
        self.running = True
        self.steps = 0
        self.step_count = 0

    def step(self):
        if not self.running:
            return
        self.steps += 1
        self.step_count += 1
        self.running = False


def test_measured_framework_order_is_counterbalanced():
    orders = balanced_framework_orders(10, random.Random(17))

    assert len(orders) == 10
    assert sum(order[0] == "ecs" for order in orders) == 5
    assert sum(order[0] == "mesa" for order in orders) == 5


def test_fixed_step_driver_continues_after_mesa_extinction():
    model = StoppingFakeModel()

    completed = advance_fixed_steps("mesa", model, steps=5)

    assert completed == 5
    assert model.steps == 5


def test_runtime_summary_reports_paired_speedup_interval():
    measurements = [
        {"run_id": "run", "repeat": repeat, "framework": framework,
         "simulation_seconds": seconds}
        for repeat, ecs_seconds, mesa_seconds in ((0, 2.0, 1.0), (1, 4.0, 2.0), (2, 3.0, 1.5))
        for framework, seconds in (("ecs", ecs_seconds), ("mesa", mesa_seconds))
    ]

    summary = summarize_measurements(measurements, seed=4)

    assert summary["paired_speedup"]["n_pairs"] == 3
    assert summary["paired_speedup"]["median"] == 2.0
    assert summary["paired_speedup"]["ci_low"] == 2.0
    assert summary["paired_speedup"]["ci_high"] == 2.0