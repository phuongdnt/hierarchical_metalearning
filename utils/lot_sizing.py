"""Lot sizing helper utilities for discrete rule selection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Sequence

RuleParams = Mapping[str, float]


@dataclass(frozen=True)
class LotSizingParameters:
    """Container for lot-sizing rule parameters."""

    foq: Mapping[str, float]
    poq: Mapping[str, float]
    sm: Mapping[str, float]


def moving_average(series: Sequence[float], window: int) -> float:
    """Return the moving average of the most recent ``window`` entries."""

    if window <= 0:
        raise ValueError("window must be positive")

    if not series:
        return 0.0

    window_slice = series[-window:]
    return float(sum(window_slice) / len(window_slice))


def exponential_smoothing(previous: float, alpha: float, new_value: float) -> float:
    """Perform single-parameter exponential smoothing."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be within [0, 1]")

    return float(alpha * new_value + (1.0 - alpha) * previous)


def _resolve_forecast(state: Mapping[str, Sequence[float]], length: int) -> List[float]:
    forecast = list(state.get("forecast", []))
    if length <= 0:
        return []
    if not forecast:
        base = float(state.get("demand_history", [0.0])[-1]) if state.get("demand_history") else 0.0
        return [base for _ in range(length)]
    if len(forecast) >= length:
        return forecast[:length]
    forecast.extend([forecast[-1]] * (length - len(forecast)))
    return forecast


def _inventory_position(state: Mapping[str, float]) -> float:
    on_hand = float(state.get("on_hand", 0.0))
    backlog = float(state.get("backlog", 0.0))
    on_order = float(state.get("on_order", 0.0))
    return on_hand - backlog + on_order


def _compute_foq(state: Mapping[str, float], params: RuleParams) -> float:
    reorder_point = float(params.get("reorder_point", 0.0))
    order_quantity = float(params.get("order_quantity", 0.0))
    target_s = params.get("target_S")
    inv_position = _inventory_position(state)

    if target_s is not None:
        target_s = float(target_s)
        if inv_position < target_s:
            return max(0.0, target_s - inv_position)
        return 0.0

    if inv_position <= reorder_point:
        return max(0.0, order_quantity)
    return 0.0


def _compute_poq(state: Mapping[str, float], params: RuleParams) -> float:
    lead_time = int(params.get("lead_time", 1))
    target_periods = int(params.get("target_periods", 1))
    coverage = max(1, lead_time + target_periods)
    forecast = _resolve_forecast(state, coverage)
    demand_total = float(sum(forecast))
    inv_position = _inventory_position(state)

    net_requirement = demand_total + float(state.get("backlog", 0.0)) - inv_position
    return max(0.0, net_requirement)


def _compute_sm(state: Mapping[str, float], params: RuleParams) -> float:
    forecast_horizon = int(params.get("forecast_horizon", 1))
    setup_cost = float(params.get("setup_cost", 0.0))
    holding_cost = float(params.get("holding_cost", 0.0))

    if forecast_horizon <= 0:
        return 0.0

    forecast = _resolve_forecast(state, forecast_horizon)
    if not forecast:
        return 0.0

    cumulative_demand = 0.0
    cumulative_holding = 0.0
    prev_avg_cost = float("inf")
    best_k = 1

    for period, demand in enumerate(forecast, start=1):
        cumulative_demand += demand
        cumulative_holding += holding_cost * demand * (period - 1)
        avg_cost = (setup_cost + cumulative_holding) / period
        if avg_cost > prev_avg_cost + 1e-8:
            break
        prev_avg_cost = avg_cost
        best_k = period

    total_demand = sum(forecast[:best_k])
    inv_position = _inventory_position(state)
    net_requirement = total_demand - inv_position
    return max(0.0, net_requirement)


_RULE_DISPATCH = {
    0: _compute_foq,
    1: _compute_poq,
    2: _compute_sm,
}


def compute_order(rule_id: int, state: Mapping[str, float], params: Mapping[str, RuleParams]) -> float:
    """Compute the order quantity for the given rule identifier."""

    if rule_id not in _RULE_DISPATCH:
        raise ValueError(f"Unknown rule id {rule_id}")

    rule_name = ["foq", "poq", "sm"][rule_id]
    rule_params = params.get(rule_name, {})
    quantity = _RULE_DISPATCH[rule_id](state, rule_params)
    return max(0.0, float(quantity))


__all__ = [
    "LotSizingParameters",
    "compute_order",
    "moving_average",
    "exponential_smoothing",
]
