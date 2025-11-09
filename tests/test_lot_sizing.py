import pytest

from utils.lot_sizing import compute_order, moving_average

DEFAULT_PARAMS = {
    'foq': {'order_quantity': 20.0, 'reorder_point': 5.0, 'target_S': None},
    'poq': {'lead_time': 2, 'target_periods': 2},
    'sm': {'setup_cost': 5.0, 'holding_cost': 1.0, 'forecast_horizon': 5},
}

def test_moving_average_handles_short_series():
    assert moving_average([1.0, 2.0], 5) == pytest.approx(1.5)
    assert moving_average([], 3) == 0.0

def test_foq_reorder_point_triggers_order():
    state = {'on_hand': 4.0, 'backlog': 0.0, 'on_order': 0.0}
    order = compute_order(0, state, DEFAULT_PARAMS)
    assert order == pytest.approx(20.0)

def test_foq_target_level():
    params = {**DEFAULT_PARAMS, 'foq': {'order_quantity': 10.0, 'reorder_point': 5.0, 'target_S': 30.0}}
    state = {'on_hand': 12.0, 'backlog': 4.0, 'on_order': 6.0}
    order = compute_order(0, state, params)
    # inventory position = 12 - 4 + 6 = 14 so order up to 30
    assert order == pytest.approx(16.0)

def test_poq_covers_lead_time_and_periods():
    params = {**DEFAULT_PARAMS, 'poq': {'lead_time': 2, 'target_periods': 1}}
    state = {'on_hand': 5.0, 'backlog': 0.0, 'on_order': 0.0, 'forecast': [8.0, 9.0, 10.0]}
    order = compute_order(1, state, params)
    expected = sum(state['forecast'][:3]) - (state['on_hand'] - state['backlog'] + state['on_order'])
    assert order == pytest.approx(expected)

def test_silver_meal_stops_when_cost_increases():
    params = {**DEFAULT_PARAMS, 'sm': {'setup_cost': 5.0, 'holding_cost': 1.0, 'forecast_horizon': 4}}
    state = {'on_hand': 0.0, 'backlog': 0.0, 'on_order': 0.0, 'forecast': [4.0, 4.0, 4.0, 4.0]}
    order = compute_order(2, state, params)
    # average cost decreases through period 2 then increases, so cover two periods
    assert order == pytest.approx(8.0)
