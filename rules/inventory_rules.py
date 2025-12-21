import numpy as np


class BaseRule:
    def __init__(self, name):
        self.name = name

    def get_action(self, current_inventory, demand_history):
        """
        Input:
            current_inventory (float): Inventory position (on-hand + pipeline).
            demand_history (list/array): Recent demand history.
        Output:
            action (float): Order quantity.
        """
        raise NotImplementedError


class FOQRule(BaseRule):
    """Fixed Order Quantity: Order a fixed amount when inventory is low."""

    def __init__(self, reorder_point=10, quantity=20):
        super().__init__(f"FOQ_R{reorder_point}_Q{quantity}")
        self.reorder_point = reorder_point
        self.quantity = quantity

    def get_action(self, current_inventory, demand_history):
        if current_inventory <= self.reorder_point:
            return float(self.quantity)
        return 0.0


class POQRule(BaseRule):
    """Periodic Order Quantity: Order up to a target level."""

    def __init__(self, target_level=30):
        super().__init__(f"POQ_T{target_level}")
        self.target_level = target_level

    def get_action(self, current_inventory, demand_history):
        order_qty = self.target_level - current_inventory
        return float(max(0.0, order_qty))


class SilverMealRule(BaseRule):
    """Silver-Meal Heuristic: Balance setup and holding costs."""

    def __init__(self, fixed_cost=50, holding_cost=2):
        super().__init__("SilverMeal")
        self.K = fixed_cost
        self.h = holding_cost

    def get_action(self, current_inventory, demand_history):
        if len(demand_history) < 3:
            return float(max(0.0, 20 - current_inventory))

        avg_demand = float(np.mean(demand_history[-5:]))
        if avg_demand <= 0:
            return 0.0

        T = np.sqrt(2 * self.K / (self.h * avg_demand))
        T = max(1, int(round(T)))
        target = avg_demand * T
        return float(max(0.0, target - current_inventory))


class EOQRule(BaseRule):
    """Economic Order Quantity: Wilson formula based on demand estimate."""

    def __init__(self, fixed_cost=50, holding_cost=2, reorder_point=12):
        super().__init__("EOQ")
        self.K = fixed_cost
        self.h = holding_cost
        self.reorder_point = reorder_point

    def get_action(self, current_inventory, demand_history):
        if current_inventory > self.reorder_point:
            return 0.0

        if len(demand_history) < 3:
            avg_demand = 10.0
        else:
            avg_demand = float(np.mean(demand_history[-5:]))

        if avg_demand <= 0:
            return 0.0

        annual_demand = avg_demand * 52
        eoq = np.sqrt(2 * self.K * annual_demand / self.h)
        order = max(avg_demand, eoq / 12)
        return float(max(0.0, order))
