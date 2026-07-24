# fl_1/server_app.py

from typing import Dict, List, Tuple
import flwr as fl

# =========================
# METRIC AGGREGATION
# =========================
def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]):
    total = sum(n for n, _ in metrics)
    agg = {}
    for n, m in metrics:
        for k, v in m.items():
            agg[k] = agg.get(k, 0.0) + n * v
    return {k: v / total for k, v in agg.items()}


# =========================
# SERVER APP (TANPA client_fn)
# =========================
app = fl.server.ServerApp(
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=10,
        min_evaluate_clients=10,
        min_available_clients=10,
        evaluate_metrics_aggregation_fn=weighted_average,
    ),
)
