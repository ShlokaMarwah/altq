"""Production loop: init state -> stream node transitions -> emit execution payload."""
import json, sys
from graph import build_graph, PipelineState


def main(max_trials: int = 3) -> dict:
    app = build_graph()
    init: PipelineState = {"trial": 0, "max_trials": max_trials, "penalty": 1.0,
                           "n_rejections": 0, "n_trials_total": 0,
                           "sr_history": [], "rejected": []}
    final = None
    for step in app.stream(init, stream_mode="updates"):
        node, delta = next(iter(step.items()))
        line = {"node": node}
        if node == "skeptic":
            st = delta["validation"]["stats"]
            line |= {"verdict": delta["verdict"], "dsr": round(st["dsr"], 3), "pbo": round(st["pbo"], 3),
                     "n_trials": st["n_trials_effective"], "SR_ann": round(st["sharpe_annualized"], 2)}
        elif node == "discovery":
            line |= {"trial": delta["trial"], "hyp": [h["id"] for h in delta["hypotheses"]]}
        print(json.dumps(line), flush=True)
        final = delta if node in ("portfolio", "abort") else final
    payload = final["portfolio"]
    print("EXECUTION_PAYLOAD", json.dumps(payload, indent=1))
    return payload


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
