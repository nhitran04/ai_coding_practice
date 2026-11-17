"""Compare passive learners on the same fixed-policy MiniGrid task.

Examples:
    python q1-comparison.py --algorithms adp direct-utility python
    q1-comparison.py --algorithms adp direct-utility td --episodes 100

TD must be implemented in q1-passive-td.py before selecting it. Missing utility
estimates count as zero. Errors are measured on the fixed policy's path from the
environment's start; unvisited off-path states are excluded explicitly.
"""

import argparse
import ast
import csv
import importlib.util
import math
import random
import sys
from functools import partial
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ALGORITHMS = {
    "adp": ("q1-passive-adp-agent.py", "PassiveADPAgent"),
    "direct-utility": (
        "q1-passive-direct-utility-estimate.py",
        "PassiveDirectUtilityAgent",
    ),
    "td": ("q1-passive-td.py", "PassiveTDAgent"),
}


def load_algorithm(name):
    """Import a sibling script, rejecting unfinished agent skeletons early."""
    filename, class_name = ALGORITHMS[name]
    path = Path(__file__).parent / filename
    tree = ast.parse(path.read_text())
    agent_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    for method in agent_node.body:
        if isinstance(method, ast.FunctionDef):
            statements = [
                node
                for node in method.body
                if not (
                    isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                )
            ]
            if not statements or all(
                isinstance(node, ast.Pass)
                or (
                    isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Constant)
                    and node.value.value is Ellipsis
                )
                for node in statements
            ):
                raise ValueError(
                    f"{name}: implement {class_name}.{method.name} in {filename} "
                    "before comparing it. Select implemented agents with "
                    "--algorithms adp direct-utility."
                )
    module_name = f"q1_comparison_{name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through the registered module.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, getattr(module, class_name)


def reference_utilities(width, height, gamma):
    """Exact returns for east-then-north moves and a reward of one on
    arrival."""
    states = [(x, height) for x in range(1, width + 1)]
    states.extend((width, y) for y in range(height - 1, 0, -1))
    transitions = len(states) - 1
    return {
        state: (gamma ** (transitions - index - 1) if index < transitions else 0.0)
        for index, state in enumerate(states)
    }


def compare(
    algorithms, episodes, runs, seed, width, height, max_steps, gamma, output_dir
):
    """Train independent agents under identical deterministic episode
    conditions."""
    if episodes < 1 or runs < 1:
        raise ValueError("episodes and runs must be positive")
    if width < 1 or height < 1 or width * height < 2:
        raise ValueError("Grid dimensions must be positive with at least two cells")
    if not 0 <= gamma < 1:
        raise ValueError("gamma must be at least zero and less than one")
    # Complete episodes are required for direct utility estimation's samples.
    if max_steps < width + height - 2:
        raise ValueError("max-steps must allow the fixed policy to reach the goal")
    loaded = {name: load_algorithm(name) for name in algorithms}
    reference = reference_utilities(width, height, gamma)
    records = []
    for name, (module, agent_class) in loaded.items():
        for run in range(runs):
            random.seed(seed + run)
            env = module.SimpleEnv(width=width, height=height, max_steps=max_steps)
            try:
                agent = agent_class(
                    partial(module.fixed_policy, goal=env.goal_pos),
                    discount_factor=gamma,
                )
                total_steps = 0
                learning_seconds = 0.0
                for episode in range(episodes + 1):
                    steps = 0
                    if episode:
                        env.reset(seed=seed + run * episodes + episode - 1)
                        agent.reset_episode()
                        started = perf_counter()
                        action = agent.act((tuple(env.agent_pos), 0.0, False))
                        learning_seconds += perf_counter() - started
                        while True:
                            if action is None:
                                raise ValueError(
                                    f"{name} returned no action before termination"
                                )
                            _, _, terminated, truncated, _ = env.step(action)
                            steps += 1
                            started = perf_counter()
                            action = agent.act(
                                (tuple(env.agent_pos), float(terminated), terminated)
                            )
                            learning_seconds += perf_counter() - started
                            if terminated or truncated:
                                agent.reset_episode()
                                if not terminated:
                                    raise ValueError(
                                        f"{name} failed to complete the fixed-policy path"
                                    )
                                break
                        total_steps += steps
                    errors = [
                        agent.U.get(s, 0.0) - value for s, value in reference.items()
                    ]
                    records.append(
                        {
                            "algorithm": name,
                            "run": run + 1,
                            "seed": seed + run,
                            "episode": episode,
                            "environment_steps": total_steps,
                            "path_rmse": math.sqrt(
                                sum(e * e for e in errors) / len(errors)
                            ),
                            "start_utility": agent.U.get((1, height), 0.0),
                            "exact_start_utility": reference[(1, height)],
                            "path_coverage": sum(s in agent.U for s in reference)
                            / len(reference),
                            "learning_seconds": learning_seconds,
                        }
                    )
            finally:
                env.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "learning.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (output_dir / "reference.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "exact_utility"])
        writer.writerows((x, y, value) for (x, y), value in reference.items())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for name in algorithms:
        means = []
        times = []
        for episode in range(episodes + 1):
            rows = [
                r for r in records if r["algorithm"] == name and r["episode"] == episode
            ]
            means.append(sum(r["path_rmse"] for r in rows) / runs)
            times.append(sum(r["learning_seconds"] for r in rows) / runs)
        axes[0].plot(range(episodes + 1), means, label=name)
        axes[1].plot(range(episodes + 1), times, label=name)
        print(
            f"{name}: final path RMSE={means[-1]:.6f}; mean learning time={times[-1]:.4f}s"
        )
    axes[0].set(ylabel="Utility RMSE on fixed-policy path", title="Utility learning")
    axes[1].set(
        ylabel="Cumulative agent processing time (seconds)", title="Learning cost"
    )
    for ax in axes:
        ax.set_xlabel("Completed episodes")
        ax.legend()
        ax.grid(alpha=0.25)
    fig.suptitle(
        f"{width}×{height}, gamma={gamma}, {runs} run(s); fixed east-then-north policy"
    )
    fig.savefig(output_dir / "comparison.png", dpi=150)
    plt.close(fig)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=ALGORITHMS,
        default=list(ALGORITHMS),
        help="Agents to compare (default: all three)",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--height", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "outputs" / "comparison",
    )
    args = parser.parse_args()
    args.algorithms = list(dict.fromkeys(args.algorithms))
    try:
        compare(**vars(args))
    except ValueError as error:
        parser.error(str(error))
    print(f"Results saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
