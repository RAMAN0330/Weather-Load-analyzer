"""
run_report.py — Orchestrator.
1. Builds parquet cache (data_loader)
2. Runs Agents 1-3 in parallel (ThreadPoolExecutor)
3. Runs Agent 4 (evening + synthesis) after 1-3 complete
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from concurrent.futures import ThreadPoolExecutor, as_completed
from scripts.data_loader import build_cache
from scripts import agent_night, agent_morning, agent_afternoon, agent_evening


def run_agent(agent_module):
    name = agent_module.__name__.split('.')[-1]
    try:
        print(f"[orchestrator] Starting {name}...")
        agent_module.main()
        print(f"[orchestrator] {name} DONE")
        return name, None
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[orchestrator] {name} FAILED: {e}")
        return name, e


def main():
    t0 = time.time()
    print("=" * 60)
    print("Haryana Monsoon Weather-Load Report — Full Run")
    print("=" * 60)

    # Phase 1: Build cache
    print("\n[Phase 1] Building data cache...")
    build_cache(force=False)

    # Phase 2: Run agents 1-3 in parallel
    print("\n[Phase 2] Running agents 1-3 in parallel...")
    parallel_agents = [agent_night, agent_morning, agent_afternoon]
    errors = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(run_agent, a): a for a in parallel_agents}
        for fut in as_completed(futures):
            name, err = fut.result()
            if err:
                errors.append((name, err))

    if errors:
        print(f"\n[orchestrator] ERROR: {len(errors)} agent(s) failed:")
        for name, err in errors:
            print(f"  {name}: {err}")
        sys.exit(1)

    # Phase 3: Agent 4 (evening + synthesis)
    print("\n[Phase 3] Running agent 4 (evening + synthesis)...")
    agent_evening.main()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Report complete in {elapsed:.1f}s")
    print(f"Outputs in: reports/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
