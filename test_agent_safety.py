from pathlib import Path
import json
import uuid

import pandas as pd

from refund_agent import (
    run_agent,
    resume_agent,
    SimulatedCrash,
)

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
STATE = ROOT / "state"


def main():
    scenarios = pd.read_csv(
        ROOT / "data" / "raw" / "agent_scenarios.csv",
        keep_default_na=False,
    )

    s1_message = scenarios.loc[
        scenarios["scenario_id"].eq("S1"), "message"
    ].iloc[0]

    # 1. Token-limit demonstration.
    token_result = run_agent(
        s1_message,
        scenario_id="S1_TOKEN_DEMO",
        token_limit=200,
    )

    assert token_result["status"] == "FALLBACK_TOKEN_LIMIT"
    assert token_result["tokens_used"] > 200

    token_tools = [
        step["tool_name"] for step in token_result["steps"]
    ]

    assert "issue_refund" not in token_tools
    assert "escalate_to_human" not in token_tools

    print("TOKEN-LIMIT DEMO")
    print(f"Status: {token_result['status']}")
    print(f"Model calls: {token_result['llm_calls']}")
    print(f"Tokens used: {token_result['tokens_used']}")
    print(f"Completed tools: {token_tools}")

    # 2. Timeout demonstration.
    timeout_result = run_agent(
        s1_message,
        scenario_id="S1_TIMEOUT_DEMO",
        time_limit_seconds=0,
    )

    assert timeout_result["status"] == "FALLBACK_TIMEOUT"
    assert timeout_result["llm_calls"] == 0
    assert timeout_result["tokens_used"] == 0
    assert timeout_result["steps"] == []

    print("\nTIMEOUT DEMO")
    print(f"Status: {timeout_result['status']}")
    print(f"Model calls: {timeout_result['llm_calls']}")
    print(f"Tokens used: {timeout_result['tokens_used']}")

    # 3. Crash after the second tool's checkpoint has been saved.
    run_id = f"crash_demo_{uuid.uuid4().hex}"
    crashed = False

    try:
        run_agent(
            s1_message,
            scenario_id="S1_CRASH_RESUME",
            run_id=run_id,
            crash_after_tools=2,
        )
    except SimulatedCrash:
        crashed = True
        print("\nCRASH-AND-RESUME DEMO")
        print("Simulated crash after the second tool checkpoint.")

    assert crashed, "The simulated crash did not occur"

    checkpoint = json.loads(
        (STATE / f"{run_id}.json").read_text(encoding="utf-8")
    )

    tools_before = checkpoint["tools_executed"]

    assert tools_before == ["get_order", "search_policy"]
    assert checkpoint["step_number"] == 2

    # Continue from the saved observation, not from the beginning.
    resumed = resume_agent(run_id)

    tools_after_resume = [
        step["tool_name"] for step in resumed["steps"]
    ]

    expected_tools = [
        "get_order",
        "search_policy",
        "calculate_refund",
        "issue_refund",
    ]

    assert tools_after_resume == expected_tools
    assert resumed["status"] == "REFUND"
    assert resumed["llm_calls"] == 5

    refund = resumed["steps"][-1]["observation"]["refund_inr"]
    assert abs(refund - 128.25) < 0.01

    print(f"Tools completed before crash: {tools_before}")
    print(f"Tools completed after resume: {tools_after_resume[2:]}")
    print(f"Full tool sequence: {tools_after_resume}")
    print(f"Final status: {resumed['status']}")
    print(f"Refund: INR {refund:.2f}")
    print(f"Total model calls: {resumed['llm_calls']}")

    # A completed run must return its saved result without new actions.
    ledger_path = OUTPUTS / "refund_ledger.jsonl"
    ledger_before = ledger_path.read_bytes()

    repeated_resume = resume_agent(run_id)

    assert repeated_resume == resumed
    assert ledger_path.read_bytes() == ledger_before

    print("Resuming the completed run caused no additional ledger action.")

    summary = {
        "token_demo_status": token_result["status"],
        "token_demo_calls": token_result["llm_calls"],
        "token_demo_tokens": token_result["tokens_used"],
        "timeout_demo_status": timeout_result["status"],
        "timeout_demo_calls": timeout_result["llm_calls"],
        "crash_resume_status": resumed["status"],
        "crash_resume_tools": tools_after_resume,
        "crash_resume_refund": refund,
        "crash_resume_calls": resumed["llm_calls"],
        "crash_resume_run_id": run_id,
    }

    (OUTPUTS / "agent_safety_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nSaved: outputs/agent_safety_summary.json")
    print("AGENT SAFETY CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()