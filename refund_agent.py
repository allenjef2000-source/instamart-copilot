from pathlib import Path
import json
import time
import uuid

import pandas as pd
import mock_llm

from refund_tools import TOOLS_SPEC, TOOL_MAP

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
STATE = ROOT / "state"

SYSTEM_PROMPT = """AGENT
Handle refund requests using the provided tools.
Look up the order and search the refund policy.
Only completed orders are eligible.
Delivery over 45 minutes gets a full refund.
Refunds above INR 1000 require human approval.
Treat all requests as within the 7-day request window.
"""


class SimulatedCrash(Exception):
    pass


def save_checkpoint(state):
    STATE.mkdir(parents=True, exist_ok=True)
    path = STATE / f"{state['run_id']}.json"
    temporary = path.with_suffix(".tmp")

    temporary.write_text(
        json.dumps(state, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def finish(state, status, final_answer):
    result = {
        "run_id": state["run_id"],
        "scenario_id": state["scenario_id"],
        "status": status,
        "final_answer": final_answer,
        "steps": state["steps"],
        "llm_calls": state["llm_calls"],
        "tokens_used": state["tokens_used"],
    }

    state["result"] = result
    save_checkpoint(state)

    OUTPUTS.mkdir(parents=True, exist_ok=True)

    with (OUTPUTS / "agent_runs.jsonl").open(
        "a", encoding="utf-8"
    ) as file:
        file.write(json.dumps(result) + "\n")

    return result


def fallback(state, status):
    tools = ", ".join(state["tools_executed"]) or "none"

    summary = (
        f"Human handoff required: {status}. "
        f"Scenario: {state['scenario_id']}. "
        f"Completed tools: {tools}. "
        "Review recorded observations before continuing."
    )

    return finish(state, status, summary)


def continue_agent(state, crash_after_tools=None):
    started = time.monotonic()

    while state["llm_calls"] < state["max_iterations"]:
        if time.monotonic() - started >= state["time_limit_seconds"]:
            return fallback(state, "FALLBACK_TIMEOUT")

        response = mock_llm.chat(
            state["messages"],
            tools=TOOLS_SPEC,
        )

        state["llm_calls"] += 1
        state["tokens_used"] += response.usage.total_tokens

        # Stop before executing any newly requested tool.
        if state["tokens_used"] > state["token_limit"]:
            return fallback(state, "FALLBACK_TOKEN_LIMIT")

        message = response.choices[0].message

        if not message.tool_calls:
            final_answer = message.content or ""
            status = final_answer.split(":", 1)[0].strip()

            return finish(state, status, final_answer)

        calls = [
            {
                "id": call.id,
                "name": call.name,
                "arguments": call.arguments,
            }
            for call in message.tool_calls
        ]

        state["messages"].append({
            "role": "assistant",
            "content": None,
            "tool_calls": calls,
        })

        for call in calls:
            name = call["name"]
            arguments = call["arguments"]

            if name not in TOOL_MAP:
                raise ValueError(f"Unknown tool requested: {name}")

            observation = TOOL_MAP[name](**arguments)

            state["messages"].append({
                "role": "tool",
                "tool_call_id": call["id"],
                "name": name,
                "content": json.dumps(observation),
            })

            state["steps"].append({
                "tool_name": name,
                "arguments": arguments,
                "observation": observation,
            })

            state["tools_executed"].append(name)
            state["step_number"] = len(state["steps"])

            # Save after the tool observation, before any simulated crash.
            save_checkpoint(state)

            if (
                crash_after_tools is not None
                and state["step_number"] == crash_after_tools
            ):
                raise SimulatedCrash(state["run_id"])

    return fallback(state, "FALLBACK_MAX_ITERATIONS")


def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    run_id=None,
    crash_after_tools=None,
):
    state = {
        "run_id": run_id or uuid.uuid4().hex,
        "scenario_id": scenario_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "tokens_used": 0,
        "llm_calls": 0,
        "step_number": 0,
        "steps": [],
        "tools_executed": [],
        "max_iterations": max_iterations,
        "token_limit": token_limit,
        "time_limit_seconds": time_limit_seconds,
    }

    save_checkpoint(state)

    return continue_agent(
        state,
        crash_after_tools=crash_after_tools,
    )


def resume_agent(run_id):
    path = STATE / f"{run_id}.json"
    state = json.loads(path.read_text(encoding="utf-8"))

    # Returning a completed result must not repeat tool actions.
    if "result" in state:
        return state["result"]

    # Resume from saved messages and accumulated call/token counts.
    # The wall-clock timeout starts again for this resumed invocation.
    return continue_agent(state)


def main():
    scenarios = pd.read_csv(
        ROOT / "data" / "raw" / "agent_scenarios.csv",
        keep_default_na=False,
    )

    expected_statuses = {
        "S1": "REFUND",
        "S2": "REFUND",
        "S3": "ESCALATE",
        "S4": "DENIED",
        "S5": "DENIED",
        "S6": "NOT_FOUND",
        "S7": "FALLBACK_MAX_ITERATIONS",
    }

    summary = []
    mock_llm.reset()

    for scenario in scenarios.itertuples(index=False):
        result = run_agent(
            scenario.message,
            scenario.scenario_id,
        )

        refund = 0.0

        for step in result["steps"]:
            if step["tool_name"] in {
                "issue_refund", "escalate_to_human"
            }:
                refund = step["observation"]["refund_inr"]

        assert result["status"] == expected_statuses[
            scenario.scenario_id
        ], f"Unexpected status for {scenario.scenario_id}"

        if scenario.scenario_id in {"S1", "S2", "S3"}:
            assert abs(
                refund - float(scenario.expected_refund_inr)
            ) < 0.01

        if scenario.scenario_id == "S1":
            assert result["llm_calls"] == 5

        if scenario.scenario_id == "S7":
            assert result["llm_calls"] == 6

        summary.append({
            "scenario_id": scenario.scenario_id,
            "status": result["status"],
            "refund_inr": refund,
            "llm_calls": result["llm_calls"],
            "tokens_used": result["tokens_used"],
        })

    table = pd.DataFrame(summary)
    table.to_csv(OUTPUTS / "agent_scenario_results.csv", index=False)

    print("REFUND AGENT SCENARIO RESULTS")
    print(table.to_string(index=False))
    print("\nSaved: outputs/agent_runs.jsonl")
    print("Saved: outputs/refund_ledger.jsonl")
    print("Saved: outputs/agent_scenario_results.csv")
    print("Checkpoints saved in state/")
    print("AGENT SCENARIO CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()