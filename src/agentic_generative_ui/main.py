#!/usr/bin/env python
"""
Agentic Generative UI flow for CrewAI Enterprise that fits AG‑UI's webhook format.
Uses **CopilotKitFlow** helpers only (no `copilotkit_stream`).
Simulates task execution step‑by‑step, emitting `tool_usage_started` /
`tool_usage_finished` so AG‑UI can stream progress.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from typing import Any, Dict, List, Optional
from datetime import datetime

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from crewai import LLM
from crewai.flow import persist, start, listen
from crewai.utilities.events.tool_usage_events import (
    ToolUsageStartedEvent,
    ToolUsageFinishedEvent,
)

from copilotkit.crewai import (
    CopilotKitFlow,
    FlowInputState,
    tool_calls_log,  # global log of tool‑usage events
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------
GENERATE_TASK_STEPS_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "generate_task_steps",
        "description": (
            "Make up 10 short gerund‑form steps needed to complete the task. "
            "Each step comes with a `pending` status that the human can toggle."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Task description"},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step_number": {"type": "integer"},
                            "description": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending"],
                                "description": "Always 'pending' on generation",
                            },
                        },
                        "required": ["step_number", "description"],
                    },
                    "description": "Ten task‑step objects",
                },
            },
            "required": ["task", "steps"],
        },
    },
}

SIMULATE_STEP_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "simulate_step",
        "description": "Simulate execution of a single task step",
        "parameters": {
            "type": "object",
            "properties": {
                "step_number": {"type": "integer", "description": "Step number being executed"},
                "description": {"type": "string", "description": "Step description"},
            },
            "required": ["step_number", "description"],
        },
    },
}

# ---------------------------------------------------------------------------
# Pydantic models and flow state
# ---------------------------------------------------------------------------
class TaskSteps(BaseModel):
    """Validated structure for generated steps."""

    task: str = Field(..., description="Task description")
    steps: List[Dict[str, Any]] = Field(..., description="Generated steps")


class AgentState(FlowInputState):
    """Conversation + task state shared across invocations."""

    task_steps: Optional[Dict[str, Any]] = None  # stored as plain dict for JSON
    simulated: bool = False
    latest_assistant_response: Optional[str] = None


# ---------------------------------------------------------------------------
# Flow implementation
# ---------------------------------------------------------------------------
@persist()
class AgenticGenerativeUIFlow(CopilotKitFlow[AgentState]):
    """Flow with two stages: chat + simulate."""

    # ----------------------- CHAT STAGE -------------------------------
    @start()
    def chat(self):
        """Main handler: may call tool; if steps exist, jump to simulate."""

        try:
            # ----- Build dynamic system prompt -----
            if self.state.task_steps is None:
                current_task_info = "No task steps created yet"
            else:
                formatted = [
                    (
                        "✅" if s.get("status", "pending") != "disabled" else "❌"
                    )
                    + f"  Step {s['step_number']}: {s['description']}" for s in self.state.task_steps.get("steps", [])
                ]
                current_task_info = f"Task: {self.state.task_steps['task']}\n" + "\n".join(formatted)

            system_prompt = f"""
You are a helpful assistant.
1. ALWAYS call `generate_task_steps` the first time a user asks to perform a task.
2. After the tool call and human toggling, craft a fun 1‑sentence summary (with emojis) of how you executed the enabled steps—never list all steps again.

Current task state ↓↓↓
{current_task_info}
"""
            messages = self.get_message_history(system_prompt=system_prompt)
            if getattr(self.state, "messages", None):
                for m in self.state.messages:
                    if m not in messages:
                        messages.append(m)

            llm = LLM(model="gpt-4o", stream=True)
            before_tool_calls = len(tool_calls_log)
            response_text = llm.call(
                messages=messages,
                tools=[GENERATE_TASK_STEPS_TOOL],
                available_functions={
                    "generate_task_steps": self.generate_task_steps_handler,
                },
            )

            final_response = self.handle_tool_responses(
                llm=llm,
                response_text=response_text,
                messages=messages,
                tools_called_count_before_llm_call=before_tool_calls,
            )

            # If we now have steps and haven't simulated them yet, queue simulation
            if self.state.task_steps and not self.state.simulated:
                self.state.latest_assistant_response = final_response
                return "route_simulate_task"

            # Otherwise, send final reply to user
            self._update_history_with_final(final_response)
            return json.dumps({"response": final_response, "id": self.state.id})

        except Exception as exc:
            logger.error("chat error: %s", exc)
            logger.debug(traceback.format_exc())
            return f"An error occurred: {exc}"

    # -------------------- SIMULATE STAGE ------------------------------
    @listen("route_simulate_task")
    def simulate_task(self):
        """Simulate task execution using LLM tool calls for each step."""

        try:
            assert self.state.task_steps, "No task steps to simulate"

            # Get enabled steps only
            enabled_steps = [
                step for step in self.state.task_steps["steps"]
                if step.get("status") != "disabled"
            ]

            if not enabled_steps:
                final_response = "No steps to execute! 🤷‍♂️"
                self._update_history_with_final(final_response)
                return json.dumps({"response": final_response, "id": self.state.id})

            # Create simulation prompt
            steps_list = "\n".join([
                f"{step['step_number']}. {step['description']}"
                for step in enabled_steps
            ])

            simulation_prompt = f"""
You must execute ALL of the following task steps by calling simulate_step for each one in order:

{steps_list}

IMPORTANT: Call simulate_step for EVERY single step above (steps 1-{len(enabled_steps)}), then provide a fun summary with emojis about completing the task.
"""

            # Use LLM to simulate steps via tool calls
            messages = [{"role": "user", "content": simulation_prompt}]
            llm = LLM(model="gpt-4o", stream=True)

            before_tool_calls = len(tool_calls_log)
            response_text = llm.call(
                messages=messages,
                tools=[SIMULATE_STEP_TOOL],
                available_functions={
                    "simulate_step": self.simulate_step_handler,
                },
            )

            final_response = self.handle_tool_responses(
                llm=llm,
                response_text=response_text,
                messages=messages,
                tools_called_count_before_llm_call=before_tool_calls,
            )

            self.state.simulated = True
            self._update_history_with_final(final_response)
            return json.dumps({"response": final_response, "id": self.state.id})

        except Exception as exc:
            logger.error("simulate_task error: %s", exc)
            logger.debug(traceback.format_exc())
            return f"An error occurred: {exc}"

    # ---------------- utility -----------------
    def _update_history_with_final(self, assistant_text: str):
        """Helper to push assistant message + any new user msgs to history."""
        for msg in getattr(self.state, "messages", []):
            if msg.get("role") == "user" and msg not in self.state.conversation_history:
                self.state.conversation_history.append(msg)
        self.state.conversation_history.append({"role": "assistant", "content": assistant_text})

    # ---------------- tool handlers -----------------
    def generate_task_steps_handler(self, task: str, steps: List[Dict[str, Any]]):
        task_steps = TaskSteps(task=task, steps=steps)
        self.state.task_steps = task_steps.model_dump()
        logger.info(f"Generated {len(steps)} steps for task: {task}")
        return f"✅ Generated {len(steps)} steps for: {task}"

    def simulate_step_handler(self, step_number: int, description: str):
        """Simulate execution of a single step with artificial delay."""
        time.sleep(1)  # Simulate work being done

        # Update the step status in state
        if self.state.task_steps:
            for step in self.state.task_steps["steps"]:
                if step["step_number"] == step_number:
                    step["status"] = "completed"
                    break

        logger.info(f"Simulated step {step_number}: {description}")
        return f"✅ Completed step {step_number}: {description}"

    def __repr__(self):  # pragma: no cover
        return json.dumps({"state": self.state.model_dump()}, indent=2)


# ---------------------------------------------------------------------------
# Local kickoff helper
# ---------------------------------------------------------------------------
def kickoff() -> int:  # pragma: no cover
    try:
        flow_instance = AgenticGenerativeUIFlow()
        user_msg = {"role": "user", "content": "Build a time machine!"}

        # Initial chat step (generates steps)
        result = flow_instance.kickoff({"messages": [user_msg], "task_steps": None})
        logger.info("First step result: %s", result)

        # If it returns route_simulate_task, manually trigger simulation
        if result == "route_simulate_task":
            logger.info("Triggering simulation...")
            simulation_result = flow_instance.simulate_task()
            logger.info("Simulation result: %s", simulation_result)
            return 0

        return 0
    except Exception as e:
        logger.error("Fatal during kickoff: %s", e)
        logger.debug(traceback.format_exc())
        return 1


# Flow entry for CrewAI CLI
flow = AgenticGenerativeUIFlow()

if __name__ == "__main__":  # pragma: no cover
    sys.exit(kickoff())
