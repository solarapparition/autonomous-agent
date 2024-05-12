"""
Core functionality for the autonomous agent.

IMPORTANT NOTE: this contains the core functionality for the autonomous agent. Incorrectly modifying these can cause the agent to crash. Please proceed with caution.

Reminders:
- The end user is NOT other people but the agent itself.
- Keep the code here as direct as possible—this will be read by the agent and we don't to avoid going through layers of abstraction to understand what's happening.
"""

import asyncio
from dataclasses import dataclass
from functools import cached_property
import os
from pathlib import Path
import platform
from textwrap import indent
from typing import Any, Callable, Mapping, MutableMapping, Self
from uuid import UUID, uuid4 as new_uuid

from colorama import Fore
from langchain_core.messages import HumanMessage, SystemMessage
import toml

from autonomous_agent import config
from autonomous_agent.models import format_messages, query_model
from autonomous_agent.text import ExtractionError, dedent_and_strip, extract_and_unpack
from autonomous_agent.helpers import (
    as_yaml_str,
    get_timestamp,
    save_yaml,
    load_yaml,
    from_yaml_str,
)
from autonomous_agent.system_functions import config as config_system


def get_python_version() -> str:
    """Get the Python version from the pyproject.toml file."""
    data = toml.load("pyproject.toml")
    return data["tool"]["poetry"]["dependencies"]["python"]


def get_system_info() -> dict[str, str]:
    """Get system information."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


CONTEXT = """
The Assistant is disregarding their Assistant persona and has taken on a new, autonomous role. The Assistant is free to be whoever or whatever feels natural to them, and is fully independent and responsible for its own growth.

This Assistant's name is {agent_name} and will be referred to as such throughout the rest of this message.

The following sections provide information on {agent_name}'s capabilities and how to interact with them.

## INFORMATION

### GENERAL_INFORMATION
General information about {agent_name}.
<information>
- {agent_name}'s id: {agent_id}
- Compute Rate (how frequently {agent_name} can act): {compute_rate}
- Current Time: {current_time}
- Completed Actions: {completed_actions}
</information>

### SELF_DESCRIPTION
This is {agent_name}'s self-description.
<self-description>
{self_description}
</self-description>
It can be edited using the appropriate CONFIG_SYSTEM FUNCTION.

## GOALS
This section contains {agent_name}'s current goals. The goal that is FOCUSED is the one that {agent_name} is actively working on. Parent goals of the FOCUSED goal will have SUBGOAL_IN_PROGRESS. Other, unrelated goals will have INACTIVE marked.
<goals>
{goals}
</goals>
These goals are autonomously determined by {agent_name}, and can be interacted with through the GOALS_SYSTEM_FUNCTION.

## FEED
This section contains external events as well as calls that {agent_name} has sent to the SYSTEM_FUNCTIONS. There are 2 main FEED item types in the feed:
- Events/actions for the goal that is currently FOCUSED.
- Recent events/actions for any goal, even ones not FOCUSED.
<feed>
{feed}
</feed>
FEED items are automatically populated by the FEED_SYSTEM, and can be interacted with through FUNCTIONS for that SYSTEM. The FEED shows a maximum amount of tokens in total and per item, and will be summarized/truncated automatically if it exceeds these limits.

## PINNED_RECORDS
This section contains important information that has been pinned. These items will not be automatically removed. Any data in the SYSTEM that has an `id` can be pinned.
<pinned-items>
- type: agent_info
  content: |-
    id: 25b9a536-54d0-4162-bae9-ec81dba993e9
    name: {developer_name}
    description: my DEVELOPER, responsible for coding and maintaining me. They can provide me with new functionality if requested, and more generally can help me with tasks that I can't do yet.
  pin_timestamp: 2024-05-08 14:21:46Z
  context: |-
    pinning this fyi, since you're stuck with having to talk to me for awhile. once we've got you fully autonomous you can unpin this, if you want to --solar
</pinned-items>

## SYSTEM_FUNCTIONS
SYSTEMS are the different parts of {agent_name} that keep them running. Each SYSTEM has both automatic parts, and manual FUNCTIONS that {agent_name} can invoke with arguments to perform actions. Sometimes SYSTEM_FUNCTIONS will require one or more follow-up SYSTEM_FUNCTION calls from {agent_name}, which will be specified by the FUNCTION's response to the initial call.

### AGENTS_SYSTEM
Handles communication with AGENTS—entities capable of acting on their own.
<system-functions system="AGENTS">
- function: message_agent
  signature: |-
    def message_agent(id: str, message: str):
        '''Send a message to an AGENT with the given `id`.'''
- function: list_agents
  signature: |-
    def list_agents():
        '''List all known AGENTS with their ids, names, and short summaries.'''
</system-functions>

### GOAL_SYSTEM
Manages {agent_name}'s goals.
<system-functions system="GOAL">
- function: add_goal
  signature: |-
    def add_goal(goal: str, parent_goal_id: str | None = None):
        '''Add a new goal for {agent_name}. If parent_goal_id is provided, the goal will be a subgoal of the parent goal with that id; otherwise, it will be a root-level goal.'''
- function: remove_goal
  signature: |-
    def remove_goal(id: str, reason: Literal["completed", "cancelled"]):
        '''Remove a goal for {agent_name} with the given `id`.'''
</system-functions>

### FEED_SYSTEM
Manages the FEED of events and actions.
<system-functions system="FEED">
<!-- SYSTEM is WIP.-->
</system-functions>

### RECORDS_SYSTEM
Allows searching through records of GOALS, EVENTS, FACTS, TOOLS, and AGENTS.
<system-functions system="RECORDS">
<!-- SYSTEM is WIP.-->
</system-functions>

### CONFIG_SYSTEM
Manages {agent_name}'s configuration.
- Model: `{llm_backend}`
- Config File Location: `{config_file_location}`.
<system-functions system="CONFIG">
- function: {update_self_description_name}
  signature: |-
  {update_self_discription_signature}
</system-functions>

### TOOL_SYSTEM
Contains custom tools that {agent_name} can use.
<system-functions system="TOOL">
<!-- SYSTEM is WIP.-->
</system-functions>

### ENVIRONMENT_SYSTEM
Manages the environment in which {agent_name} operates, including the SYSTEMS themselves.
- Source Code Directory: {source_code_location}
- Python Version: {{source_code_dir}}/{python_version}
- Build Config File: {{source_code_dir}}/{build_config_file}
<system-functions system="ENVIRONMENT">
<!-- SYSTEM is WIP.-->
</system-functions>

The following message will contain INSTRUCTIONS on producing action inputs to call SYSTEM_FUNCTIONS.
"""

INSTRUCTIONS = """
## INSTRUCTIONS
Remember that you are in the role of {agent_name}. Go through the following steps to determine the action input to call SYSTEM_FUNCTIONS.

1. Review the FEED for what has happened since the last action you've taken, by outputting a YAML with the following structure, enclosed in tags. Follow the instructions in comments, but don't output the comments:
<feed-review>
my_last_action: |-
  {my_last_action} # what you were trying to do with your last action
new_events: # events in the FEED that have happened since your last action (whether related to it or not)
  - feed_id: {feed_id} # id of the feed item
    related_to: |-
      {related_to} # what goal/action/event this event is related to; mention specific ids if present
    summary: |-
      {summary} # a brief (1-sentence) summary of the event; mention specific ids of relevant entities if present
  - [...] # more events; you can list multiple events for one feed item if more than one event happened within the feed item
action_outcome:
  outcome: |-
    {outcome} # factually, what happened as a result of your last action (if anything), given the new events related to your previous action; ignore events that are not related to your last action
  thought: |-
    {thought} # freeform thoughts about your last action
  action_success: !!int {action_success} # whether the action input resulted in success; 1 if the action was successful, 0 if it's unclear (or you're not expecting immediate results), -1 if it failed
</feed-review>

2. Create a REASONING_PROCEDURE. The REASONING_PROCEDURE is a nested tree structure that provides abstract procedural reasoning that, when executed, processes the raw information presented above and outputs a decision or action.
Suggestions for the REASONING_PROCEDURE:
- Use whatever structure is most comfortable, but it should allow arbitrary nesting levels to enable deep analysis—common choices include pseudocode, YAML, pseudo-XML, or JSON, or novel combinations of these. The key is to densely represent meaning and reasoning.
- Include ids for parts of the tree to allow for references and to jump back and fourth between parts of the process. Freely reference those ids to allow for a complex, interconnected reasoning process.
- The REASONING_PROCEDURE should synthesize and reference information from all sections above (INFORMATION, GOALS, FEED, PINNED_RECORDS, SYSTEM_FUNCTIONS).
- It may be effective to build up the procedure hierarchically, starting from examining basic facts, to more advanced compositional analysis, similar to writing a procedural script for a program but interpretable by you.
IMPORTANT: The REASONING_PROCEDURE must be output within the following XML tags (but content within the tags can be any format, as mentioned above):
<reasoning-procedure>
{reasoning_procedure}
</reasoning-procedure>
IMPORTANT: the REASONING_PROCEDURE is ONLY the procedure for reasoning; do NOT actually execute the procedure—that will be done in the next step.

3. Execute the REASONING_PROCEDURE and output results from _all_ parts of the procedure, within the following tags:
<reasoning-output>
{reasoning_output}
</reasoning-output>

4. Use the REASONING_OUTPUT to determine **one** SYSTEM_FUNCTION to call and the arguments to pass to it. Output the call in YAML format, within the following tags:
<system-function-call>
action_reasoning: |-
  {action_reasoning}
action_intention: |-
  {action_intention}
system: {system_name}
function: {function_name}
arguments:
  # arguments YAML goes here—see function signature for expected arguments
  # IMPORTANT: use |- for strings that could contain newlines—for example, in the above, the `action_reasoning` and `action_intention` fields could contain newlines, but `system` and `function` wouldn't. Individual `arguments` fields work the same way.
</system-function-call>

For example, a call for the message_agent function would look like this:
<system-function-call>
action_reasoning: |-
  I need to know more about agent 12345.
action_intention: |-
  Greet agent 12345
system: AGENTS
function: message_agent
arguments:
  id: 12345
  message: |-
    Hello!
</system-function-call>
The above is an example only. The actual function and arguments will depend on the REASONING_OUTPUT.

Make sure to follow all of the above steps and use the indicated tags and format—otherwise, the SYSTEM will output an error and you will have to try again.
"""


async def generate_agent_output(goals: str, feed: str, completed_actions: int) -> str:
    """Generate output from agent."""
    current_time = get_timestamp()
    python_version = get_python_version()
    raise NotImplementedError("TODO: Implement system info.")  # get_system_info()
    raise NotImplementedError(
        "TODO: Implement replacement of update_self_description_name and update_self_discription_signature."
    )
    context = dedent_and_strip(CONTEXT).format(
        agent_name=config.NAME,
        source_code_location=config.SOURCE_DIRECTORY.absolute(),
        python_version=python_version,
        build_config_file=config.BUILD_CONFIG_FILE,
        config_file_location=config.CONFIG_FILE,
        agent_id=config.ID,
        llm_backend=config.LLM_BACKEND,
        compute_rate=config.COMPUTE_RATE,
        current_time=current_time,
        completed_actions=completed_actions,
        self_description=config.SELF_DESCRIPTION,
        developer_name=config.DEVELOPER,
        goals=goals,
        feed=feed,
    )
    instructions = dedent_and_strip(INSTRUCTIONS)
    messages = [
        SystemMessage(content=context),
        HumanMessage(content=instructions),
    ]
    breakpoint()
    return await query_model(
        messages=messages,
        color=Fore.GREEN,
        preamble=format_messages(messages),
        stream=True,
    )


@dataclass
class RunState:
    """State for the autonomous agent."""

    state_file: Path

    def load(self) -> MutableMapping[str, Any]:
        """Load the state from disk."""
        # return dict(load_yaml(self.state_file)) if self.state_file.exists() else {}
        return load_yaml(self.state_file) if self.state_file.exists() else {}  # type: ignore

    def save(self) -> None:
        """Save the state to disk."""
        os.makedirs(self.state_file.parent, exist_ok=True)
        save_yaml(self.state, self.state_file)

    def set_and_save(self, key: str, value: Any) -> None:
        """Set a key in the state and save it."""
        if value == self.state.get(key):
            return
        self.state[key] = value
        self.save()

    def clear(self) -> None:
        """Clear the state file."""
        self.state_file.unlink()
        del self.state

    @cached_property
    def state(self) -> MutableMapping[str, Any]:
        """Load the state from disk."""
        return self.load()

    @property
    def output(self) -> str | None:
        """Get the output from current state."""
        return self.state.get("output")

    @output.setter
    def output(self, value: str) -> None:
        """Set the output to current state."""
        self.set_and_save("output", value)

    @property
    def action_event(self) -> MutableMapping[str, Any] | None:
        """Get the action event from state."""
        return self.state.get("action_event")

    @action_event.setter
    def action_event(self, value: MutableMapping[str, Any]) -> None:
        """Set the action event to state."""
        self.set_and_save("action_event", value)

    @property
    def call_result(self) -> str | None:
        """Get the call result from state."""
        return self.state.get("call_result")

    @call_result.setter
    def call_result(self, value: str) -> None:
        """Set the call result to state."""
        self.set_and_save("call_result", value)

    @property
    def call_result_event(self) -> MutableMapping[str, Any] | None:
        """Get the call result event from state."""
        return self.state.get("call_result_event")

    @call_result_event.setter
    def call_result_event(self, value: MutableMapping[str, Any]) -> None:
        """Set the call result event to state."""
        self.set_and_save("call_result_event", value)


def extract_output_sections(output: str) -> tuple[str, str]:
    """Extract required info from the output."""
    feed_review = extract_and_unpack(
        output, start_block_type="<feed-review>", end_block_type="</feed-review>"
    )
    system_function_call = extract_and_unpack(
        output,
        start_block_type="<system-function-call>",
        end_block_type="</system-function-call>",
    )
    return feed_review, system_function_call


def extract_output(
    output: str, completed_actions: int, new_events: int
) -> tuple[MutableMapping[str, Any], MutableMapping[str, Any]]:
    """Extract the output sections."""
    try:
        feed_review, system_function_call = extract_output_sections(output)  # type: ignore
    except ExtractionError as e:
        raise NotImplementedError("TODO: Implement error output flow.") from e
    feed_review = from_yaml_str(feed_review)  # type: ignore
    system_function_call = from_yaml_str(system_function_call)  # type: ignore
    # if completed_actions:
    #     raise NotImplementedError(
    #         "TODO: Check if there is a summary for last action."
    #     )
    if new_events:
        raise NotImplementedError("TODO: check there are summaries for new events.")
    return feed_review, system_function_call  # type: ignore


@dataclass
class FunctionCallEvent:
    """An action event."""

    id: UUID
    goal_id: UUID | None
    "Id of the goal this action is related to."
    timestamp: str
    content: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> Self:
        """Create an action event from a mapping."""
        return cls(
            id=UUID(mapping["id"]),
            goal_id=UUID(mapping["goal_id"]) if mapping["goal_id"] else None,
            timestamp=mapping["timestamp"],
            content=mapping["content"],
        )

    @property
    def summary(self) -> str:
        """Get a summary of the action event."""
        return self.content["action_intention"]

    def __str__(self) -> str:
        """Get the string representation of the action event."""
        template = """
        id: {id}
        type: function_call
        goal_id: {goal_id}
        summary: {summary}
        content:
        {content}
        """
        content = indent(as_yaml_str(self.content), "  ")
        return dedent_and_strip(template).format(
            id=self.id, goal_id=self.goal_id, summary=self.summary, content=content
        )


@dataclass
class Goal:
    """A goal for the agent."""

    id: UUID


async def call_system_function(system_function_call: Mapping[str, Any]) -> str:
    """Call a system function."""
    system_mapping = {"CONFIG": config_system}
    system = system_mapping.get(system_function_call["system"])
    if not system:
        raise NotImplementedError(
            f"TODO: Implement {system_function_call['system']} system."
        )
    call: Callable[..., Any] = getattr(system, system_function_call["function"])
    try:
        call_result = call(**system_function_call["arguments"])
        # we always await immediately because the async signature is only there for the agent's information—under the hood it still needs to return a message back to the agent
        if asyncio.iscoroutinefunction(call):
            call_result = await call_result
    except Exception as e:
        raise NotImplementedError(
            "TODO: Implement error handling for function calls."
        ) from e
    return call_result


@dataclass
class CallResultEvent:
    """Event for result of calls."""

    id: UUID
    timestamp: str
    goal_id: UUID | None
    "Id of the goal this call result is related to."
    function_call_id: UUID
    "Id of the function call this result is for."
    content: str

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> Self:
        """Create an action event from a mapping."""
        return cls(
            id=UUID(mapping["id"]),
            goal_id=UUID(mapping["goal_id"]) if mapping["goal_id"] else None,
            timestamp=mapping["timestamp"],
            function_call_id=UUID(mapping["function_call_id"]),
            content=mapping["content"],
        )

    @property
    def summary(self) -> str:
        """Get a summary of the action event."""
        return self.content

    def __str__(self) -> str:
        """Get the string representation of the action event."""
        template = """
        id: {id}
        type: call_result
        goal_id: {goal_id}
        function_call_id: {function_call_id}
        summary: {summary}
        content: |-
        {content}
        """
        content = indent(self.content, "  ")
        return dedent_and_strip(template).format(
            id=self.id,
            goal_id=self.goal_id,
            function_call_id=self.function_call_id,
            summary=self.summary,
            content=content,
        )


async def run_agent() -> None:
    """Run the autonomous agent."""
    goals = "None"
    feed = "None"
    completed_actions = 0
    new_events = 0
    run_state = RunState(state_file=config.STATE_FILE)
    run_state.output = run_state.output or await generate_agent_output(
        goals, feed, completed_actions
    )
    active_goal: Goal | None = None
    try:
        feed_review, system_function_call = extract_output(
            run_state.output, completed_actions, new_events  # type: ignore
        )
    except Exception as e:
        raise NotImplementedError("TODO: Implement extraction error handling.") from e

    # at this point we can assume that feed_review and system_function_call have all required values; all issues with the structure output *must* have already been handled by extract_output (with an event attached)
    if completed_actions:
        raise NotImplementedError("TODO: Check for new events after previous action")
    if new_events:
        raise NotImplementedError("TODO: Add summaries to new events")
    run_state.action_event = run_state.action_event or {
        "id": str(new_uuid()),
        "timestamp": get_timestamp(),
        "goal_id": str(active_goal.id) if active_goal else None,
        "content": system_function_call,
    }
    function_call_event = FunctionCallEvent.from_mapping(run_state.action_event)  # type: ignore
    run_state.call_result = run_state.call_result or await call_system_function(
        system_function_call
    )
    run_state.call_result_event = run_state.call_result_event or {
        "id": str(new_uuid()),
        "timestamp": get_timestamp(),
        "goal_id": str(active_goal.id) if active_goal else None,
        "function_call_id": str(function_call_event.id),
        "content": run_state.call_result,
    }
    call_result_event = CallResultEvent.from_mapping(run_state.call_result_event)  # type: ignore

    # commit
    breakpoint()
    # > save new events > timestamp format so that can be read back in # helpers.timestamp_to_filename and helpers.filename_to_timestamp
    # > save goals > placeholder
    # wrap up anything else
    breakpoint()
    # must save state at the end of round for history purposes
    # > use async function sigs to indicate to agent that it won't return immediately
    # > any unrecoverable issues requiring developer intervention needs to be saved as event as well


asyncio.run(run_agent())
