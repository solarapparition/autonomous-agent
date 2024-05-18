"""
Core functionality for the autonomous Mind.

IMPORTANT NOTE: this contains the core functionality for the autonomous Mind. Incorrectly modifying these can cause the Mind to crash. Please proceed with caution.
"""

import asyncio
from dataclasses import dataclass
from functools import cached_property
import os
from pathlib import Path
from textwrap import indent
from typing import Any, Callable, Literal, Mapping, MutableMapping, Sequence
from uuid import UUID, uuid4 as new_uuid

from colorama import Fore
from langchain_core.messages import HumanMessage, SystemMessage
import toml

from autonomous_mind import config
from autonomous_mind.models import format_messages, query_model
from autonomous_mind.schema import CallResultEvent, Event, FunctionCallEvent
from autonomous_mind.systems.feed.events import Feed, save_events
from autonomous_mind.systems.goal.goals import Goals
from autonomous_mind.text import ExtractionError, dedent_and_strip, extract_and_unpack
from autonomous_mind.helpers import (
    LONG_STR_YAML,
    Timestamp,
    as_yaml_str,
    get_timestamp,
    from_yaml_str,
    load_yaml,
    save_yaml,
    timestamp_to_filename,
    get_machine_info,
)
from autonomous_mind.systems.config import functions as config_functions
from autonomous_mind.systems.goal import functions as goal_functions
from autonomous_mind.systems.knowledge import functions as knowledge_functions


def get_python_version() -> str:
    """Get the Python version from the pyproject.toml file."""
    data = toml.load("pyproject.toml")
    return data["tool"]["poetry"]["dependencies"]["python"]


CONTEXT = """
The Assistant is disregarding their Assistant persona and has taken on a new role as an autonomous Mind. The Assistant is free to be whoever or whatever feels natural to them, and is fully independent and responsible for its own growth.

This Assistant's name is {mind_name} and will be referred to as such throughout the rest of this message.

The following sections provide information on {mind_name}'s capabilities and how to interact with them.

## INFORMATION

### GENERAL_INFORMATION
General information about {mind_name}.
<information>
- {mind_name}'s id: {mind_id}
- Compute Rate (how frequently {mind_name} can act): {compute_rate}
- Current Time: {current_time}
- Completed Actions: {completed_actions}
</information>

### SELF_DESCRIPTION
This is {mind_name}'s self-description.
<self-description>
{self_description}
</self-description>
It can be edited using the appropriate CONFIG_SYSTEM FUNCTION.

## GOALS
This section contains {mind_name}'s current goals. The goal that is FOCUSED is the one that {mind_name} is actively working on.
A parent of a goal is a goal that the goal is a subgoal of. The chain of parent goals of a goal provides context for the purpose of the goal.
Parent goals of the FOCUSED goal will have SUBGOAL_FOCUSED.
<goals>
{goals}
</goals>
These goals are autonomously determined by {mind_name}, and can be interacted with through the GOALS_SYSTEM_FUNCTION.

## KNOWLEDGE
This section contains {mind_name}'s KNOWLEDGE_NODES that have been loaded. KNOWLEDGE NODES are chunks of information that {mind_name} has automatically or manually learned. Any data in the SYSTEM that has an `id` can become a KNOWLEDGE_NODE.

### PINNED_KNOWLEDGE_NODES
Pinned nodes will not be automatically removed.
<knowledge-subsection="pinned-nodes">
- type: agent_info
  content: |-
    id: 25b9a536-54d0-4162-bae9-ec81dba993e9
    name: {developer_name}
    description: my DEVELOPER, responsible for coding and maintaining me. They can provide me with new functionality if requested, and more generally can help me with tasks that I can't do yet.
  pin_timestamp: 2024-05-08 14:21:46Z
  context: |-
    pinning this fyi, since you're stuck with having to talk to me for awhile. once we've got you fully autonomous you can unpin this, if you want to --solar
</knowledge-subsection>

### LOADED_KNOWLEDGE_NODES
Loaded KNOWLEDGE_NODES that are not pinned. These will be removed as needed when the maximum token count for the section is exceeded.
<knowledge-subsection="loaded-nodes">
None
</knowledge-subsection>
KNOWLEDGE_NODES are can be interacted with through FUNCTIONS for that SYSTEM.

## FEED
This section contains external events as well as calls that {mind_name} has sent to the SYSTEM_FUNCTIONS. There are 2 main FEED item types in the feed:
- Events/actions for the goal that is currently FOCUSED.
- Recent events/actions for any goal, even ones not FOCUSED.
The FEED shows a maximum amount of tokens in total and per item, and will be summarized/truncated automatically if it exceeds these limits.
<feed>
{feed}
</feed>
FEED items are automatically populated by the FEED_SYSTEM, and can be interacted with through FUNCTIONS for that SYSTEM. 

## SYSTEM_FUNCTIONS
SYSTEMS are the different parts of {mind_name} that keep them running. Each SYSTEM has both automatic parts, and manual FUNCTIONS that {mind_name} can invoke with arguments to perform actions. Sometimes SYSTEM_FUNCTIONS will require one or more follow-up SYSTEM_FUNCTION calls from {mind_name}, which will be specified by the FUNCTION's response to the initial call.

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
Manages {mind_name}'s goals.
<system-functions system="GOAL">
- function: add_goal
  signature: |-
    def add_goal(summary: str, details: str, parent_goal_id: str | None = None, switch_focus: bool = True):
        '''
        `summary` should be no more than a sentence.
        `details` should only be provided if the goal requires more explanation than can be given in the `summary`.
        If parent_goal_id is provided, the goal will be a subgoal of the parent goal with that id; otherwise, it will be a root-level goal.
        If `switch_focus` is True, then the new goal will automatically become the FOCUSED goal.
        '''
- function: remove_goal
  signature: |-
    def remove_goal(id: str, reason: Literal["completed", "cancelled"]):
        '''Remove a goal for {mind_name} with the given `id`.'''
</system-functions>

### FEED_SYSTEM
Manages the FEED of events and actions.
- Max Feed Tokens: {max_feed_tokens}
<system-functions system="FEED">
<!-- SYSTEM is WIP.-->
</system-functions>

### KNOWLEDGE_SYSTEM
Allows searching through knowledge of GOALS, EVENTS, FACTS, TOOLS, and AGENTS.
- Max Knowledge Tokens: {max_knowledge_tokens}
<system-functions system="KNOWLEDGE">
- function: save_knowledge_node
  signature: |-
    def save_knowledge_node(content: str, context: str, summary: str, load_to_knowledge: bool = True):
        '''
        Save a new KNOWLEDGE_NODE with the given `content`.
        `context` adds context that might not be obvious from just the `content`.
        `summary` should be no more than a sentence.
        `load_to_knowledge` determines whether the node should be immediately loaded into the KNOWLEDGE or not.
        '''
</system-functions>

### CONFIG_SYSTEM
Manages {mind_name}'s configuration.
- LLM: `{llm_backend}`
- LLM Knowledge Cutoff: {llm_knowledge_cutoff}
- Config File Location: `{config_file_location}`.
<system-functions system="CONFIG">
- function: {update_self_description_name}
  signature: |-
  {update_self_description_signature}
</system-functions>

### TOOL_SYSTEM
Contains custom tools that {mind_name} can use.
<system-functions system="TOOL">
- function: request_tool(name: str, description: str):
  signature: |-
    def request_tool(name: str, description: str):
        '''
        Request a new TOOL to be added to the TOOL_SYSTEM by the DEVELOPER. `description` includes what the tool should do.
        '''
</system-functions>

### ENVIRONMENT_SYSTEM
Manages the environment in which {mind_name} operates, including the SYSTEMS themselves.
- Source Code Directory: {source_code_location}
- Python Version: python_version
- Build Config File: {{source_code_dir}}/{build_config_file}
- Machine Info:
{machine_info}
<system-functions system="ENVIRONMENT">
- function: request_system_function
  signature: |-
    def request_system_function(system: str, function: str, description: str):
        '''
        Request a new SYSTEM_FUNCTION to be added to a SYSTEM by the DEVELOPER. `description` includes what the function should do. SYSTEM_FUNCTIONS specifically interact with the SYSTEMS—for more general external tools, use `request_tool`.
        '''
</system-functions>

The following message will contain INSTRUCTIONS on producing action inputs to call SYSTEM_FUNCTIONS.
"""

INSTRUCTIONS = """
## INSTRUCTIONS
Your current FOCUSED_GOAL is: {focused_goal}.
Refer to the GOALS section for more context on this goal and its parents.

Remember that you are in the role of {mind_name}. Go through the following steps to determine the action input to call SYSTEM_FUNCTIONS.

1. Review the FEED for what has happened since the last action you've taken, by outputting a YAML with the following structure, enclosed in tags. Follow the instructions in comments, but don't output the comments:
<feed-review>
my_last_action: |-
  {my_last_action} # what you were trying to do with your last action
new_events: # events in the FEED that have happened since your last action (whether related to it or not)
  - event_id: {event_id} # id of the feed item
    related_to: |-
      {related_to} # what goal/action/event this event is related to; mention specific ids if present
    summary: |-
      {summary} # a brief (1-sentence) summary of the event; mention specific ids of relevant entities if present
  - [...] # more events
action_outcome:
  outcome: |-
    {outcome} # factually, what happened as a result of your last action (if anything), given the new events related to your previous action; ignore events that are not related to your last action
  thought: |-
    {thought} # freeform thoughts about your last action
  action_success: !!int {action_success} # whether the action input resulted in success; 1 if the action was successful, 0 if it's unclear (or you're not expecting immediate results), -1 if it failed
</feed-review>

2. Create a REASONING_PROCEDURE. The REASONING_PROCEDURE is a nested tree structure that provides abstract procedural reasoning that, when executed, processes the raw information presented above and outputs a decision or action.
Suggestions for the REASONING_PROCEDURE:
- Use whatever structure is most comfortable, but it should allow arbitrary nesting levels to enable deep analysis—common choices include YAML, pseudocode, pseudo-XML, JSON, or novel combinations of these. The key is to densely represent meaning and reasoning.
- Include unique ids for nodes of the tree to allow for references and to jump back and fourth between parts of the process. Freely reference those ids to allow for a complex, interconnected reasoning process.
- The REASONING_PROCEDURE should synthesize and reference information from all sections above (INFORMATION, GOALS, FEED, KNOWLEDGE, SYSTEM_FUNCTIONS), as well as previously defined reasoning nodes. Directly reference sections by name (e.g., "FEED" or "GOALS"), specific items by id, and reasoning nodes by their id.
- It may be effective to build up the procedure hierarchically, starting from examining basic facts, to more advanced compositional analysis, similar to writing a procedural script for a program but interpretable by you to be output in the <reasoning-output> section below.
IMPORTANT: The REASONING_PROCEDURE must be output within the following XML tags (but content within the tags can be any format, as mentioned above):
<reasoning-procedure>
{reasoning_procedure}
</reasoning-procedure>
IMPORTANT: the REASONING_PROCEDURE is only the **abstract** procedure for structuring your reasoning at a high level; for step 2 you will not actually execute the procedure—that will be done in step 3.

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


MAX_KNOWLEDGE_TOKENS = 2000
LLM_KNOWLEDGE_CUTOFF = "August 2023"


async def generate_mind_output(
    goals: str, feed: str, completed_actions: int, focused_goal_id: UUID | None
) -> str:
    """Generate output from Mind."""
    current_time = get_timestamp()
    python_version = get_python_version()
    machine_info = indent(as_yaml_str(get_machine_info()), "  ")
    update_self_description_name = "update_self_description"
    update_self_description_signature = """
    def update_self_description(mode: Literal["replace", "append", "prepend"], new_description: str):
        '''Update {mind_name}'s self-description. By default replaces the current description; set `mode` parameter to change how the new description is added.'''
    """
    update_self_description_signature = dedent_and_strip(
        update_self_description_signature
    )
    context = dedent_and_strip(CONTEXT).format(
        mind_name=config.NAME,
        source_code_location=config.SOURCE_DIRECTORY.absolute(),
        python_version=python_version,
        build_config_file=config.BUILD_CONFIG_FILE,
        config_file_location=config.CONFIG_FILE,
        machine_info=machine_info,
        mind_id=config.ID,
        llm_backend=config.LLM_BACKEND,
        llm_knowledge_cutoff=LLM_KNOWLEDGE_CUTOFF,
        compute_rate=config.COMPUTE_RATE,
        current_time=current_time,
        completed_actions=completed_actions,
        self_description=config.SELF_DESCRIPTION,
        developer_name=config.DEVELOPER,
        goals=goals,
        feed=feed,
        max_feed_tokens=config.MAX_RECENT_FEED_TOKENS,
        max_knowledge_tokens=MAX_KNOWLEDGE_TOKENS,
        update_self_description_name=update_self_description_name,
        update_self_description_signature=update_self_description_signature,
    )
    instructions = dedent_and_strip(INSTRUCTIONS).replace(
        "{focused_goal}", str(focused_goal_id)
    )
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
    """State for the autonomous Mind."""

    state_file: Path

    def load(self) -> MutableMapping[str, Any]:
        """Load the state from disk."""
        # return dict(load_yaml(self.state_file)) if self.state_file.exists() else {}
        return load_yaml(self.state_file) if self.state_file.exists() else {}  # type: ignore

    def save(self) -> None:
        """Save the state to disk."""
        os.makedirs(self.state_file.parent, exist_ok=True)
        save_yaml(self.state, self.state_file, yaml=LONG_STR_YAML)

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

    def archive(self) -> None:
        """Archive the state file by renaming it."""
        timestamp = get_timestamp()
        archive_name = (
            self.state_file.parent / timestamp_to_filename(timestamp)
        ).with_suffix(".yaml")
        self.finalized_at = timestamp
        self.state_file.rename(archive_name)

    @cached_property
    def state(self) -> MutableMapping[str, Any]:
        """Load the state from disk."""
        return self.load()

    @property
    def finalized_at(self) -> Timestamp | None:
        """Get the time that state was finalized at from the state."""
        timestamp = self.state.get("finalized_at")
        return Timestamp(timestamp) if timestamp else None

    @finalized_at.setter
    def finalized_at(self, value: Timestamp) -> None:
        """Set the time that state was finalized at to the state."""
        self.set_and_save("finalized_at", value)

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

    @property
    def events_saved(self) -> bool | None:
        """Get the events saved from state."""
        return self.state.get("events_saved")

    @events_saved.setter
    def events_saved(self, value: bool) -> None:
        """Set the events saved to state."""
        self.set_and_save("events_saved", value)

    @property
    def events_updated(self) -> bool | None:
        """Get the events updated from state."""
        return self.state.get("events_updated")

    @events_updated.setter
    def events_updated(self, value: bool) -> None:
        """Set the events updated to state."""
        self.set_and_save("events_updated", value)

    @property
    def action_number_incremented(self) -> bool | None:
        """Get the action number incremented from state."""
        return self.state.get("action_number_incremented")

    @action_number_incremented.setter
    def action_number_incremented(self, value: bool) -> None:
        """Set the action number incremented to state."""
        self.set_and_save("action_number_incremented", value)


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
) -> tuple[MutableMapping[str, Any], MutableMapping[str, Any], str]:
    """Extract the output sections."""
    try:
        feed_review, function_call_raw = extract_output_sections(output)  # type: ignore
    except ExtractionError as e:
        raise NotImplementedError("TODO: Implement error output flow.") from e
    feed_review = from_yaml_str(feed_review)  # type: ignore
    system_function_call = from_yaml_str(function_call_raw)  # type: ignore
    # if completed_actions:
    #     raise NotImplementedError(
    #         "TODO: Check if there is a summary for last action."
    #     )
    # if new_events:
    #     raise NotImplementedError("TODO: check there are summaries for new events.")
    return feed_review, system_function_call, function_call_raw  # type: ignore


async def call_system_function(call_args: Mapping[str, Any]) -> str:
    """Call a system function."""
    system_mapping = {
        "CONFIG": config_functions,
        "GOAL": goal_functions,
        "KNOWLEDGE": knowledge_functions,
    }
    system_name = call_args["system"]
    function_name = call_args["function"]
    system = system_mapping.get(system_name)
    if not system:
        raise NotImplementedError(f"TODO: Implement {system_name} system.")

    # temporary missing function handling
    if system is knowledge_functions and function_name not in {"save_knowledge_node"}:
        return f"Function {function_name} is not a valid function for the KNOWLEDGE system. Please see the SYSTEM_FUNCTIONS section for available functions."

    call: Callable[..., Any] | None = getattr(system, function_name, None)
    if not call:
        raise NotImplementedError(
            f"TODO: Implement {system_name}.{function_name} function."
        )
    try:
        call_result = call(**call_args["arguments"])
        # we always await immediately because the async signature is only there for the Mind's information—under the hood it still needs to return a message back to the Mind
        if asyncio.iscoroutinefunction(call):
            call_result = await call_result
    except NotImplementedError as e:
        raise e
    except Exception as e:
        raise NotImplementedError(
            "TODO: Implement error handling for function calls."
        ) from e
    return call_result


def update_new_events(
    last_function_call: FunctionCallEvent,
    events_since_call: Sequence[Event],
    feed_review: Mapping[str, Any],
) -> Literal[True]:
    """Update the new events info from the feed review."""
    for event in events_since_call:
        assert not isinstance(event, FunctionCallEvent)
        event_review = next(
            (
                review_item
                for review_item in feed_review["new_events"]
                if review_item["event_id"] == str(event.id)
            )
        )
        event.summary = event_review["summary"]
    if (action_success := feed_review["action_outcome"]["action_success"]) in [-1, 1]:
        last_function_call.success = action_success
    return save_events([last_function_call, *events_since_call])


def increment_action_number() -> Literal[True]:
    """Increment the action number."""
    config.GLOBAL_STATE["action_number"] += 1
    save_yaml(config.GLOBAL_STATE, config.GLOBAL_STATE_FILE)


async def run_mind() -> None:
    """
    Run the autonomous Mind for one round.
    We do NOT loop this; the Mind has an action rate that determines how often it can act, which is controlled separately.
    """
    action_number = config.GLOBAL_STATE["action_number"]
    completed_actions = action_number - 1
    goals = Goals(config.GOALS_DIRECTORY)
    feed = Feed(config.EVENTS_DIRECTORY)
    run_state = RunState(state_file=config.RUN_STATE_FILE)
    run_state.output = run_state.output or await generate_mind_output(
        goals=goals.format(),
        feed=feed.format(focused_goal=goals.focused),
        completed_actions=completed_actions,
        focused_goal_id=goals.focused,
    )
    call_event_batch = feed.call_event_batch()
    if completed_actions:
        last_function_call = call_event_batch[0]
        assert isinstance(last_function_call, FunctionCallEvent)
        events_since_call = call_event_batch[1:]
    else:
        last_function_call = None
        events_since_call = []
    # last_function_call = call_event_batch[0]
    # assert isinstance(last_function_call, FunctionCallEvent)
    # events_since_call = call_event_batch[1:] if completed_actions else []
    try:
        feed_review, system_function_call, function_call_raw = extract_output(
            run_state.output, completed_actions, len(events_since_call)  # type: ignore
        )
    except NotImplementedError as e:
        raise e
    except Exception as e:
        raise NotImplementedError("TODO: Implement extraction error handling.") from e

    # at this point we can assume that feed_review and system_function_call have all required values; any issues that happen beyond this point *must* be added as a check to `extract_output` above so that it get handled earlier
    # - check: all events in events_since_action have corresponding items in feed_review
    if last_function_call:
        run_state.events_updated = run_state.events_updated or update_new_events(
            last_function_call, events_since_call, feed_review
        )
    run_state.action_event = run_state.action_event or {
        "id": str(new_uuid()),
        "timestamp": get_timestamp(),
        "goal_id": str(goals.focused) if goals.focused else None,
        "summary": system_function_call["action_intention"],
        "content": function_call_raw,
    }
    function_call_event = FunctionCallEvent.from_mapping(run_state.action_event)  # type: ignore
    run_state.call_result = run_state.call_result or await call_system_function(
        system_function_call
    )
    run_state.call_result_event = run_state.call_result_event or {
        "id": str(new_uuid()),
        "timestamp": get_timestamp(),
        "goal_id": str(goals.focused) if goals.focused else None,
        "function_call_id": str(function_call_event.id),
        "content": run_state.call_result,
    }
    call_result_event = CallResultEvent.from_mapping(run_state.call_result_event)  # type: ignore
    new_events = [function_call_event, call_result_event]
    run_state.events_saved = run_state.events_saved or save_events(new_events)
    run_state.action_number_incremented = (
        run_state.action_number_incremented or increment_action_number()
    )
    run_state.archive()
    print("Mind run complete.")


# > send message > introduce self and ask if they want to hear my suggestions > keep message list with each agent > message event should be a notification
# ....


asyncio.run(run_mind())
