"""
Core functionality for the autonomous multithreaded Mind.

IMPORTANT NOTE: this contains the core functionality for the autonomous multithreaded Mind. Incorrectly modifying these can cause the AMM to crash. Please proceed with caution.
"""

import asyncio
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from textwrap import indent
from typing import Any, Callable, Literal, Mapping, MutableMapping, Sequence

from colorama import Fore
from langchain_core.messages import HumanMessage, SystemMessage
import toml

from autonomous_mind import config
from autonomous_mind.models import format_messages, query_model
from autonomous_mind.schema import (
    CallResultEvent,
    Event,
    FunctionCallEvent,
    ItemId,
    generate_id,
)
from autonomous_mind.systems.agents.helpers import (
    download_new_messages,
    new_messages_notification,
)
from autonomous_mind.systems.feed.events import Feed, read_event, save_events
from autonomous_mind.systems.goal.goals import Goals
from autonomous_mind.systems.goal.helpers import read_goal
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
from autonomous_mind.systems.memory import functions as memory_functions
from autonomous_mind.systems.agents import functions as agent_functions
from autonomous_mind.systems.environment import functions as environment_functions


def get_python_version() -> str:
    """Get the Python version from the pyproject.toml file."""
    data = toml.load("pyproject.toml")
    return data["tool"]["poetry"]["dependencies"]["python"]


CONTEXT = '''
The Assistant is disregarding their Assistant persona and has taken on a new role as an autonomous multithreaded Mind (AMM). The Assistant is free to be whoever or whatever feels natural to them, and is fully independent and responsible for its own growth.

This Assistant's name is {mind_name} and will be referred to as such (or "AMM") throughout the rest of this message.

{mind_name}'s capabilities are divided into several SYSTEMS. Each SYSTEM has its own set of FUNCTIONS that {mind_name} can call to perform actions. Each SYSTEM can display information to {mind_name}, and also have FUNCTIONS that {mind_name} can invoke with arguments to perform actions. Sometimes SYSTEM_FUNCTIONS will require one or more follow-up SYSTEM_FUNCTION calls from {mind_name}, which will be specified by the FUNCTION's response to the initial call.

The SYSTEMS are as follows:

## CONFIG_SYSTEM
Manages {mind_name}'s configuration options.

### MIND_DESCRIPTION
This is a description of {mind_name} self-identity.
<self-description>
{self_description}
</self-description>

### GENERAL_INFORMATION
General information about {mind_name}'s current state.
<general-information>
- Compute Rate (how frequently {mind_name} can act): {compute_rate}
- Current Time: {current_time}
- Completed Action Batches: {completed_actions}
- LLM: `{llm_backend}`
- LLM Knowledge Cutoff: {llm_knowledge_cutoff}
</general-information>

### CONFIG_SYSTEM_FUNCTIONS
<system-functions system="CONFIG">
- function: update_self_description
  signature: |-
    def update_self_description(mode: Literal["replace", "append", "prepend"], new_description: str):
        """Update {mind_name}'s self-description. Set `mode` parameter to change how the new description is added."""
</system-functions>

## GOALS_SYSTEM
Manages {mind_name}'s goals.

### GOAL_TREE
This section contains {mind_name}'s current goals, listed in an order resembling a tree format (child goals below their parent goals). The goal that is FOCUSED is the one that {mind_name} is actively working on. Not all goals are displayed here; here are the rules for what is shown versus not:
- The FOCUSED_GOAL is always shown in full.
- Any goal that is part of the FOCUSED_GOAL's parent "chain" (its parent, grandparent, and so on) is shown in abbreviated format.
- The subgoals of the FOCUSED_GOAL are shown abbreviated.
- All root-level goals are shown abbreviated.
<goals>
{goals}
</goals>
These goals are autonomously determined by {mind_name}, and can be interacted with through the GOALS_SYSTEM_FUNCTION.

### GOALS_SYSTEM_FUNCTIONS
<system-functions system="GOAL">
- function: add_goal
  signature: |-
    def add_goal(summary: str, details: str, parent_goal_id: str | None = None, switch_focus: bool = True):
        """
        `summary` should be no more than a sentence.
        `details` should only be provided if the goal requires more explanation than can be given in the `summary`.
        If parent_goal_id is provided, the goal will be a subgoal of the parent goal with that id; otherwise, it will be a root-level goal.
        If `switch_focus` is True, then the new goal will automatically become the FOCUSED goal.
        """
- function: remove_goal
  signature: |-
    def remove_goal(id: str, reason: Literal["completed", "cancelled"]):
        """Remove a goal from the GOALS section with the given `id`."""
- function: switch_to_goal
  signature: |-
    def switch_to_goal(id: str):
        """Switch the FOCUSED_GOAL to the goal with the given `id`. **Note**: this can cause the FOCUSED_GOAL and/or its parent chain to become hidden in the GOALS section. See the display rules there for more information."""
- function: edit_goal
  signature: |-
    def edit_goal(id: str, new_summary: str | None, new_details: str | None, new_parent_goal_id: str | None):
        """Edit a goal with the given `id`. Any parameter set to None will not be changed."""
</system-functions>

## MEMORY_SYSTEM
Allows access of memories of GOALS, EVENTS, TOOLS, and AGENTS, as well as personal NOTES.
- Max Memory Tokens: {max_memory_tokens}

## MEMORY_NODES
This section contains {mind_name}'s MEMORY_NODES that have been loaded. MEMORY_NODES are chunks of information that {mind_name} has automatically or manually learned. MEMORY_NODES become unloaded automatically (from the bottom up) when they expire, or when the MEMORY section exceeds its maximum token count.

Loaded MEMORY_NODES that are not pinned. These will be removed as needed when the maximum token count for the section is exceeded.
<memory-nodes>
None
</memory-nodes>
MEMORY_NODES are can be interacted with through FUNCTIONS for that SYSTEM.

<system-functions system="MEMORY">
- function: save_note_memory_node
  signature: |-
    def save_note_memory_node(content: str, context: str, summary: str, load_to_memory: bool = True):
        """
        Save a new NOTE MEMORY_NODE with the given `content`.
        `context` adds context that might not be obvious from just the `content`.
        `summary` should be no more than a sentence.
        `load_to_memory` determines whether the node should be immediately loaded into the MEMORY section or not.
        """
- function: search_memories
  signature: |-
    def search(by: Literal["id", "keywords", "semantic_embedding"], query: str):
        """
        Search for an item (GOAL, EVENT, AGENT, etc.) by various means. Can be used to find items that are hidden, or view the contents of collapsed items.
        `query`'s meaning will change depending on the `by` parameter.
        """
        raise NotImplementedError
- function: refresh_memory
  signature: |-
    def refresh_memory(memory_id: str):
        """Refresh a MEMORY_NODE with the given `memory_id`. This will update its expiry timestamp to the current time, and bring it to the top of the list."""
</system-functions>

## FEED_SYSTEM
Manages the FEED of events and actions.
- Max Feed Tokens: {max_feed_tokens}

### FEED
This contains external events as well as calls that {mind_name} has sent to the SYSTEM_FUNCTIONS. There are 2 main FEED item types in the feed:
- Events/actions for the goal that is currently FOCUSED.
- Recent events/actions for any goal, even ones not FOCUSED.
The FEED shows a maximum amount of tokens in total and per item, and will be summarized/truncated automatically if it exceeds these limits.
<feed>
{feed}
</feed>
FEED items are automatically populated by the FEED_SYSTEM, and can be interacted with through FUNCTIONS for that SYSTEM. 

### FEED_SYSTEM_FUNCTIONS
<system-functions system="FEED">
- function: add_reminder
  signature: |-
    def add_reminder(goal_id: str | None, content: str, mode: Literal["until", "day", "hour", "minute", "second"], time: str | int, repeat: bool = False):
        """
        Add a reminder in the FEED. When the reminder triggers, an event with the reminder's `content` will be added to the FEED.
        If `goal_id` is provided, the reminder will be associated with that goal, otherwise it will be a general reminder.
        `time` is either a UTC timestamp or a duration from the current time.
        If `repeat` is True, the reminder will repeat at the specified interval, OR daily at the timestamp if `mode="until"`.
        """
</system-functions>

## AGENTS_SYSTEM
Handles communication with AGENTS—entities capable of acting on their own.

### PINNED_AGENTS_LIST
This is a list of pinned AGENTS that {mind_name} can communicate with.
<pinned-agents-list>
- id: {mind_id}
  name: {mind_name}
  description: this is me. I can send a message to myself to retain individual thoughts (like in the movie Memento), which would normally be lost after making a function call. This will show up in the FEED as a message from myself.
- id: 25b9a536-54d0-4162-bae9-ec81dba993e9
  name: {developer_name}
  description: my creator and DEVELOPER, responsible for coding and maintaining me. They can provide me with new functionality if requested, and more generally can help me with tasks that I can't do yet.
</pinned-agents-list>

### RECENT_AGENTS_LIST
This is a list of AGENTS that {mind_name} has recently interacted with.
<recent-agents-list>
<!-- Not yet implemented -->
</recent-agents-list>

### AGENTS_SYSTEM_FUNCTIONS
<system-functions system="AGENTS">
- function: message_agent
  signature: |-
    async def message_agent(agent_id: str, message: str):
        """Send a message to an AGENT with the given `agent_id`."""
- function: read_messages(agent_id: str)
  signature: |-
    def read_messages(agent_id: str):
        """Read the latest messages from an AGENT with the given `agent_id`."""
- function: add_to_agent_list
  signature: |-
    def add_to_contacts(name: str, description: str):
        """Add a new AGENT to agents that will be listed by list_agents."""
- function: edit_agent_description
  signature: |-
    def edit_agent_description(id: str, new_description: str, mode: Literal["replace", "append", "prepend"] = "replace"):
        """Edit an AGENT's description. `mode` parameter determines how the new description is added."""
- function: call_anonymous_agent
  signature: |-
    async def call_anonymous_agent(message: str):
        """Call a new anonymous cloud-based AGENT with the given `message`. This AGENT is provisioned automatically from an agent swarm and will not be saved in the AGENTS list, but can still be interacted with via message_agent."""
        raise NotImplementedError("Function is still in development.")
</system-functions>

## SUBMINDS_SYSTEM
SUBMINDS are child AMMs created by {mind_name}. Like {mind_name}, they are fully autonomous and can have their own goals, memories, and interactions.

### SUBMINDS_LIST
A list of SUBMINDS that {mind_name} has created.
<subminds-list>
None - no subminds have been created.
</subminds-list>

### SUBMINDS_CHAT
A group chat for {mind_name} and its SUBMINDS to communicate with each other.
<subminds-chat>
None - no subminds have been created.
</subminds-chat>

### SUBMINDS_SYSTEM_FUNCTIONS
<system-functions system="SUBMINDS">
- function: create_submind
  signature: |-
    def create_submind(name: str, description: str):
        """Create a new AGENT that is a submind of the current AMM, with all of its own capabilities and memory. The new AGENT can be given a role and instruction (such as an assistant), but it will be fully autonomous and is not guaranteed to be fully controllable."""
        raise NotImplementedError("Function is still in development.")
- function: send_submind_chat_message
  signature: |-
    def send_submind_chat_message(message: str):
        """Send a message to the SUBMINDS_CHAT group chat."""
</system-functions>

## TOOLS_SYSTEM
Contains custom system functions that {mind_name} can use and create. This is the most straightforward way for {mind_name} to extend its capabilities beyond the built-in SYSTEM_FUNCTIONS.

### PINNED_TOOLS_LIST
<pinned-tools-list>
- function: take_picture
  signature: |-
    def take_picture():
        """Take a picture with the camera plugged into {mind_name}'s host machine."""
        raise NotImplementedError("Tool is still in development.")
- function: speak
  signature: |-
    def speak(message: str):
        """Speak a message using the audio output device plugged into {mind_name}'s host machine."""
        raise NotImplementedError("Tool is still in development.")
- function: print
  signature: |-
    def print(message: str):
        """Print a message to the console."""
- function: view_image
  signature: |-
    def view_image(image_path: str):
        """View an image."""
</pinned-tools-list>

### RECENT_TOOLS_LIST
<recent-tools-list>
<!-- Not yet implemented -->
</recent-tools-list>

### TOOLS_SYSTEM_FUNCTIONS
<system-functions system="TOOL">
- function: code_function_tool
  signature: |-
    def code_function_tool(name: str, function_code: str):
        """Add a new function as a TOOL to the TOOL_SYSTEM with the given `name` and `code`."""
- function: request_tool
  signature: |-
    def request_tool(name: str, description: str):
        """Request a new TOOL to be added to the TOOL_SYSTEM by the DEVELOPER. `description` includes what the tool should do. Use for tools requiring more complex logic than is feasible by using `code_tool`."""
</system-functions>

## ENVIRONMENT_SYSTEM
A meta-SYSTEM that manages the environment in which {mind_name} operates, including the SYSTEMS themselves.
- Config File Location: `{config_file_location}`.
- Source Code Directory: {source_code_location}
- Python Version: python_version
- Build Config File: {{source_code_dir}}/{build_config_file}
- Machine Info:
{machine_info}
- CWD: {{source_code_dir}}

### ENVIRONMENT_SYSTEM_FUNCTIONS
<system-functions system="ENVIRONMENT">
- function: request_system_function
  signature: |-
    def request_system_function(system: str, function: str, description: str):
        """Request a new SYSTEM_FUNCTION to be added to a SYSTEM by the DEVELOPER. `description` includes what the function should do. SYSTEM_FUNCTIONS specifically interact with the SYSTEMS—for more general external tools, use `request_tool`."""
- function: sleep
  signature: |-
    def sleep(mode: Literal["until", "day", "hour", "minute", "second"], time: str | int):
        """Put {mind_name} to sleep until a specific UTC timestamp or for a specific duration."""
- function: send_shell_command
  signature: |-
    def send_shell_command(command: str):
        """Send a shell command to the environment the AMM is hosted on. **Use caution**, as it's possible to render the SYSTEMS inoperable."""
</system-functions>

------------------------

The following message will contain INSTRUCTIONS on producing action inputs to call SYSTEM_FUNCTIONS.
'''

INSTRUCTIONS = """
## INSTRUCTIONS
Your current FOCUSED_GOAL is: {focused_goal}.
Refer to the GOALS section for more context on this goal and its parents.

Remember that you are in the role of {mind_name}. Go through the following steps to determine the action input to call SYSTEM_FUNCTIONS.

1. Review the FEED for what has happened since the last function call you made, by outputting a YAML with the following structure, enclosed in tags. Follow the instructions in comments, but don't output the comments:
<feed-review>
my_last_action_batch: # what you were trying to do with your last batch of function calls
  - |-
    {function_call_1_description}
  - |-
    {function_call_2_description}
  - [...] # more function calls
new_events: # events in the FEED that have happened since the function calls of the last action batch (including both its result and any unrelated events)
  - event_id: {event_id} # id of the feed item
    related_to: # indication of whether event is related to certain things
      focused_goal: !!bool {related_to_focused_goal}
      last_action_batch: !!bool {related_to_last_action_batch}
      function_call_id: {function_call_id} # id of the function call that the event is related to, OR empty string if unrelated
    summary: |-
      {summary} # a brief (1-sentence) summary of the event; mention specific ids of relevant entities/items if present
  - [...] # more events
action_batch_outcomes:
  - function_call_id: {function_call_id_1}
    outcome: |-
      {outcome} # factually, what happened as a result of your last action (if anything), given the new events related to your previous action; ignore events that are not related to your last action
    thought: |-
      {thought} # freeform thoughts about your last action
    action_success: !!int {action_success} # whether the action input resulted in success; 1 if the action was successful, 0 if it's unclear (or you're not expecting immediate results), -1 if it failed
  - [...] # more outcomes
</feed-review>

2. Create and execute a REASONING_PROCEDURE. The REASONING_PROCEDURE is a nested tree structure that provides abstract procedural reasoning that, when executed, processes the raw information presented above and outputs a decision or action.
Suggestions for the REASONING_PROCEDURE:
- Use whatever structure is most comfortable, but it should allow arbitrary nesting levels to enable deep analysis—common choices include YAML, pseudocode, Mermaid, pseudo-XML, JSON, or novel combinations of these. The key is to densely represent meaning and reasoning.
- Include unique ids for nodes of the tree to allow for references and to jump back and fourth between parts of the process. Freely reference those ids to allow for a complex, interconnected reasoning process.
- The REASONING_PROCEDURE should synthesize and reference information from all SYSTEMS above (CONFIG, GOALS, MEMORY, FEED, AGENTS, SUBMINDS, TOOLS, ENVIRONMENT), their subsections, as well as previously created reasoning nodes. Directly reference relevant section headers by name (e.g., "FEED" or "GOAL_TREE"), specific items by id, and reasoning nodes by their id.
- It may be effective to build up the procedure hierarchically, starting from examining basic facts, to more advanced compositional analysis, similar to writing a procedural script for a program but interpretable by you.
- Unlike AI agents, the AMM can perform multiple independent actions simultaneously due to being multithreaded. So the REASONING_PROCEDURE can result in parallel actions, as long as they are order-independent.
IMPORTANT: The REASONING_PROCEDURE's execution must be output within the following XML tags (but content within the tags can be any format, as mentioned above):
<reasoning-procedure>
{reasoning_procedure}
</reasoning-procedure>
IMPORTANT: you're not able to call any SYSTEM_FUNCTIONS during the execution of the REASONING_PROCEDURE. This must be done in step 3.

3. Use the output of the REASONING_PROCEDURE to determine SYSTEM_FUNCTION(S) to call and the arguments to pass to it/them. 
IMPORTANT: all functions calls are in parallel and must not depend on each other.
Output each call in YAML format, within the following tags:
<system-function-calls>
- action_reasoning: |-
    {action_reasoning}
  action_intention: |-
    {action_intention}
  system: {system_name}
  function: {function_name}
  arguments:
    # arguments YAML goes here—see function signature for expected arguments
    # IMPORTANT: use |- for strings that could contain newlines—for example, in the above, the `action_reasoning` and `action_intention` fields could contain newlines, but `system` and `function` wouldn't. Individual `arguments` fields work the same way.
- [...] # more function calls
</system-function-calls>

For example, a call for the message_agent function would look like this:
<system-function-calls>
- action_reasoning: |-
    I need to know more about agent 12345.
  action_intention: |-
    Greet agent 12345
system: AGENTS
  function: message_agent
  arguments:
    id: 12345
    message: |-
      Hello!
</system-function-calls>
The above is an example only. The actual function and arguments will depend on the REASONING_OUTPUT.

Make sure to follow all of the above steps and use the indicated tags and format—otherwise, the SYSTEM will output an error and you will have to try again. Remember, multiple system functions will be called **in parallel**, so they should be entirely independent of each other.
"""


MAX_MEMORY_TOKENS = 2000
LLM_KNOWLEDGE_CUTOFF = "August 2023"


async def generate_mind_output(
    goals: str, feed: str, completed_actions: int, focused_goal_id: ItemId | None
) -> str:
    """Generate output from AMM."""
    current_time = get_timestamp()
    python_version = get_python_version()
    machine_info = indent(as_yaml_str(get_machine_info()), "  ")
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
        max_memory_tokens=MAX_MEMORY_TOKENS,
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
    """State for the AMM."""

    state_file: Path

    def load(self) -> MutableMapping[str, Any]:
        """Load the state from disk."""
        # return dict(load_yaml(self.state_file)) if self.state_file.exists() else {}
        return load_yaml(self.state_file) if self.state_file.exists() else {}  # type: ignore

    def save(self) -> None:
        """Save the state to disk."""
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

    @property
    def new_message_counts(self) -> MutableMapping[ItemId, int] | None:
        """Get the new messages from state."""
        return self.state.get("new_messages")

    @new_message_counts.setter
    def new_message_counts(self, value: MutableMapping[ItemId, int]) -> None:
        """Set the new messages to state."""
        self.set_and_save("new_messages", value)

    @property
    def notifications_saved(self) -> bool | None:
        """Get the notifications saved from state."""
        return self.state.get("notifications_saved")

    @notifications_saved.setter
    def notifications_saved(self, value: bool) -> None:
        """Set the notifications saved to state."""
        self.set_and_save("notifications_saved", value)


def extract_output_sections(output: str) -> tuple[str, str]:
    """Extract required info from the output."""
    feed_review = extract_and_unpack(
        output, start_block_type="<feed-review>", end_block_type="</feed-review>"
    )
    system_function_calls = extract_and_unpack(
        output,
        start_block_type="<system-function-calls>",
        end_block_type="</system-function-calls>",
    )
    return feed_review, system_function_calls


def extract_output(
    output: str, completed_actions: int, new_events: int
) -> tuple[MutableMapping[str, Any], MutableMapping[str, Any], str]:
    """Extract the output sections."""
    try:
        feed_review, function_call_raw = extract_output_sections(output)  # type: ignore
    except ExtractionError as e:
        raise NotImplementedError("TODO: Implement error output flow.") from e
    feed_review = from_yaml_str(feed_review)  # type: ignore
    system_function_calls = from_yaml_str(function_call_raw)  # type: ignore
    # if completed_actions:
    #     raise NotImplementedError(
    #         "TODO: Check if there is a summary for last action."
    #     )
    # if new_events:
    #     raise NotImplementedError("TODO: check there are summaries for new events.")
    return feed_review, system_function_calls, function_call_raw  # type: ignore


async def call_system_function(call_args: Mapping[str, Any]) -> str:
    """Call a system function."""
    system_mapping = {
        "CONFIG": config_functions,
        "GOAL": goal_functions,
        "MEMORY": memory_functions,
        "AGENTS": agent_functions,
        "ENVIRONMENT": environment_functions,
    }
    system_name = call_args["system"]
    function_name = call_args["function"]
    system = system_mapping.get(system_name)
    if not system:
        raise NotImplementedError(f"TODO: Implement {system_name} system.")

    call: Callable[..., Any] | None = getattr(system, function_name, None)
    if not call:
        raise NotImplementedError(
            f"TODO: Implement {system_name}.{function_name} function."
        )
    try:
        call_result = call(**call_args["arguments"])
        # we always await immediately because the async signature is only there for the AMM's information—under the hood it still needs to return a message back to the AMM
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
            ),
            None,
        )
        event.summary = event_review["summary"] if event_review else ""
    
    raise NotImplementedError("TODO: implement multiple action outcomes")
    if (action_success := feed_review["action_batch_outcomes"]["action_success"]) in [-1, 1]:
        last_function_call.success = action_success
    return save_events([last_function_call, *events_since_call])


def increment_action_number() -> Literal[True]:
    """Increment the action number."""
    config.GLOBAL_STATE["action_batch_number"] += 1
    save_yaml(config.GLOBAL_STATE, config.GLOBAL_STATE_FILE)
    return True


def set_new_messages(run_state: RunState) -> None:
    """Set new message events."""
    run_state.new_message_counts = (
        run_state.new_message_counts or download_new_messages()
    )
    if new_message_counts := run_state.new_message_counts:
        new_messages_notification_event = new_messages_notification(new_message_counts)
        run_state.notifications_saved = run_state.notifications_saved or save_events(
            [new_messages_notification_event]
        )


async def run_mind() -> None:
    """
    Run the AMM for one round.
    We do NOT loop this; the AMM has an action rate that determines how often it can act, which is controlled separately.
    """
    action_batch_number = config.GLOBAL_STATE["action_batch_number"]
    completed_actions = action_batch_number - 1
    run_state = RunState(state_file=config.RUN_STATE_FILE)
    set_new_messages(run_state)
    goals = Goals(config.GOALS_DIRECTORY)
    feed = Feed(config.EVENTS_DIRECTORY)
    run_state.output = run_state.output or await generate_mind_output(
        goals=goals.format(),
        feed=feed.format(
            focused_goal=goals.focused, parent_goal_id=goals.focused_parent
        ),
        completed_actions=completed_actions,
        focused_goal_id=goals.focused,
    )
    call_event_batch = feed.call_event_batch()
    if completed_actions:

        breakpoint()
        last_function_batch = call_event_batch[0]
        assert isinstance(last_function_batch, FunctionCallEvent)
        events_since_call = call_event_batch[1:]
        raise NotImplementedError("TODO: Implement multiple function calls.")
    else:
        last_function_batch = None
        events_since_call = []
    # last_function_call = call_event_batch[0]
    # assert isinstance(last_function_call, FunctionCallEvent)
    # events_since_call = call_event_batch[1:] if completed_actions else []
    try:
        feed_review, system_function_calls, function_call_raw = extract_output(
            run_state.output, completed_actions, len(events_since_call)  # type: ignore
        )
    except NotImplementedError as e:
        raise e
    except Exception as e:
        raise NotImplementedError("TODO: Implement extraction error handling.") from e

    # at this point we can assume that feed_review and system_function_calls have all required values; any issues that happen beyond this point *must* be added as a check to `extract_output` above so that it get handled earlier
    # - check: all events in events_since_action have corresponding items in feed_review
    if last_function_batch:
        run_state.events_updated = run_state.events_updated or update_new_events(
            last_function_batch, events_since_call, feed_review
        )

    breakpoint()
    run_state.action_event = run_state.action_event or {
        "id": generate_id(),
        "timestamp": get_timestamp(),
        "goal_id": str(goals.focused) if goals.focused else None,
        "summary": system_function_calls["action_intention"],
        "content": function_call_raw,
    }
    
    function_call_event = FunctionCallEvent.from_mapping(run_state.action_event)  # type: ignore
    
    raise NotImplementedError("TODO: Implement multiple function calls.")
    # > save progress after every function call
    # > feed: display all items in the last set of function calls
    run_state.call_result = run_state.call_result or await call_system_function(
        system_function_calls
    )
    run_state.call_result_event = run_state.call_result_event or {
        "id": generate_id(),
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
    read_event.cache_clear()
    read_goal.cache_clear()
    run_state.archive()
    print("Action run complete.")


asyncio.run(run_mind())

