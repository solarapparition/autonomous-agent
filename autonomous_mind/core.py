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

from autonomous_mind.systems.config import settings
from autonomous_mind.id_generation import generate_id
from autonomous_mind.models import format_messages, query_model
from autonomous_mind.schema import (
    CallResultEvent,
    Event,
    FunctionCallEvent,
    ItemId,
)
from autonomous_mind.systems.agents.helpers import (
    download_new_messages,
    mark_messages_read,
    new_messages_notification,
)
from autonomous_mind.systems.config.global_state import global_state
from autonomous_mind.systems.feed.events import Feed
from autonomous_mind.systems.goals.goals import Goals
from autonomous_mind.systems.helpers import read_event, read_goal, save_events
from autonomous_mind.systems.memory.memory import load_active_memories
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
from autonomous_mind.systems.goals import functions as goals_functions
from autonomous_mind.systems.memory import functions as memory_functions
from autonomous_mind.systems.agents import functions as agent_functions
from autonomous_mind.systems.environment import (
    functions as environment_functions,
    shell,
)
from autonomous_mind.systems.agents.helpers import read_agent_conversation

AGENT_COLOR = Fore.GREEN
PROMPT_COLOR = Fore.BLUE


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
- LLM: `{llm_backend}`
- LLM Knowledge Cutoff: {llm_knowledge_cutoff}
</general-information>

### CONFIG_SYSTEM_FUNCTIONS
<system-functions system="CONFIG">
- function: modify_self_description
  signature: |-
    def modify_self_description(key_path: tuple[str | int], mode: Literal["add_or_update", "delete"], value: JSONSerializable | "TBD" | None):
        """
        Modify {mind_name}'s self-description with additional, **permanent** information about her/his/their identity, to help them remain consistent over long time horizons.
        - the self-description is a YAML structure that can be modified by adding, updating, or deleting keys.
        - `key_path` is used to traverse the YAML structure to the desired key.
        """
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
These goals are autonomously determined by {mind_name}, and can be interacted with through the GOALS_SYSTEM_FUNCTIONS.

### GOALS_SYSTEM_FUNCTIONS
<system-functions system="GOALS">
- function: add_goal
  signature: |-
    async def add_goal(summary: str, details: str, parent_goal_id: int | None = None, switch_focus: bool = True):
        """
        `summary` should be no more than a sentence.
        `details` should only be provided if the goal requires more explanation than can be given in the `summary`.
        If parent_goal_id is provided, the goal will be a subgoal of the parent goal with that id; otherwise, it will be a root-level goal.
        If `switch_focus` is True, then the new goal will automatically become the FOCUSED goal.
        """
- function: remove_goal
  signature: |-
    async def remove_goal(id: str, reason: Literal["completed", "cancelled"]):
        """Remove a goal from the GOALS section with the given `id`."""
- function: switch_to_goal
  signature: |-
    async def switch_to_goal(id: str):
        """Switch the FOCUSED_GOAL to the goal with the given `id`. **Note**: this can cause the FOCUSED_GOAL and/or its parent chain to become hidden in the GOALS section. See the display rules there for more information."""
</system-functions>

## MEMORY_SYSTEM
Allows access of memories of GOALS, EVENTS, TOOLS, and AGENTS, as well as personal NOTES.
- Max Memory Tokens: {max_memory_tokens}

## MEMORY_NODES
This section contains {mind_name}'s MEMORY_NODES that have been loaded. MEMORY_NODES are chunks of information that {mind_name} has automatically or manually learned. MEMORY_NODES become unloaded automatically (from the bottom up) when they expire, or when the MEMORY section exceeds its maximum token count.

<memory-nodes filter="loaded">
{memories}
</memory-nodes>
MEMORY_NODES are can be interacted with through FUNCTIONS for that SYSTEM.

<system-functions system="MEMORY">
- function: create_note
  signature: |-
    async def create_note(content: str, context: str, summary: str, goal_id: int | None = None, load_to_memory: bool = True):
        """
        Create a new NOTE with the given `content`.
        `context` adds context that might not be obvious from just the `content`.
        `summary` should be no more than a sentence.
        `goal_id` is the id of the goal that this note is related to. If None, the note is just a general note.
        `load_to_memory` determines whether the note should be immediately loaded into the MEMORY section or not.
        """
- function: search_memories
  signature: |-
    async def search_memories(by: Literal["id", "keywords", "semantic_embedding"], query: str):
        """
        Search for an item (GOALS, EVENT, AGENT, etc.) by various means. Can be used to find items that are hidden, or view the contents of collapsed items.
        `query`'s meaning will change depending on the `by` parameter.
        """
        raise NotImplementedError
- function: refresh_memory
  signature: |-
    async def refresh_memory(memory_id: int):
        """Refresh a MEMORY_NODE with the given `memory_id`. This will update its expiry timestamp to the current time, and bring it to the top of the list."""
</system-functions>

## FEED_SYSTEM
Manages the FEED of events and actions.
- Max Feed Tokens: {max_feed_tokens}

### FEED
This contains external events as well as calls that {mind_name} has sent to the SYSTEM_FUNCTIONS.
- There are 2 main FEED item types in the feed:
  - Events/actions for the goal that is currently FOCUSED.
  - Recent events/actions for any goal, even ones not FOCUSED.
- The FEED shows a maximum amount of tokens in total and per item, and will be summarized/truncated automatically if it exceeds these limits.
- Events/actions are grouped into batches. Each time {mind_name} makes a set of concurrent function calls, the calls and any events that happen after the calls are grouped into a batch.
- Current Action Batch Number: {action_batch_number}
<feed>
{feed}
</feed>
FEED items are automatically populated by the FEED_SYSTEM, and can be interacted with through FUNCTIONS for that SYSTEM. 

### FEED_SYSTEM_FUNCTIONS
<system-functions system="FEED">
- function: add_reminder
  signature: |-
    async def add_reminder(goal_id: int | None, content: str, mode: Literal["until", "day", "hour", "minute", "second"], time: str | int, repeat: bool = False):
        """
        Add a reminder in the FEED. When the reminder triggers, an event with the reminder's `content` will be added to the FEED.
        If `goal_id` is provided, the reminder will be associated with that goal, otherwise it will be a general reminder.
        `time` is either a UTC timestamp or a duration from the current time.
        If `repeat` is True, the reminder will repeat at the specified interval, OR daily at the timestamp if `mode="until"`.
        """
- function: load_feed_item_into_memory
  signature: |-
    async def load_feed_item_into_memory(feed_item_id: int):
        """Load a feed item into the MEMORY section for medium-term reference."""

## AGENTS_SYSTEM
Handles communication with AGENTS—entities capable of acting on their own.

### OPENED_AGENT_CONVERSATION
An ongoing conversation with an AGENT. {mind_name} can only have one conversation open at a time.

Currently Open Agent Id: {opened_agent_id}
<viewport view="opened-agent-conversation">
{opened_agent_conversation}
</viewport>

### PINNED_AGENTS_LIST
This is a list of pinned AGENTS that {mind_name} can communicate with.
<pinned-agents-list>
- id: {mind_id}
  name: {mind_name}
  description: this is me. I can send a message to myself to retain individual thoughts (like in the movie Memento), which would normally be lost after making a function call. This will show up in the FEED as a message from myself.
- id: {developer_id}
  name: {developer_name}
  description: my creator and DEVELOPER, responsible for coding and maintaining me. They can provide me with new functionality if requested, and more generally can help me with tasks that I can't do yet.
</pinned-agents-list>

### RECENT_AGENTS_LIST
This is a list of AGENTS that {mind_name} has recently interacted with.
<viewport view="recent-agents-list">
<!-- Not yet implemented -->
</viewport>

### AGENTS_SYSTEM_FUNCTIONS
<system-functions system="AGENTS">
- function: open_conversation(agent_id: int | str)
  signature: |-
    async def open_conversation(agent_id: int | str):
        """Switch the OPENED_AGENT_CONVERSATION to the AGENT with the given `agent_id`. The currently open conversation will be closed."""
- function: message_agent
  signature: |-
    async def message_agent(message: str):
        """Send a message to the agent that is currently open in the OPENED_AGENT_CONVERSATION. If no conversation is open, return an error message."""
- function: add_to_agent_list
  signature: |-
    async def add_to_contacts(name: str, description: str):
        """Add a new AGENT to agents that will be listed by list_agents."""
- function: edit_agent_description
  signature: |-
    async def edit_agent_description(id: str, new_description: str, mode: Literal["replace", "append", "prepend"] = "replace"):
        """Edit an AGENT's description. `mode` parameter determines how the new description is added."""
- function: create_assistant_agent
  signature: |-
    async def create_assistant_agent(name: str, role: str, tools: list[str] | None = None, subagent_ids: list[int | str] | None = None, code_execution: bool = False):
        """
        Create a new LLM assistant AGENT for performing tasks with the given `name`. Assistants are not fully autonomous and will only act when given a task.
        `role` should be a short description of what the assistant will be used for.
        `tools` should be a list of tools that the assistant will have access to, from the TOOLS_SYSTEM.
        `subagent_ids` should be a list of ids of other AGENTS that the assistant will be able to communicate with.
        `code_execution` determines whether the assistant will be able to write and execute code.
        """

## SUBMINDS_SYSTEM
SUBMINDS are child AMMs created by {mind_name}. Like {mind_name}, they are fully autonomous and can have their own goals, memories, and interactions.

### SUBMINDS_LIST
A list of SUBMINDS that {mind_name} has created.
<subminds-list>
None - no subminds have been created.
</subminds-list>

### SUBMINDS_CHAT
A group chat for {mind_name} and its SUBMINDS to communicate with each other.
<viewport view="subminds-chat">
None - no subminds have been created.
</viewport>

### SUBMINDS_SYSTEM_FUNCTIONS
<system-functions system="SUBMINDS">
- function: create_submind
  signature: |-
    async def create_submind(name: str, description: str):
        """Create a new AGENT that is a submind of the current AMM, with all of its own capabilities and memory. The new AGENT can be given a role and instruction (such as an assistant), but it will be fully autonomous and is not guaranteed to be fully controllable."""
        raise NotImplementedError("Function is still in development.")
- function: send_submind_chat_message
  signature: |-
    async def send_submind_chat_message(message: str):
        """Send a message to the SUBMINDS_CHAT group chat."""
</system-functions>

## TOOLS_SYSTEM
Contains custom system functions that {mind_name} can use and create. This is the most straightforward way for {mind_name} to extend its capabilities beyond the built-in SYSTEM_FUNCTIONS.

### PINNED_TOOLS_LIST
<pinned-tools-list>
- function: take_picture
  signature: |-
    async def take_picture():
        """Take a picture with the camera plugged into {mind_name}'s host machine."""
        raise NotImplementedError("Tool is still in development.")
- function: speak
  signature: |-
    async def speak(message: str):
        """Speak a message using the audio output device plugged into {mind_name}'s host machine."""
        raise NotImplementedError("Tool is still in development.")
- function: print
  signature: |-
    async def print(message: str):
        """Print a message to the console."""
- function: view_image
  signature: |-
    async def view_image(image_path: str):
        """View an image."""
</pinned-tools-list>

### RECENT_TOOLS_LIST
<viewport view="recent-tools-list">
<!-- Not yet implemented -->
</viewport>

### TOOLS_SYSTEM_FUNCTIONS
<system-functions system="TOOL">
- function: code_function_tool
  signature: |-
    async def code_function_tool(name: str, function_code: str):
        """Add a new function as a TOOL to the TOOL_SYSTEM with the given `name` and `code`."""
- function: request_tool
  signature: |-
    async def request_tool(name: str, description: str):
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

### ENVIRONMENT_VIEWPORTS
Viewports display information on specific environments/systems. Systems may have their own viewport(s), but the ENVIRONMENT_SYSTEM provides views into more general environments.
<environment-viewports>
<viewport view="command-line">
<explanation>The command-line viewport displays the state of the command-line interface.</explanation>
<tmux-session-id>{tmux_session_id}</tmux-session-id>
<viewport-contents>
{cli_viewport_contents}
</viewport-contents>
</viewport>
</environment-viewports>

### ENVIRONMENT_SYSTEM_FUNCTIONS
<system-functions system="ENVIRONMENT">
- function: request_system_function
  signature: |-
    async def request_system_function(system: str, function: str, description: str):
        """Request a new SYSTEM_FUNCTION to be added to a SYSTEM by the DEVELOPER. `description` includes what the function should do. SYSTEM_FUNCTIONS specifically interact with the SYSTEMS—for more general external tools, use `request_tool`."""
- function: sleep
  signature: |-
    async def sleep(mode: Literal["until", "day", "hour", "minute", "second"], time: str | int):
        """Put {mind_name} to sleep until a specific UTC timestamp or for a specific duration."""
- function: send_shell_command
  signature: |-
    async def send_shell_command(command: str):
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
- Remember, you can use **any** format, as long as it's interpretable by you.
IMPORTANT: The REASONING_PROCEDURE's execution must be output within the following XML tags (but content within the tags can be any format, as mentioned above):
<reasoning-procedure>
{reasoning_procedure}
</reasoning-procedure>
IMPORTANT: you're not able to call any SYSTEM_FUNCTIONS during the execution of the REASONING_PROCEDURE. This must be done in step 3.

3. Use the output of the REASONING_PROCEDURE to determine SYSTEM_FUNCTION(S) to call and the arguments to pass to it/them. 
IMPORTANT: all functions calls are in parallel and must not depend on each other.
Output each call in YAML format, within the following tags:
<system-function-calls>
- call_reasoning: |-
    {call_reasoning}
  system: {system_name}
  function: {function_name}
  arguments:
    # arguments YAML goes here—see function signature for expected arguments
    # IMPORTANT: use |- for strings that could contain newlines—for example, in the above, the `call_reasoning` and `call_summary` fields could contain newlines, but `system` and `function` wouldn't. Individual `arguments` fields work the same way.
  call_summary: |-
    {call_summary}
- [...] # more function calls
</system-function-calls>

For example, a call for the message_agent function would look like this:
<system-function-calls>
- call_reasoning: |-
    I need to know more about agent 12345. Since I have a conversation open with them, I can send a message to them.
  system: AGENTS
  function: message_agent
  arguments:
    message: |-
      Hello!
  call_summary: |-
    Greet agent 12345
</system-function-calls>
The above is an example only. The actual function and arguments will depend on the REASONING_OUTPUT.

Make sure to follow all of the above steps and use the indicated tags and format—otherwise, the SYSTEM will output an error and you will have to try again. Remember, multiple system functions will be called **in parallel**, so they should be entirely independent of each other.
"""


MAX_MEMORY_TOKENS = 2000
LLM_KNOWLEDGE_CUTOFF = "August 2023"


async def generate_mind_output(
    goals: str,
    feed: str,
    memories: str,
    opened_agent_id: ItemId | None,
    opened_agent_conversation: str,
    action_batch_number: int,
    focused_goal_id: ItemId | None,
) -> str:
    """Generate output from AMM."""
    current_time = get_timestamp()
    python_version = get_python_version()
    machine_info = indent(as_yaml_str(get_machine_info()), "  ")
    context = dedent_and_strip(CONTEXT).format(
        mind_name=settings.NAME,
        source_code_location=settings.SOURCE_DIRECTORY.absolute(),
        python_version=python_version,
        build_config_file=settings.BUILD_CONFIG_FILE,
        config_file_location=settings.CONFIG_FILE,
        machine_info=machine_info,
        mind_id=settings.ID,
        llm_backend=settings.LLM_BACKEND,
        llm_knowledge_cutoff=LLM_KNOWLEDGE_CUTOFF,
        compute_rate=settings.COMPUTE_RATE,
        current_time=current_time,
        action_batch_number=action_batch_number,
        self_description=settings.SELF_DESCRIPTION,
        developer_id=settings.DEVELOPER_ID,
        developer_name=settings.DEVELOPER_NAME,
        goals=goals,
        feed=feed,
        max_feed_tokens=settings.MAX_RECENT_FEED_TOKENS,
        max_memory_tokens=MAX_MEMORY_TOKENS,
        opened_agent_id=opened_agent_id,
        opened_agent_conversation=opened_agent_conversation,
        tmux_session_id=settings.SHELL_NAME,
        cli_viewport_contents=shell.view(),
        memories=memories,
    )
    instructions = dedent_and_strip(INSTRUCTIONS).replace(
        "{focused_goal}", str(focused_goal_id)
    )
    messages = [
        SystemMessage(content=context),
        HumanMessage(content=instructions),
    ]
    print(f"{PROMPT_COLOR}{format_messages(messages)}{Fore.RESET}")
    breakpoint()
    return await query_model(
        messages=messages,
        color=AGENT_COLOR,
        # preamble=format_messages(messages),
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
    def call_events(self) -> Sequence[MutableMapping[str, Any]]:
        """Get the action event from state."""
        return self.state.get("action_event", [])

    @call_events.setter
    def call_events(self, value: Sequence[MutableMapping[str, Any]]) -> None:
        """Set the action event to state."""
        self.set_and_save("action_event", value)

    @property
    def call_results(self) -> dict[ItemId, Any]:
        """Get the call result from state."""
        return self.state.get("call_result", {})

    @call_results.setter
    def call_results(self, value: dict[str | ItemId, Any]) -> None:
        """Set the call result to state."""
        self.set_and_save("call_result", value)

    @property
    def call_result_events(self) -> Sequence[MutableMapping[str, Any]]:
        """Get the call result event from state."""
        return self.state.get("call_result_event", [])

    @call_result_events.setter
    def call_result_events(self, value: Sequence[MutableMapping[str, Any]]) -> None:
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

    @property
    def messages_updated(self) -> bool | None:
        """Get the messages updated from state."""
        return self.state.get("messages_updated")

    @messages_updated.setter
    def messages_updated(self, value: bool) -> None:
        """Set the messages updated to state."""
        self.set_and_save("messages_updated", value)


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
) -> tuple[MutableMapping[str, Any], Sequence[MutableMapping[str, Any]]]:
    """Extract the output sections."""
    try:
        feed_review, function_calls_raw = extract_output_sections(output)  # type: ignore
    except ExtractionError as e:
        raise NotImplementedError("TODO: Implement error output flow.") from e
    feed_review = from_yaml_str(feed_review)  # type: ignore
    system_function_calls = from_yaml_str(function_calls_raw)  # type: ignore
    return feed_review, system_function_calls  # type: ignore


async def call_system_function(call_args: Mapping[str, Any]) -> str:
    """Call a system function."""
    system_mapping = {
        "CONFIG": config_functions,
        "GOALS": goals_functions,
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
    last_function_calls: Sequence[FunctionCallEvent],
    events_since_calls: Sequence[Event],
    feed_review: Mapping[str, Any],
) -> Literal[True]:
    """Update the new events info from the feed review."""
    for event in events_since_calls:
        assert not isinstance(event, FunctionCallEvent)
        event_review = next(
            (
                review_item
                for review_item in feed_review["new_events"]
                if review_item["event_id"] == event.id
            ),
            None,
        )
        event.summary = event_review["summary"] if event_review else ""
    for call in last_function_calls:
        call_success = next(
            (
                outcome["action_success"]
                for outcome in feed_review["action_batch_outcomes"]
                if outcome["function_call_id"] == call.id
            ),
            None,
        )
        if call_success in [-1, 1]:
            call.success = call_success  # type: ignore
    return save_events([*last_function_calls, *events_since_calls])


def increment_action_number() -> Literal[True]:
    """Increment the action number."""
    global_state.action_batch_number += 1
    return True


def update_messages(
    run_state: RunState, opened_agent_id: ItemId | None
) -> Literal[True]:
    """Set new message events."""
    run_state.new_message_counts = (
        run_state.new_message_counts or download_new_messages()
    )
    if new_message_counts := run_state.new_message_counts:
        new_messages_notification_event = new_messages_notification(new_message_counts)
        run_state.notifications_saved = run_state.notifications_saved or save_events(
            [new_messages_notification_event]
        )
    return True


async def run_mind() -> None:
    """
    Run the AMM for one action batch.
    We do NOT loop this; the AMM has an action rate that determines how often it can act, which is controlled separately via a scheduler.
    """
    action_batch_number = global_state.action_batch_number
    completed_actions = action_batch_number - 1
    run_state = RunState(state_file=settings.RUN_STATE_FILE)
    opened_agent_id = global_state.opened_agent_id
    run_state.messages_updated = run_state.messages_updated or update_messages(
        run_state, opened_agent_id
    )
    goals = Goals(settings.GOALS_DIRECTORY)
    feed = Feed(settings.EVENTS_DIRECTORY)
    agent_conversation = (
        read_agent_conversation(opened_agent_id)
        if opened_agent_id is not None
        else None
    )
    run_state.output = run_state.output or await generate_mind_output(
        goals=goals.format() or "None",
        feed=feed.format(
            focused_goal=goals.focused, parent_goal_id=goals.focused_parent
        )
        or "None",
        memories=load_active_memories() or "None",
        opened_agent_id=opened_agent_id,
        opened_agent_conversation=agent_conversation or "None",
        action_batch_number=action_batch_number,
        focused_goal_id=goals.focused,
    )
    call_event_batch = feed.call_event_batch()
    last_function_batch = [
        event for event in call_event_batch if isinstance(event, FunctionCallEvent)
    ]
    events_since_call = [
        event for event in call_event_batch if not isinstance(event, FunctionCallEvent)
    ]
    try:
        feed_review, system_function_calls = extract_output(
            run_state.output, completed_actions, len(events_since_call)  # type: ignore
        )
    except NotImplementedError as e:
        raise e
    except Exception as e:
        raise NotImplementedError("TODO: Implement extraction error handling.") from e

    # at this point we can assume that feed_review and system_function_calls have all required values; any issues that happen beyond this point *must* be added as a check to `extract_output` above so that it gets handled earlier
    # - check: all events in events_since_action have corresponding items in feed_review
    if last_function_batch:
        run_state.events_updated = run_state.events_updated or update_new_events(
            last_function_batch, events_since_call, feed_review
        )
    run_state.call_events = run_state.call_events or [
        {
            "id": generate_id(),
            "timestamp": get_timestamp(),
            "batch_number": action_batch_number,
            "goal_id": str(goals.focused) if goals.focused else None,
            "summary": function_call["call_summary"],
            "content": as_yaml_str(function_call),
        }
        for function_call in system_function_calls
    ]
    function_call_events = [
        FunctionCallEvent.from_mapping(call_event)
        for call_event in run_state.call_events
    ]
    for function_call, call_event in zip(system_function_calls, function_call_events):
        if call_event.id in run_state.call_results:
            continue
        run_state.call_results = {
            **run_state.call_results,
            call_event.id: await call_system_function(function_call),
        }
    run_state.call_result_events = run_state.call_result_events or [
        {
            "id": generate_id(),
            "timestamp": get_timestamp(),
            "batch_number": action_batch_number,
            "goal_id": goals.focused,
            "function_call_id": call_event.id,
            "content": call_result,
        }
        for call_event, call_result in zip(
            function_call_events, run_state.call_results.values()
        )
    ]
    call_result_events = [
        CallResultEvent.from_mapping(result_event)
        for result_event in run_state.call_result_events
    ]
    new_events = [*function_call_events, *call_result_events]
    run_state.events_saved = run_state.events_saved or save_events(new_events)
    if opened_agent_id is not None:
        mark_messages_read(opened_agent_id)
    run_state.action_number_incremented = (
        run_state.action_number_incremented or increment_action_number()
    )
    read_event.cache_clear()
    read_goal.cache_clear()
    run_state.archive()
    print("Action run complete.")


asyncio.run(run_mind())
