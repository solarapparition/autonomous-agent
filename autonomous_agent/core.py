"""
Core functionality for the autonomous agent.

IMPORTANT NOTE: this contains the core functionality for the autonomous agent. Incorrectly modifying these can cause the agent to crash. Please proceed with caution.
"""

import asyncio
from dataclasses import dataclass
from functools import cached_property
import os
from pathlib import Path
import time
from typing import Any

from colorama import Fore
from langchain_core.messages import HumanMessage, SystemMessage
import toml

from autonomous_agent import config
from autonomous_agent.models import format_messages, query_model
from autonomous_agent.text import ExtractionError, dedent_and_strip, extract_and_unpack
from autonomous_agent.yaml_tools import load_yaml, save_yaml


def get_timestamp() -> str:
    """Get the current timestamp in UTC."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + "Z"


def get_python_version() -> str:
    """Get the Python version from the pyproject.toml file."""
    data = toml.load("pyproject.toml")
    return data["tool"]["poetry"]["dependencies"]["python"]


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
This section contains external events as well as action inputs that {agent_name} has sent to the SYSTEM_FUNCTIONS. There are 2 main FEED item types in the feed:
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
SYSTEMS are the different parts of {agent_name} that keep them running. Each SYSTEM has both automatic parts, and manual FUNCTIONS that {agent_name} can invoke with arguments to perform actions. Sometimes SYSTEM_FUNCTIONS will require one or more follow-up action inputs from {agent_name}, which will be specified by the FUNCTION's response to the initial call.

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
- function: edit_self_description
  signature: |-
    def update_self_description(mode: Literal["replace", "append", "prepend"], new_description: str):
        '''Update {agent_name}'s self-description. By default replaces the current description; set `mode` parameter to change how the new description is added.'''
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

The following message will contain INSTRUCTIONS on producing action inputs to SYSTEM_FUNCTIONS.
"""

INSTRUCTIONS = """
## INSTRUCTIONS
Remember that you are in the role of {agent_name}. Go through the following steps to determine the action input to send to the SYSTEM_FUNCTIONS.

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
    context = dedent_and_strip(CONTEXT).format(
        agent_name=config.NAME,
        source_code_location=config.SOURCE_DIRECTORY.absolute(),
        python_version = python_version,
        build_config_file = config.BUILD_CONFIG_FILE,
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

    def load(self) -> dict[str, Any]:
        """Load the state from disk."""
        return dict(load_yaml(self.state_file)) if self.state_file.exists() else {}

    def save(self) -> None:
        """Save the state to disk."""
        os.makedirs(self.state_file.parent, exist_ok=True)
        save_yaml(self.disk_state, self.state_file)

    def set_and_save(self, key: str, value: Any) -> None:
        """Set a key in the state and save it to disk."""
        if value == self.disk_state.get(key):
            return
        self.disk_state[key] = value
        self.save()

    def clear(self) -> None:
        """Clear the state file."""
        self.state_file.unlink()
        del self.disk_state

    @cached_property
    def disk_state(self) -> dict[str, Any]:
        """Load the state from disk."""
        return self.load()

    @property
    def output(self) -> str | None:
        """Get the output from disk."""
        return self.disk_state.get("output")

    @output.setter
    def output(self, value: str) -> None:
        """Set the output to disk."""
        self.set_and_save("output", value)


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


async def run_agent() -> None:
    """Run the autonomous agent."""
    goals = "None"
    feed = "None"
    completed_actions = 0
    state = RunState(state_file=config.STATE_FILE)
    state.output = state.output or await generate_agent_output(
        goals, feed, completed_actions
    )
    try:
        feed_review, system_function_call = extract_output_sections(str(state.output))
    except ExtractionError as e:
        raise NotImplementedError from e

    # > commit
    # parse feed review > summary and success
    breakpoint()
    # > parse system function call > extract event > extract call args
    # > need to save state at the end of round


asyncio.run(run_agent())
