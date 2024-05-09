"""Model utilities."""

from functools import wraps
from typing import Callable, Generic, Iterable, Sequence, TypeVar

from colorama import Fore
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage  # , AIMessage, ToolCall
from langchain_anthropic import ChatAnthropic

# from langchain_core.tools import tool


OPUS = "claude-3-opus-20240229"
CORE_MODEL = ChatAnthropic(
    temperature=0.8, model=OPUS, verbose=False, max_tokens_to_sample=4096
)


def format_messages(messages: Sequence[BaseMessage]) -> str:
    """Format model messages into something printable."""
    return "\n\n---\n\n".join(
        [f"[{message.type.upper()}]:\n\n{message.content}" for message in messages]  # type: ignore
    )


T = TypeVar("T")
QueryFunction = Callable[..., T]


def wrap_printout(
    query_func: QueryFunction[T], color: str, preamble: str | None, printout: bool
) -> QueryFunction[T]:
    """Wrap a query function to print the result."""

    @wraps(query_func)
    def wrapper(*args: object, **kwargs: object) -> T:
        if preamble is not None:
            print(f"\033[1;34m{preamble}\033[0m")
        result = query_func(*args, **kwargs)
        if printout:
            print(f"{color}{result}{Fore.RESET}")
        return result

    return wrapper


async def query_model(
    messages: Sequence[BaseMessage],
    model: BaseChatModel = CORE_MODEL,
    color: str = Fore.RESET,
    preamble: str | None = None,
    printout: bool = True,
    stream: bool = False,
) -> str:
    """Query an LLM chat model. `preamble` is printed before the result."""

    if stream:

        async def query(messages: Sequence[BaseMessage]) -> str:
            output = model.astream(messages)
            chunks: list[str] = []
            async for chunk in output:
                if printout:
                    print(f"{color}{chunk.content}{Fore.RESET}", end="", flush=True)
                chunks.append(chunk.content)
            return "".join(chunks)

        wrapped_query = wrap_printout(query, color, preamble, printout=False)

    else:

        async def query(messages: Sequence[BaseMessage]) -> str:
            return str(model.ainvoke(messages).content)

        wrapped_query = wrap_printout(query, color, preamble, printout)

    return await wrapped_query(messages)

    # if preamble is not None and printout:
    #     print(f"\033[1;34m{preamble}\033[0m")
    # result = str(model.invoke(messages).content)
    # if printout:
    #     print(f"{color}{result}{Fore.RESET}")
    # return result


# def query_model_tool_call(
#     model: BaseChatModel,
#     messages: Sequence[BaseMessage],
#     tools: Sequence[Callable[..., Any]],
#     color: str = Fore.RESET,
#     preamble: str | None = None,
#     printout: bool = True,
# ) -> list[ToolCall]:
#     """Query an LLM chat model, returning the result of a tool call."""
#     model = model.bind_tools([tool(t) for t in tools])
#     if preamble is not None and printout:
#         print(f"\033[1;34m{preamble}\033[0m")
#     result: AIMessage = model.invoke(messages)
#     call_results = result.tool_calls
#     if printout:
#         print(f"{color}{call_results}{Fore.RESET}")
#     return call_results


# def call_tools(
#     tools: Sequence[Callable[..., Any]], call_results: list[dict[str, Any]]
# ) -> list[dict[str, Any]]:
#     """Run the tools on the call args."""
#     for call_result in call_results:
#         next_tool = next(t for t in tools if t.__name__ == call_result["name"])
#         call_result["output"] = next_tool(**call_result["args"])
#     return call_results
