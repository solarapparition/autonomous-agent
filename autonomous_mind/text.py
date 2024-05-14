"""Extract blocks from text."""

from dataclasses import dataclass
import re
from textwrap import dedent, indent


def dedent_and_strip(text: str) -> str:
    """Dedent and strip text."""
    return dedent(text).strip()


@dataclass(frozen=True)
class ExtractionError(Exception):
    """Raised when an extraction fails."""

    num_blocks_found: int
    text: str | None = None
    start_block_type: str | None = None
    end_block_type: str | None = None

    @property
    def message(self) -> str:
        """Get the error message."""
        template = """
        Failed to extract a block:
        - Start block type: {start_block_type}
        - End block type: {end_block_type}
        - Number of blocks found: {num_blocks_found}
        - Text:
        {text}
        """
        text = indent(str(self.text or "N/A"), "  ")
        output = template.format(
            start_block_type=self.start_block_type or "N/A",
            end_block_type=self.end_block_type or "N/A",
            num_blocks_found=self.num_blocks_found,
            text=text,
        )
        return dedent_and_strip(output)


def extract_block(text: str, block_type: str) -> str | None:
    """Extract a code block from the text."""
    pattern = (
        r"```{block_type}\n(.*?)```".format(  # pylint:disable=consider-using-f-string
            block_type=block_type
        )
    )
    match = re.search(pattern, text, re.DOTALL)
    return match[1].strip() if match else None


def extract_blocks(
    text: str, start_block_type: str, end_block_type: str = "", prefix: str = ""
) -> list[str] | None:
    """Extracts specially formatted blocks of text from the LLM's output. `block_type` corresponds to a label for a markdown code block such as `yaml` or `python`."""
    pattern = r"{prefix}{start_block_type}\n(.*?){prefix}{end_block_type}".format(  # pylint:disable=consider-using-f-string
        start_block_type=start_block_type,
        end_block_type=end_block_type,
        prefix=prefix,
    )
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches] if matches else None


def unpack_block(
    text: str,
    extracted_result: list[str] | None,
    start_block_type: str,
    end_block_type: str,
    allow_multiple: bool,
) -> str:
    """Validate and unpack the extracted block."""
    if (not extracted_result) or (len(extracted_result) > 1 and not allow_multiple):
        num_blocks_found = len(extracted_result) if extracted_result else 0
        raise ExtractionError(
            text=text,
            start_block_type=start_block_type,
            end_block_type=end_block_type,
            num_blocks_found=num_blocks_found,
        )
    return extracted_result[-1]


def extract_and_unpack(
    text: str,
    start_block_type: str,
    end_block_type: str = "",
    prefix: str = "",
    allow_multiple: bool = True,
) -> str:
    """Extract and unpack a block."""
    extracted_result = extract_blocks(
        text, start_block_type, end_block_type, prefix=prefix
    )
    return unpack_block(
        text,
        extracted_result,
        start_block_type,
        end_block_type,
        allow_multiple=allow_multiple,
    )
