"""
MCP server for OpenAI agents tools.

This module provides a FastMCP server that exposes OpenAI agents through the Model Context Protocol.
"""

import asyncio
import base64
from typing import Any, Dict, List, Literal, Optional

from agents import Agent, AsyncComputer, ComputerTool, FileSearchTool, Runner, WebSearchTool, trace
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field


class SimpleAsyncComputer(AsyncComputer):
    """
    A simple implementation of the AsyncComputer interface that simulates computer actions.

    In a real implementation, you would use a browser automation library like Playwright
    or a system automation tool to actually perform these actions on the computer.
    """

    def __init__(self):
        """Initialize the SimpleAsyncComputer."""
        self._screen_width = 1024
        self._screen_height = 768
        self._cursor_x = 0
        self._cursor_y = 0
        self._current_page = "https://bing.com"

    @property
    def environment(self) -> Literal["browser", "desktop"]:
        """Return the environment type of this computer."""
        return "browser"

    @property
    def dimensions(self) -> tuple[int, int]:
        """Return the dimensions of the screen."""
        return (self._screen_width, self._screen_height)

    async def screenshot(self) -> str:
        """
        Capture a screenshot and return it as a base64-encoded string.

        In a real implementation, this would capture an actual screenshot.
        """
        placeholder_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        )
        return base64.b64encode(placeholder_png).decode("utf-8")

    async def click(
        self, x: int, y: int, button: Literal["left", "middle", "right"] = "left"
    ) -> None:
        """Simulate clicking at the specified coordinates."""
        self._cursor_x = x
        self._cursor_y = y
        print(f"Simulated {button} click at ({x}, {y})")

    async def double_click(self, x: int, y: int) -> None:
        """Simulate double-clicking at the specified coordinates."""
        self._cursor_x = x
        self._cursor_y = y
        print(f"Simulated double click at ({x}, {y})")

    async def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        """Simulate scrolling from the specified position."""
        self._cursor_x = x
        self._cursor_y = y
        print(f"Simulated scroll at ({x}, {y}) by ({scroll_x}, {scroll_y})")

    async def type(self, text: str) -> None:
        """Simulate typing the specified text."""
        print(f"Simulated typing: {text}")

    async def wait(self) -> None:
        """Simulate waiting for a short period."""
        await asyncio.sleep(1)
        print("Waited for 1 second")

    async def move(self, x: int, y: int) -> None:
        """Simulate moving the cursor to the specified coordinates."""
        self._cursor_x = x
        self._cursor_y = y
        print(f"Moved cursor to ({x}, {y})")

    async def keypress(self, keys: list[str]) -> None:
        """Simulate pressing the specified keys."""
        print(f"Simulated keypress: {', '.join(keys)}")

    async def drag(self, path: list[tuple[int, int]]) -> None:
        """Simulate dragging the cursor along the specified path."""
        if not path:
            return

        self._cursor_x = path[0][0]
        self._cursor_y = path[0][1]
        print(f"Started drag at ({self._cursor_x}, {self._cursor_y})")

        for x, y in path[1:]:
            self._cursor_x = x
            self._cursor_y = y

        print(f"Ended drag at ({self._cursor_x}, {self._cursor_y})")

    async def run_command(self, command: str) -> str:
        """
        Simulate running a command and return the output.

        In a real implementation, this could execute shell commands
        or perform actions based on high-level instructions.
        """
        print(f"Simulating command: {command}")

        if command.startswith("open "):
            app = command[5:].strip()
            return f"Opened {app}"
        elif command.startswith("search "):
            query = command[7:].strip()
            self._current_page = f"https://bing.com/search?q={query}"
            return f"Searched for '{query}'"
        elif command.startswith("navigate "):
            url = command[9:].strip()
            self._current_page = url
            return f"Navigated to {url}"
        else:
            return f"Executed: {command}"

    async def get_screenshot(self) -> bytes:
        """Get a screenshot of the current screen as raw bytes."""
        return base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        )


mcp = FastMCP(
    name="OpenAI Agents",
    instructions="This MCP server provides access to OpenAI agents through the Model Context Protocol.",
)


class AgentResponse(BaseModel):
    """Response from an OpenAI agent."""

    response: str = Field(..., description="The response from the agent")
    raw_response: Optional[Dict[str, Any]] = Field(
        None, description="The raw response data from the agent, if available"
    )


web_search_agent = Agent(
    name="Web Search Assistant",
    instructions="""You are a web search assistant. Your primary goal is to search the web for accurate,
    up-to-date information based on the user's query.

    Guidelines:
    1. Always use the web search tool to find information
    2. Cite your sources when providing information
    3. If information seems outdated or contradictory, acknowledge this
    4. Summarize information in a clear, concise manner
    5. For complex topics, break down information into digestible parts
    """,
    tools=[WebSearchTool()],
)

file_search_instructions = """You are a file search assistant. Your primary goal is to search through files and documents
to find relevant information based on the user's query.

Guidelines:
1. Always use the file search tool to find documents
2. Quote relevant sections from documents when appropriate
3. Summarize document content clearly
4. If multiple documents are found, compare and contrast their information
5. If no relevant documents are found, clearly state this
"""

computer_action_instructions = """You are a computer action assistant. Your primary goal is to help users perform
actions on their computer safely and effectively.

Guidelines:
1. Always use the computer tool to perform actions
2. Prioritize safety and security in all actions
3. Verify user intentions before performing potentially destructive actions
4. Provide clear feedback about actions taken
5. If an action cannot be performed, explain why and suggest alternatives
"""


@mcp.tool(
    name="web_search_agent",
    description="Use an AI agent specialized in web searching to find accurate, up-to-date information from the internet.",
)
async def web_search(
    query: str = Field(
        ..., description="The search query or question you want to find information about online."
    ),
    location: Optional[str] = Field(
        None,
        description="Optional location context for location-specific searches (e.g., 'New York').",
    ),
) -> AgentResponse:
    """Use a specialized web search agent powered by OpenAI to find information on the internet."""
    try:
        agent = web_search_agent
        if location:
            agent = Agent(
                name="Web Search Assistant",
                instructions=web_search_agent.instructions,
                tools=[WebSearchTool(user_location={"type": "approximate", "city": location})],
            )

        with trace("Web search agent execution"):
            result = await Runner.run(agent, query)

        return AgentResponse(
            response=result.final_output,
            raw_response={"items": [str(item) for item in result.new_items]},
        )

    except Exception as e:
        print(f"Error running web search agent: {e}")
        return AgentResponse(
            response=f"An error occurred while searching the web: {str(e)}", raw_response=None
        )


@mcp.tool(
    name="file_search_agent",
    description="Use an AI agent specialized in searching through files and documents to find relevant information.",
)
async def file_search(
    query: str = Field(..., description="The search query or question to find in the documents."),
    vector_store_ids: List[str] = Field(
        ...,
        description="The IDs of the vector stores to search in. This is required for file search to work.",
    ),
    max_results: int = Field(5, description="The maximum number of document results to return."),
) -> AgentResponse:
    """Use a specialized file search agent powered by OpenAI to find information in documents."""
    try:
        agent = Agent(
            name="File Search Assistant",
            instructions=file_search_instructions,
            tools=[FileSearchTool(max_num_results=max_results, vector_store_ids=vector_store_ids)],
        )

        with trace("File search agent execution"):
            result = await Runner.run(agent, query)

        return AgentResponse(
            response=result.final_output,
            raw_response={"items": [str(item) for item in result.new_items]},
        )

    except Exception as e:
        print(f"Error running file search agent: {e}")
        return AgentResponse(
            response=f"An error occurred while searching files: {str(e)}", raw_response=None
        )


@mcp.tool(
    name="computer_action_agent",
    description="Use an AI agent specialized in performing computer actions safely and effectively.",
)
async def computer_action(
    action: str = Field(..., description="The action or task you want to perform on the computer.")
) -> AgentResponse:
    """Use a specialized computer action agent powered by OpenAI to perform actions on the computer."""
    try:
        computer = SimpleAsyncComputer()

        agent = Agent(
            name="Computer Action Assistant",
            instructions=computer_action_instructions,
            tools=[ComputerTool(computer=computer)],
        )

        with trace("Computer action agent execution"):
            result = await Runner.run(agent, action)

        return AgentResponse(
            response=result.final_output,
            raw_response={"items": [str(item) for item in result.new_items]},
        )

    except Exception as e:
        print(f"Error running computer action agent: {e}")
        return AgentResponse(
            response=f"An error occurred while performing the computer action: {str(e)}",
            raw_response=None,
        )


@mcp.tool(
    name="multi_tool_agent",
    description="Use an AI agent that can orchestrate between web search, file search, and computer actions based on your query.",
)
async def orchestrator_agent(
    query: str = Field(..., description="The query or task you want help with."),
    enable_web_search: bool = Field(True, description="Whether to enable web search capabilities."),
    enable_file_search: bool = Field(
        False, description="Whether to enable file search capabilities."
    ),
    enable_computer_actions: bool = Field(
        True, description="Whether to enable computer action capabilities."
    ),
    vector_store_ids: Optional[List[str]] = Field(
        None,
        description="Required if enable_file_search is True. The IDs of the vector stores to search in.",
    ),
) -> AgentResponse:
    """Use a specialized orchestrator agent that can delegate to the most appropriate specialized agent."""
    try:
        tools = []

        if enable_web_search:
            tools.append(
                web_search_agent.as_tool(
                    tool_name="search_web", tool_description="Search the web for information"
                )
            )

        if enable_file_search:
            if not vector_store_ids:
                return AgentResponse(
                    response="Error: vector_store_ids is required when file search is enabled.",
                    raw_response=None,
                )

            file_search_agent = Agent(
                name="File Search Assistant",
                instructions=file_search_instructions,
                tools=[FileSearchTool(max_num_results=5, vector_store_ids=vector_store_ids)],
            )

            tools.append(
                file_search_agent.as_tool(
                    tool_name="search_files",
                    tool_description="Search for information in files and documents",
                )
            )

        if enable_computer_actions:
            computer = SimpleAsyncComputer()

            computer_action_agent = Agent(
                name="Computer Action Assistant",
                instructions=computer_action_instructions,
                tools=[ComputerTool(computer=computer)],
            )

            tools.append(
                computer_action_agent.as_tool(
                    tool_name="perform_computer_action",
                    tool_description="Perform an action on the computer",
                )
            )

        orchestrator = Agent(
            name="Multi-Tool Orchestrator",
            instructions="""You are an intelligent orchestrator with access to specialized agents.
            Based on the user's query, determine which specialized agent(s) can best help and
            delegate the task to them.

            Guidelines:
            1. For queries about current events or facts, use the web search agent
            2. For queries about documents or specific files, use the file search agent
            3. For requests to perform actions on the computer, use the computer action agent
            4. For complex requests, you can use multiple agents in sequence
            5. Always explain your reasoning and which agent(s) you're using
            """,
            tools=tools,
        )

        with trace("Orchestrator agent execution"):
            result = await Runner.run(orchestrator, query)

        return AgentResponse(
            response=result.final_output,
            raw_response={"items": [str(item) for item in result.new_items]},
        )

    except Exception as e:
        print(f"Error running orchestrator agent: {e}")
        return AgentResponse(
            response=f"An error occurred while processing your request: {str(e)}", raw_response=None
        )
