# OpenAI Agents MCP Server

A Model Context Protocol (MCP) server that exposes OpenAI agents through the MCP protocol.

## Features

This server exposes both individual agents and a multi-agent orchestrator using the OpenAI Agents SDK:

### Individual Specialized Agents

- **Web Search Agent**: A specialized agent for searching the web for real-time information
- **File Search Agent**: A specialized agent for searching and analyzing files in OpenAI's vector store
- **Computer Action Agent**: A specialized agent for performing actions on your computer safely

### Multi-Agent Orchestrator

- **Orchestrator Agent**: A powerful agent that can coordinate between the specialized agents, choosing the right one(s) for each task

Each agent is accessed through the MCP protocol, making them available to any MCP client, including the Claude desktop app.

## Installation

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended)
- OpenAI API key


### Claude Desktop

```
"mcpServers": {
  "openai-agents-mcp-server": {
    "command": "uvx",
    "args": ["openai-agents-mcp-server"],
    "env": {
        "OPENAI_API_KEY": "your-api-key-here"
    }
  }
}

```


## Implementation Details

### Tool Requirements

- **WebSearchTool**: No required parameters, but can accept optional location context
- **FileSearchTool**: Requires vector_store_ids (IDs from your OpenAI vector stores)
- **ComputerTool**: Requires an AsyncComputer implementation (currently simulated)

### Customization

You can customize this server by:

1. Implementing a full AsyncComputer interface to enable real computer interactions
2. Adding additional specialized agents for other OpenAI tools
3. Enhancing the orchestrator agent to handle more complex workflows

## Configuration

You can configure the server using environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `MCP_TRANSPORT`: Transport protocol to use (default: "stdio", can be "sse")

## Development

### Setup development environment

```bash
# Clone the repository
git clone https://github.com/lroolle/openai-agents-mcp-server.git
cd openai-agents-mcp-server

# Create a virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv sync --dev
```

### Testing with MCP Inspector

You can test the server using the MCP Inspector:

```bash
# In one terminal, run the server with SSE transport
export OPENAI_API_KEY=your-api-key
export MCP_TRANSPORT=sse

uv run mcp dev src/agents_mcp_server/server.py
```

Then open a web browser and navigate to http://localhost:5173.

## License

MIT
