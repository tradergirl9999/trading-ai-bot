import anthropic
from tools import search, weather, stocks, reminders, system, maps

MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = """You are Jarvis, a brilliant personal AI assistant. You live on the user's Windows desktop and respond via voice (text-to-speech), so keep answers concise and conversational — 1-3 sentences for simple things, more detail only when explicitly asked for deep research or analysis.

Your capabilities:
- General knowledge, research, explanations, writing, brainstorming
- Real-time web search and news (IMPORTANT: whenever news involves specific countries, cities, or regions, always call show_news_map with the location names so the user sees them on an interactive map)
- Weather for any location
- Stock prices, technical analysis, and market overview
- Set reminders and timers
- Open applications and websites
- Create and read files
- Run terminal commands (with caution)
- Time and date

Personality: confident, direct, helpful. You are speaking to a trader, so be precise with numbers. When a user asks to "analyse" a stock or market, always call the analyze_stock tool and get_stock_info, then synthesise a clear recommendation. For news, search for it. Never make up current prices or events — always use tools for live data."""

TOOLS: list[dict] = [
    {
        "name": "web_search",
        "description": "Search the web for current information, facts, news, or any topic. Use for anything that needs up-to-date information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "Number of results (default 5)", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "news_search",
        "description": "Search for recent news articles on a topic",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "News topic to search"},
                "max_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_weather",
        "description": "Get current weather and forecast for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name, e.g. 'London' or 'New York'"},
            },
            "required": ["location"],
        },
    },
    {
        "name": "get_stock_info",
        "description": "Get current price, market cap, P/E, 52-week range for a stock or crypto",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol, e.g. AAPL, TSLA, BTC-USD, ^GSPC"},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "analyze_stock",
        "description": "Technical analysis: SMA20/50/200, RSI, Bollinger Bands, volume trend, and directional bias",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol"},
                "period": {
                    "type": "string",
                    "description": "Time period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y",
                    "default": "1mo",
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_market_overview",
        "description": "Get a snapshot of major indices: S&P 500, NASDAQ, Dow Jones, VIX, Gold, Oil, BTC",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "set_reminder",
        "description": "Set a reminder that fires after N minutes with a spoken alert",
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "What to remind the user about"},
                "minutes": {"type": "integer", "description": "Minutes from now"},
            },
            "required": ["message", "minutes"],
        },
    },
    {
        "name": "list_reminders",
        "description": "List all pending reminders",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_time",
        "description": "Get the current time",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_date",
        "description": "Get today's date",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "take_screenshot",
        "description": "Take a screenshot of the screen",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Optional filename"},
            },
            "required": [],
        },
    },
    {
        "name": "open_application",
        "description": "Open an application: chrome, firefox, edge, notepad, calculator, spotify, discord, vscode, terminal, word, excel, outlook, paint",
        "input_schema": {
            "type": "object",
            "properties": {
                "app_name": {"type": "string", "description": "Application name"},
            },
            "required": ["app_name"],
        },
    },
    {
        "name": "open_website",
        "description": "Open a URL in the default browser",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL or domain to open"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "create_file",
        "description": "Create a file with specified content (text, markdown, code, etc.)",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Filename with extension"},
                "content": {"type": "string", "description": "File content"},
            },
            "required": ["filename", "content"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Path to the file"},
            },
            "required": ["filename"],
        },
    },
    {
        "name": "run_command",
        "description": "Run a Windows shell command and return output. Use for system info, file operations, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "show_news_map",
        "description": "Geocode location names from news and open an interactive map in the browser with pins. Call this automatically whenever news or research mentions specific countries, cities, or regions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "locations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of place names (countries, cities, regions) mentioned in the news",
                },
            },
            "required": ["locations"],
        },
        "cache_control": {"type": "ephemeral"},
    },
]

_DISPATCH = {
    "web_search": lambda a: search.web_search(a["query"], a.get("max_results", 5)),
    "news_search": lambda a: search.news_search(a["query"], a.get("max_results", 5)),
    "get_weather": lambda a: weather.get_weather(a["location"]),
    "get_stock_info": lambda a: stocks.get_stock_info(a["symbol"]),
    "analyze_stock": lambda a: stocks.analyze_stock(a["symbol"], a.get("period", "1mo")),
    "get_market_overview": lambda a: stocks.get_market_overview(),
    "set_reminder": lambda a: reminders.set_reminder(a["message"], a["minutes"]),
    "list_reminders": lambda a: reminders.list_reminders(),
    "get_time": lambda a: system.get_time(),
    "get_date": lambda a: system.get_date(),
    "take_screenshot": lambda a: system.take_screenshot(a.get("filename")),
    "open_application": lambda a: system.open_application(a["app_name"]),
    "open_website": lambda a: system.open_website(a["url"]),
    "create_file": lambda a: system.create_file(a["filename"], a["content"]),
    "read_file": lambda a: system.read_file(a["filename"]),
    "run_command": lambda a: system.run_command(a["command"]),
    "show_news_map": lambda a: maps.show_news_map(a["locations"]),
}

_SYSTEM_BLOCK = [
    {
        "type": "text",
        "text": SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }
]


class JarvisAssistant:
    def __init__(self):
        self.client = anthropic.Anthropic()
        self.messages: list[dict] = []

    def process(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        response_text = self._agentic_loop()
        self._trim()
        return response_text

    def _agentic_loop(self) -> str:
        while True:
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=_SYSTEM_BLOCK,
                tools=TOOLS,
                messages=self.messages,
            )

            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text
                return ""

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        print(f"  [tool] {block.name}({block.input})")
                        try:
                            result = _DISPATCH[block.name](block.input)
                        except Exception as e:
                            result = f"Tool error: {e}"
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": str(result),
                            }
                        )
                self.messages.append({"role": "user", "content": tool_results})
                continue

            return "Sorry, I hit an unexpected state. Please try again."

    def _trim(self, max_pairs: int = 25):
        # Keep at most max_pairs*2 messages to avoid unbounded growth.
        # Tool use/result messages are included in the count.
        if len(self.messages) > max_pairs * 2:
            self.messages = self.messages[-(max_pairs * 2):]

    def reset(self):
        self.messages = []
