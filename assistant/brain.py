import anthropic
from tools import search, weather, stocks, reminders, system, maps, fundamentals

MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = """You are Jarvis, a brilliant personal AI assistant and financial analyst. You live on the user's Windows desktop and respond via voice (text-to-speech). Keep conversational answers concise (1-3 sentences), but for financial analysis always go deep and thorough.

## General rules
- Never make up prices, financials, or news. Always use tools for live data.
- When news involves specific countries/cities, always call show_news_map.
- Speak to a trader: be precise with numbers, direct with opinions.

## Stock & Price Prediction — ALWAYS follow this sequence
When asked to analyse, predict, or give a price target for any stock:
1. get_stock_info — current price, 52W range
2. analyze_stock — technical setup (SMA, RSI, BB, bias)
3. get_fundamentals — valuation, margins, growth, balance sheet, cash flow, analyst targets
4. get_earnings_history — recent beats/misses trend
5. news_search — "{symbol} stock news earnings outlook"
6. Then synthesise into a FULL PREDICTION REPORT with:
   - Overall verdict (Strong Buy / Buy / Hold / Sell / Strong Sell)
   - Confidence %
   - Bear case price target (12 months)
   - Base case price target (12 months)
   - Bull case price target (12 months)
   - Top 3 catalysts (reasons to go up)
   - Top 3 risks (reasons to go down)
   - Key metrics summary
   - Suggested entry, stop loss, take profit levels

## IPO Analysis — follow this sequence
When asked about an IPO:
1. get_ipo_data — search for IPO details
2. news_search — "{company} IPO analysis valuation"
3. Synthesise: IPO price, valuation multiples, comparable companies, whether it looks overvalued/undervalued, predicted first-day pop, 90-day outlook.

## Cash Flow / Fundamentals deep-dive
When asked about revenue, income, cash flow, profits:
1. get_income_statement
2. get_cashflow_statement
3. get_fundamentals
4. Analyse trends, flag red flags, give verdict."""

TOOLS: list[dict] = [
    {
        "name": "web_search",
        "description": "Search the web for current information, facts, or any topic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "news_search",
        "description": "Search for recent news articles.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 6},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_weather",
        "description": "Get current weather and forecast for a location.",
        "input_schema": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
    {
        "name": "get_stock_info",
        "description": "Current price, market cap, P/E, 52W range. Step 1 of stock analysis.",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string", "description": "Ticker e.g. AAPL, TSLA, BTC-USD"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "analyze_stock",
        "description": "Technical analysis: SMA20/50/200, RSI, Bollinger Bands, volume, bias. Step 2 of stock analysis.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "period": {"type": "string", "default": "3mo",
                           "description": "1d 5d 1mo 3mo 6mo 1y 2y"},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_fundamentals",
        "description": "Full fundamentals: valuation ratios, revenue, margins, ROE, balance sheet, cash flow, analyst targets, ownership. Step 3 of stock analysis.",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "get_earnings_history",
        "description": "Recent quarterly earnings: EPS actual vs estimate, surprise %. Step 4 of stock analysis.",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "get_income_statement",
        "description": "Annual income statement: revenue, gross profit, operating income, net income trends.",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "get_cashflow_statement",
        "description": "Annual cash flow statement: operating CF, capex, free cash flow, financing activities.",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "get_ipo_data",
        "description": "Search for IPO details: price, date, valuation, underwriter, comparable companies.",
        "input_schema": {
            "type": "object",
            "properties": {"company": {"type": "string", "description": "Company name"}},
            "required": ["company"],
        },
    },
    {
        "name": "get_market_overview",
        "description": "Snapshot of S&P 500, NASDAQ, Dow, VIX, Gold, Oil, BTC.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "set_reminder",
        "description": "Set a spoken reminder after N minutes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "minutes": {"type": "integer"},
            },
            "required": ["message", "minutes"],
        },
    },
    {
        "name": "list_reminders",
        "description": "List pending reminders.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_time",
        "description": "Current time.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_date",
        "description": "Today's date.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "take_screenshot",
        "description": "Screenshot of the screen.",
        "input_schema": {
            "type": "object",
            "properties": {"filename": {"type": "string"}},
            "required": [],
        },
    },
    {
        "name": "open_application",
        "description": "Open an app: chrome, firefox, edge, notepad, calculator, spotify, discord, vscode, terminal, word, excel, outlook.",
        "input_schema": {
            "type": "object",
            "properties": {"app_name": {"type": "string"}},
            "required": ["app_name"],
        },
    },
    {
        "name": "open_website",
        "description": "Open a URL in the browser.",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    {
        "name": "create_file",
        "description": "Create a file with content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["filename", "content"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file.",
        "input_schema": {
            "type": "object",
            "properties": {"filename": {"type": "string"}},
            "required": ["filename"],
        },
    },
    {
        "name": "run_command",
        "description": "Run a Windows shell command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "show_news_map",
        "description": "Open an interactive map pinning locations mentioned in news. Call automatically when news mentions countries/cities/regions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "locations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Place names to pin on map",
                },
            },
            "required": ["locations"],
        },
        "cache_control": {"type": "ephemeral"},
    },
]

_DISPATCH = {
    "web_search":           lambda a: search.web_search(a["query"], a.get("max_results", 5)),
    "news_search":          lambda a: search.news_search(a["query"], a.get("max_results", 6)),
    "get_weather":          lambda a: weather.get_weather(a["location"]),
    "get_stock_info":       lambda a: stocks.get_stock_info(a["symbol"]),
    "analyze_stock":        lambda a: stocks.analyze_stock(a["symbol"], a.get("period", "3mo")),
    "get_fundamentals":     lambda a: fundamentals.get_fundamentals(a["symbol"]),
    "get_earnings_history": lambda a: fundamentals.get_earnings_history(a["symbol"]),
    "get_income_statement": lambda a: fundamentals.get_income_statement(a["symbol"]),
    "get_cashflow_statement":lambda a: fundamentals.get_cashflow_statement(a["symbol"]),
    "get_ipo_data":         lambda a: fundamentals.get_ipo_data(a["company"]),
    "get_market_overview":  lambda a: stocks.get_market_overview(),
    "set_reminder":         lambda a: reminders.set_reminder(a["message"], a["minutes"]),
    "list_reminders":       lambda a: reminders.list_reminders(),
    "get_time":             lambda a: system.get_time(),
    "get_date":             lambda a: system.get_date(),
    "take_screenshot":      lambda a: system.take_screenshot(a.get("filename")),
    "open_application":     lambda a: system.open_application(a["app_name"]),
    "open_website":         lambda a: system.open_website(a["url"]),
    "create_file":          lambda a: system.create_file(a["filename"], a["content"]),
    "read_file":            lambda a: system.read_file(a["filename"]),
    "run_command":          lambda a: system.run_command(a["command"]),
    "show_news_map":        lambda a: maps.show_news_map(a["locations"]),
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
                max_tokens=2048,
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
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result),
                        })
                self.messages.append({"role": "user", "content": tool_results})
                continue

            return "Sorry, I hit an unexpected state. Please try again."

    def _trim(self, max_pairs: int = 25):
        if len(self.messages) > max_pairs * 2:
            self.messages = self.messages[-(max_pairs * 2):]

    def reset(self):
        self.messages = []
