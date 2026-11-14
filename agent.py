"""Minimal tool-using agent loop from "Building Agentic AI Systems: A Practical Guide".

Usage:
    pip install anthropic
    export ANTHROPIC_API_KEY=...
    python agent.py
"""
import json
import anthropic

client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY
MODEL = "claude-sonnet-5-5"

# 1. Tools: a name, a description the model reads, and a JSON Schema for inputs
TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current temperature in Celsius for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
    {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression such as '3 * (4 + 5)'.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
]


# 2. Tool execution: plain code, run by YOUR program, never by the model
def run_tool(name: str, args: dict) -> str:
    if name == "get_weather":
        demo_data = {"Hyderabad": 31, "London": 9}   # stub: replace with a real API
        return json.dumps({"city": args["city"], "temp_c": demo_data.get(args["city"], 20)})
    if name == "calculator":
        expr = args["expression"]
        if not set(expr) <= set("0123456789+-*/(). "):
            return "error: unsupported characters"
        return str(eval(expr, {"__builtins__": {}}))   # demo only; use a real parser in production
    return f"error: unknown tool {name}"


# 3. The control loop
def run_agent(task: str, max_steps: int = 10) -> str:
    messages = [{"role": "user", "content": task}]
    for _ in range(max_steps):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system="You are a careful assistant. Use tools when they help.",
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":          # model is done
            return "".join(b.text for b in response.content if b.type == "text")

        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = run_tool(block.name, block.input)
                results.append({"type": "tool_result",
                                "tool_use_id": block.id,
                                "content": output})
        messages.append({"role": "user", "content": results})
    return "Stopped: step limit reached."


if __name__ == "__main__":
    print(run_agent("How much warmer is Hyderabad than London right now, in Fahrenheit?"))
