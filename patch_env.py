with open(".env.example", "r") as f:
    content = f.read()
content += "AI_API_GROK_API_KEY=\nAI_API_GROK_MODEL=grok-4-1-fast-non-reasoning\n"
with open(".env.example", "w") as f:
    f.write(content)
