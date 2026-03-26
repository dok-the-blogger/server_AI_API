with open("main.py", "r") as f:
    content = f.read()

import_line = "from openai import AsyncOpenAI\n"
if "from openai import" not in content:
    content = content.replace("from gigachat import GigaChat", f"from gigachat import GigaChat\n{import_line}")

xai_setup = """
    if settings.GROK_API_KEY:
        app.state.xai_client = AsyncOpenAI(
            api_key=settings.GROK_API_KEY,
            base_url="https://api.x.ai/v1",
        )
    else:
        app.state.xai_client = None

"""
content = content.replace("load_profiles()\n", f"load_profiles()\n{xai_setup}")

with open("main.py", "w") as f:
    f.write(content)
