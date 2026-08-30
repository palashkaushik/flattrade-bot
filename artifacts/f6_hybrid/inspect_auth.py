import httpx

r_js = httpx.get("https://auth.flattrade.in/js/app.38fb4d91.js")
text = r_js.text

pos = text.find("/auth/session")
while pos != -1:
    start = max(0, pos - 100)
    end = min(len(text), pos + 200)
    print("--- /auth/session SNIPPET ---")
    print(text[start:end])
    pos = text.find("/auth/session", pos + 1)
