# AI API

Универсальный API для работы с LLM. Принимает текстовые запросы от сервисов, возвращает ответы модели.

## Статус

Заглушка. LLM-интеграция не реализована, на любой запрос возвращается `"OK"`.

## Запуск

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env
    # отредактировать .env
    python main.py

## Проверка

    curl http://127.0.0.1:9000/health
    curl -X POST http://127.0.0.1:9000/chat \
      -H "Content-Type: application/json" \
      -d '{"message": "привет", "user_id": 123}'

## API

### GET /health
Возвращает `{"status": "ok"}`.

### POST /chat
Принимает JSON:
- `message` (str, обязательно) — текст сообщения
- `user_id` (int, обязательно) — ID пользователя
- `profile` (str, опционально) — пресет сервиса
- `session_id` (str, опционально) — ID сессии для продолжения диалога
- `context` (dict, опционально) — доп. данные от сервиса

Возвращает JSON:
- `response` (str) — ответ
- `session_id` (str | null) — ID сессии
