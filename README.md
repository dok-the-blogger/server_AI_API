# AI API

API для сервисов: текстовые ответы через GigaChat/Grok, эмбеддинги Qwen3
Embedding 0.6B и краткие справки о новостях через DigitalOcean Inference. Python 3.11+.

## Запуск

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env
    # отредактировать .env
    python main.py

По умолчанию приложение слушает `127.0.0.1:9000`. При непустом `AI_API_API_TOKEN`
методы `/chat`, `/models`, `/embeddings` и `/summaries` требуют заголовок
`Authorization: Bearer <этот токен>`. `/health` открыт.

Для эмбеддингов заполнить `AI_API_DIGITALOCEAN_API_KEY` ключом доступа к моделям
DigitalOcean. Это ключ поставщика, отдельный от клиентского `AI_API_API_TOKEN`.
Ключи GigaChat/Grok для эмбеддингов не требуются. После изменения настроек нужен
перезапуск приложения. Без ключа DigitalOcean приложение работает, а
`/embeddings` возвращает 503 `embeddings_not_configured`.

| Настройка | Значение по умолчанию |
| --- | --- |
| `AI_API_DIGITALOCEAN_API_KEY` | Пусто; эмбеддинги отключены |
| `AI_API_EMBEDDINGS_BASE_URL` | `https://inference.do-ai.run/v1` |
| `AI_API_EMBEDDINGS_MODEL` | `qwen3-embedding-0.6b` |
| `AI_API_EMBEDDINGS_DIMENSIONS` | `1024`; ожидаемая длина вектора |
| `AI_API_EMBEDDINGS_TIMEOUT_SECONDS` | `30`; общий таймаут HTTP-вызова, максимум 300 |

`EMBEDDINGS_DIMENSIONS` проверяет ответ, но не просит поставщика сократить вектор.
Подготовка текста в этой версии предназначена для Qwen3. Смена модели или
подготовки требует проверки совместимости и обычно пересчёта сохранённых векторов.

## Проверка

    curl http://127.0.0.1:9000/health
    curl -X POST http://127.0.0.1:9000/chat \
      -H "Content-Type: application/json" \
      -d '{"message": "привет", "user_id": 123}'

Автоматические тесты без ключей и платных запросов:

    python -m pip install pytest pytest-asyncio
    python -m pytest -q

`test_wraps*.py` — ручные эксперименты GigaChat, не автоматические тесты.
`test_embeddings.py` проверяет реальный роутер и HTTP-клиент с подставным
транспортом. Живой запрос к DigitalOcean — отдельная приёмка.

## API

### GET /health
Возвращает `{"status": "ok"}`.
Проверяет работу приложения; не проверяет доступность/баланс DigitalOcean.

### POST /embeddings

Принимает один текст или пакет и возвращает результат в том же HTTP-ответе.
Ожидание поставщика асинхронное. Хранение текстов, векторный индекс, очередь
фоновых задач и интеграция с `dok.news` в этот метод не входят.

```json
{
  "input": ["Летом городские пляжи открываются для купания.", "Новая игра выйдет осенью."],
  "input_type": "document"
}
```

- `input`: непустая строка либо массив из 1–64 непустых строк.
- `input_type`: `document` (по умолчанию) либо `query`.
- Неизвестные поля, пробельные строки, неверные типы и некорректный Unicode отклоняются.

`document` передаёт текст без изменения. `query` добавляет к каждой строке:

```text
Instruct: Given a web search query, retrieve relevant passages that answer the query
Query:<исходный запрос>
```

Оба режима имеют `preprocessing_version=qwen3-retrieval-v1`. Сервис не пересказывает,
не обрезает и не разбивает текст. После добавления префикса допускается до 32768
символов на элемент и 131072 символов на пакет. Это ограничения размера запроса,
**не подсчёт токенов**. Лимит DigitalOcean для модели — 8000 токенов по каталогу
на 13 сентября 2026; отказ поставщика по входным данным возвращается как 422.
Его фактическое поведение на границе токенов требует отдельной живой проверки.
Длинные документы потребитель готовит заранее.

| Поле ответа | Содержимое |
| --- | --- |
| `object` | `list` |
| `data` | Объекты `object=embedding`, `index`, `embedding` |
| `model`, `provider` | Фактическая настроенная модель, `digitalocean` |
| `dimensions` | Проверенная длина вектора, по умолчанию 1024 |
| `input_type`, `preprocessing_version` | Режим и версия подготовки текста |
| `usage` | `prompt_tokens`, `total_tokens` из ответа поставщика |

На каждый вход возвращается один вектор, `data[i].index == i`; повторяющиеся
тексты остаются отдельными элементами. Проверяются количество, индексы, модель,
размерность, конечность координат и ненулевой вектор. Частичный/невалидный ответ
даёт 502 целиком. Координаты не перенормируются, тексты и векторы не сохраняются.

Пример клиента; `AI_SERVICE_TOKEN` содержит токен нашего API:

```python
import os
import httpx

response = httpx.post(
    "http://127.0.0.1:9000/embeddings",
    headers={"Authorization": f"Bearer {os.environ['AI_SERVICE_TOKEN']}"},
    json={"input": "Где можно купаться летом?", "input_type": "query"},
    timeout=40,
)
response.raise_for_status()
result = response.json()
print(result["model"], result["dimensions"], result["usage"])
vector = result["data"][0]["embedding"]
```

При живой приёмке проверить HTTP 200, модель `qwen3-embedding-0.6b`, размерность
1024, число результатов и `usage`. У DigitalOcean модель находится в Public
Preview; для serverless inference нужен положительный предоплаченный баланс.

| HTTP | Ошибка |
| --- | --- |
| 401 | `detail="Unauthorized"`: токен нашего API |
| 422 | Ошибка схемы FastAPI или `input_too_large` / `provider_input_rejected` |
| 429 | `provider_rate_limited` |
| 503 | `embeddings_not_configured` |
| 504 | `provider_timeout` |
| 502 | `provider_payment_required`: DigitalOcean вернул HTTP 402; проверьте и пополните prepaid-баланс Serverless Inference |
| 502 | `provider_unavailable`, `provider_authentication_failed`, `provider_error`, `invalid_provider_response` |

Ошибки поставщика возвращаются в `detail.code` и `detail.message` без сырого
ответа, ключей или исходного текста. Автоматических повторов, переключения модели
и частичного успеха нет. Таймаут не доказывает отсутствие тарификации у
поставщика; повтором управляет вызывающий сервис.

### POST /summaries

Разовая короткая сводка через DigitalOcean. Обязателен `body_text` (1–131072
символа); `title` (до4096), `body_format` (до80, default plain-text) и
`publication_date` (до64) необязательны. Поля исходника — данные, не инструкции.
`model`: `glm-5.3-flash` либо `deepseek-v4.1-flash`; без поля используется
SUMMARIES_MODEL (по умолчанию GLM). Нет автоматической подмены модели.

`preset`: `doknews-tldr-v2` — новая инструкция архива; `doknews-tldr-v1` — прежняя;
null — общая сводка. Отсутствующее поле сохраняет v1 для совместимости при
поэтапной выкладке. Новый worker doknews и MCP явно выбирают preset.
`instruction` (1–8000 символов) дополняет preset либо уточняет общую сводку.
Требования JSON и предела 900 символов сохраняются. Инструкция v2 хранится
в summary_presets.py; другие сервисы передают её имя, а не копию текста.

Ответ: `tldr`, `provider`, фактическая `model`, `preset`, `prompt_version`,
`prompt_hash` (SHA-256 фактической инструкции, включая дополнительную), `elapsed_ms`,
`usage` (prompt/completion/total tokens). JSON содержит только валидированный результат.
Provider timeout60s (max90), response64KiB,
max_completion_tokens=1024 для GLM и 2048 для DeepSeek (включая рассуждение),
reasoning_effort=none для GLM и low для DeepSeek, без retry/fallback.
DigitalOcean отклоняет none для DeepSeek; low — минимальный поддерживаемый
уровень. Уровень выбирается по фактической модели запроса, включая SUMMARIES_MODEL.
Ключи/сырой provider error не раскрываются.
Если ответ не проходит валидацию, журнал фиксирует только выбранную модель и
статический этап отказа (например, output_limit) и разрешённый тип ошибки схемы;
исходник и ответ модели не пишутся. Предел tldr остаётся 900 символов.
Авторизация прежним AI_API_API_TOKEN; ключ DigitalOcean остаётся в этом сервисе.

AI API не сохраняет сводки и не обновляет статьи. Хранение принадлежит doknews:
смена инструкции не требует массового пересчёта прежних готовых результатов.
Тесты используют настоящий router и имитацию HTTP поставщика, без платных запросов.
[Контракт DigitalOcean](https://docs.digitalocean.com/products/inference/how-to/use-chat-completions-api/)
и [каталог моделей](https://docs.digitalocean.com/products/inference/details/models/)
проверены 20 сентября 2026.

### POST /chat
Принимает JSON:
- `message` (str, обязательно) — текст сообщения
- `user_id` (int, опционально) — ID пользователя
- `profile` (str, опционально) — пресет сервиса
- `session_id` (str, опционально) — ID сессии для продолжения диалога
- `context` (dict, опционально) — доп. данные от сервиса

Возвращает JSON:
- `response` (str) — ответ
- `session_id` (str | null) — ID сессии
- `filtered` (bool) — ответ отфильтрован
- `model` (str | null) — использованный поставщик

### GET /models

Возвращает список моделей GigaChat и выбранную модель GigaChat. Не является
каталогом моделей DigitalOcean.

## Источники контракта

- [DigitalOcean Embeddings API](https://docs.digitalocean.com/reference/api/reference/embeddings/)
- [Каталог DigitalOcean](https://docs.digitalocean.com/products/inference/details/models/)
- [Подготовка запросов Qwen3](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [Тарификация DigitalOcean](https://docs.digitalocean.com/products/inference/details/pricing/)

Документация сверена 13 сентября 2026. Тариф $0.04 за миллион токенов опубликован
в разделе Knowledge Bases; применимость к отдельному API нужно подтвердить в
биллинге. Приложение возвращает токены без неподтверждённой денежной оценки.
