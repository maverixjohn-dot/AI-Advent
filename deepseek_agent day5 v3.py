#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Локальный агент для DeepSeek API и GigaChat API.

Переменные окружения:
  DEEPSEEK_API_KEY      — ключ DeepSeek (обязательна для моделей DeepSeek)
  GIGACHAT_CREDENTIALS  — ключ авторизации GigaChat (обязательна для GigaChat)
  GIGACHAT_SCOPE        — опционально, по умолчанию GIGACHAT_API_PERS
  GIGACHAT_VERIFY_SSL   — "true", если установлен корневой сертификат НУЦ Минцифры;
                          иначе проверка SSL отключена (сертификат Сбера самоподписанный)
  GIGACHAT_API_URL      — опционально, переопределить endpoint генерации
  GIGACHAT_AUTH_URL     — опционально, переопределить endpoint OAuth
  GIGACHAT_TIMEOUT      — опционально, таймаут чтения ответа, сек (умолчание 300)

Зависимости: pip install requests
Запуск:       python3 deepseek_agent.py

ВНИМАНИЕ: у GigaChat freemium генерация идёт в ОДНОМ потоке —
несколько параллельных наборов с моделью GigaChat получат HTTP 429.
"""

import os
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText

import requests

# ------------------------------------------------------------------ провайдеры

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
# С 17.07.2026 целевой URL GigaChat API — api.giga.chat; старый домен
# gigachat.devices.sberbank.ru выводится из эксплуатации (developers.sber.ru).
GIGA_API_URL = os.environ.get(
    "GIGACHAT_API_URL", "https://api.giga.chat/v1/chat/completions")
GIGA_AUTH_URL = os.environ.get(
    "GIGACHAT_AUTH_URL", "https://ngw.devices.sberbank.ru:9443/api/v2/oauth")
# Freemium-очередь GigaChat может держать запрос долго — таймаут и повтор.
GIGA_TIMEOUT = int(os.environ.get("GIGACHAT_TIMEOUT", "300"))

DEEPSEEK_MODELS = ("deepseek-v4-flash", "deepseek-v4-pro")
# В API нет «GigaChat-4-Ultra»: семейству Ultra соответствует ID GigaChat-3-Ultra
# (страница тарифов developers.sber.ru, обновлена 31.08.2026).
GIGACHAT_MODELS = ("GigaChat-3-Ultra",)

ALL_MODELS = DEEPSEEK_MODELS + GIGACHAT_MODELS
MODEL_PROVIDER = ({m: "deepseek" for m in DEEPSEEK_MODELS} |
                  {m: "gigachat" for m in GIGACHAT_MODELS})

MAX_PARALLEL = 4          # максимум параллельных наборов параметров
TOKENS_MIN, TOKENS_MAX = 64, 65536

STRATEGIES = {
    "direct":    "Прямой ответ",
    "instruct":  "Ответ с инструкцией",
    "preprompt": "Предварительный промпт",
    "experts":   "Группа экспертов",
}

# Thinking mode DeepSeek (api-docs.deepseek.com/guides/thinking_mode):
# thinking вкл -> reasoning_effort low/high/max; выкл -> temperature.
EFFORTS = ("low", "high", "max")
# Reasoning effort GigaChat (схема Chat официального SDK ai-forever/gigachat).
GC_EFFORTS = ("low", "medium", "high")

# Прайс DeepSeek, USD за 1M токенов (api-docs.deepseek.com/quick_start/pricing).
# Используются поля usage prompt_cache_hit/miss_tokens — стоимость с учётом кэша.
DEEPSEEK_PRICES = {
    "deepseek-v4-flash": {"in": 0.22, "in_cache": 0.007, "out": 0.66},
    "deepseek-v4-pro":   {"in": 0.66, "in_cache": 0.022, "out": 1.98},
}

GIGA_VERIFY_SSL = os.environ.get("GIGACHAT_VERIFY_SSL", "").lower() == "true"
if not GIGA_VERIFY_SSL:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ------------------------------------------------------------------- буфер обмена

def enable_clipboard(widget):
    """Контекстное меню (ПКМ) + Ctrl+V для Text/Entry.
    На Linux у tkinter нет привязки Ctrl+V по умолчанию — добавляем явно."""
    menu = tk.Menu(widget, tearoff=0)
    menu.add_command(label="Вставить",
                     command=lambda: widget.event_generate("<<Paste>>"))
    menu.add_command(label="Копировать",
                     command=lambda: widget.event_generate("<<Copy>>"))
    menu.add_command(label="Вырезать",
                     command=lambda: widget.event_generate("<<Cut>>"))
    widget.bind("<Button-3>",
                lambda e: menu.tk_popup(e.x_root, e.y_root))
    for key in ("<Control-v>", "<Control-V>"):
        widget.bind(key,
                    lambda e: (widget.event_generate("<<Paste>>"), "break")[1])


def enable_copy(widget, get_all):
    """Копирование из read-only поля: Ctrl+C (выделение), ПКМ-меню,
    «Копировать всё». На Linux Ctrl+C не привязан по умолчанию — добавляем."""
    def copy_all():
        widget.clipboard_clear()
        widget.clipboard_append(get_all())

    menu = tk.Menu(widget, tearoff=0)
    menu.add_command(label="Копировать",
                     command=lambda: widget.event_generate("<<Copy>>"))
    menu.add_command(label="Копировать всё", command=copy_all)
    widget.bind("<Button-3>",
                lambda e: menu.tk_popup(e.x_root, e.y_root))
    for key in ("<Control-c>", "<Control-C>"):
        widget.bind(key,
                    lambda e: (widget.event_generate("<<Copy>>"), "break")[1])
    return copy_all


# ------------------------------------------------------------- GigaChat OAuth

_giga_token = {"value": None, "expires_at": 0.0}
_giga_lock = threading.Lock()


def gigachat_access_token(credentials, scope):
    """Токен доступа GigaChat с кэшем (TTL 30 мин, обновляем с запасом)."""
    credentials = credentials.strip()
    try:
        credentials.encode("latin-1")
    except UnicodeEncodeError:
        raise RuntimeError(
            "GIGACHAT_CREDENTIALS содержит недопустимые символы (кириллицу?). "
            "Значение должно быть Base64-ключом авторизации из личного кабинета "
            "developers.sber.ru: только латинские буквы, цифры, '+', '/', '='. "
            "Скопируйте само значение ключа, без слова «ключ» и без кавычек.")
    with _giga_lock:
        now = time.time()
        if _giga_token["value"] and now < _giga_token["expires_at"]:
            return _giga_token["value"]
        r = requests.post(
            GIGA_AUTH_URL,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "RqUID": str(uuid.uuid4()),
                "Authorization": f"Basic {credentials}",
            },
            data={"scope": scope},
            verify=GIGA_VERIFY_SSL,
            timeout=30,
        )
        if r.status_code != 200:
            raise RuntimeError(f"GigaChat OAuth HTTP {r.status_code}: {r.text[:400]}")
        _giga_token["value"] = r.json()["access_token"]
        _giga_token["expires_at"] = now + 1500   # 25 мин из 30
        return _giga_token["value"]


# ------------------------------------------------------------------- вызовы API

def call_deepseek(messages, *, model, cfg, api_key, timeout=180):
    api_key = api_key.strip()
    try:
        api_key.encode("latin-1")
    except UnicodeEncodeError:
        raise RuntimeError(
            "DEEPSEEK_API_KEY содержит недопустимые символы (кириллицу?). "
            "Ключ должен быть ASCII-строкой вида sk-.... Скопируйте значение "
            "ключа без лишних слов, пробелов и кавычек.")
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": cfg["max_tokens"],
        "thinking": {"type": "enabled" if cfg["thinking"] else "disabled"},
        "stream": False,
    }
    if cfg["thinking"]:
        payload["reasoning_effort"] = cfg["effort"]
    else:
        payload["temperature"] = cfg["temperature"]
    if cfg["fmt"] == "json_object":
        payload["response_format"] = {"type": "json_object"}

    r = requests.post(
        DEEPSEEK_API_URL, json=payload,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"DeepSeek HTTP {r.status_code}: {r.text[:600]}")
    data = r.json()
    usage = data.get("usage") or {}

    cost = None
    p = DEEPSEEK_PRICES.get(model)
    if p and usage:
        hit = usage.get("prompt_cache_hit_tokens") or 0
        miss = usage.get("prompt_cache_miss_tokens")
        if miss is None:
            miss = max(0, (usage.get("prompt_tokens") or 0) - hit)
        cost = (hit * p["in_cache"] + miss * p["in"]
                + (usage.get("completion_tokens") or 0) * p["out"]) / 1e6
    return data["choices"][0]["message"]["content"], usage, cost


def call_gigachat(messages, *, model, cfg, credentials, scope, timeout=None):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": cfg["max_tokens"],
        "temperature": cfg["temperature"],
        "stream": False,
    }
    if cfg.get("top_p") is not None:
        payload["top_p"] = cfg["top_p"]
    if cfg.get("rep_penalty") is not None:
        payload["repetition_penalty"] = cfg["rep_penalty"]
    if cfg.get("gc_effort"):
        payload["reasoning_effort"] = cfg["gc_effort"]

    token = gigachat_access_token(credentials, scope)
    headers = {"Authorization": f"Bearer {token}",
               "Content-Type": "application/json",
               "Accept": "application/json"}
    # Freemium-очередь: при read timeout / обрыве соединения повторяем один раз
    timeout = timeout or GIGA_TIMEOUT
    last_err = None
    for _attempt in (1, 2):
        try:
            r = requests.post(GIGA_API_URL, json=payload, headers=headers,
                              verify=GIGA_VERIFY_SSL, timeout=timeout)
            break
        except (requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError) as e:
            last_err = e
    else:
        raise RuntimeError(
            f"GigaChat: соединение/таймаут после 2 попыток ({timeout} с "
            f"каждая). Проверьте VPN (попробуйте без него), снизьте max_tokens "
            f"или повторите позже — очередь freemium. Детали: {last_err}")
    if r.status_code != 200:
        raise RuntimeError(f"GigaChat HTTP {r.status_code}: {r.text[:600]}")
    data = r.json()
    usage = data.get("usage") or {}
    # Freemium: 50 млн токенов Ultra за 12 мес — стоимость 0 ₽
    return data["choices"][0]["message"]["content"], usage, 0.0


def call_model(cfg, messages, keys):
    provider = MODEL_PROVIDER[cfg["model"]]
    if provider == "deepseek":
        if not keys.get("deepseek"):
            raise RuntimeError("Не задана переменная окружения DEEPSEEK_API_KEY.")
        return call_deepseek(messages, model=cfg["model"], cfg=cfg,
                             api_key=keys["deepseek"])
    if not keys.get("gigachat"):
        raise RuntimeError("Не задана переменная окружения GIGACHAT_CREDENTIALS.")
    return call_gigachat(messages, model=cfg["model"], cfg=cfg,
                         credentials=keys["gigachat"],
                         scope=keys.get("gigachat_scope") or "GIGACHAT_API_PERS")


# ------------------------------------------------------------------- стратегии

def run_variant(cfg, task, keys):
    """Выполняет один набор параметров.
    Возвращает (заголовок, текст, метрики)."""
    started = time.perf_counter()
    provider = MODEL_PROVIDER[cfg["model"]]
    model = cfg["model"]

    json_hint = ""
    if provider == "deepseek" and cfg["fmt"] == "json_object":
        # DeepSeek требует, чтобы слово "json" встречалось в сообщениях
        json_hint = "Ответь строго в формате JSON."

    def base_messages(sys_extra=""):
        parts = [p for p in (cfg["system"].strip(), sys_extra.strip(), json_hint) if p]
        msgs = []
        if parts:
            msgs.append({"role": "system", "content": "\n\n".join(parts)})
        msgs.append({"role": "user", "content": task})
        return msgs

    totals = {"prompt": 0, "completion": 0, "cost": 0.0}

    def call(msgs):
        content, usage, cost = call_model(cfg, msgs, keys)
        totals["prompt"] += int(usage.get("prompt_tokens", 0) or 0)
        totals["completion"] += int(usage.get("completion_tokens", 0) or 0)
        if cost:
            totals["cost"] += cost
        return content

    strategy = cfg["strategy"]
    out = []

    if strategy == "direct":
        out.append(call(base_messages()))

    elif strategy == "instruct":
        instruction = cfg["instruction"].strip()
        if not instruction:
            raise ValueError("Стратегия «Ответ с инструкцией»: инструкция не задана.")
        out.append("═══ Инструкция ═══\n" + instruction)
        out.append("═══ Ответ ═══\n" + call(base_messages(sys_extra=instruction)))

    elif strategy == "preprompt":
        meta = (
            "Ты — инженер промптов. Составь один оптимальный промпт, который "
            "позволит языковой модели наилучшим образом решить задачу ниже. "
            "Выдай ТОЛЬКО сам промпт, без пояснений.\n\nЗадача:\n" + task
        )
        built = call([{"role": "system", "content": json_hint or "Ты — инженер промптов."},
                      {"role": "user", "content": meta}])
        # сгенерированный промпт — промежуточный шаг, в результат не выводится
        out.append(call(base_messages()[:-1] +
                        [{"role": "user", "content": built.strip()}]))

    elif strategy == "experts":
        experts = [(n.strip(), d.strip()) for n, d in cfg["experts"] if n.strip()]
        if not experts:
            raise ValueError("Стратегия «Группа экспертов»: не задано ни одной роли.")
        answers = []
        for name, desc in experts:
            sys_prompt = desc or f"Ты — эксперт: {name}."
            if json_hint:
                sys_prompt += "\n" + json_hint
            ans = call([{"role": "system", "content": sys_prompt},
                        {"role": "user", "content": task}])
            answers.append((name, ans))
            out.append(f"═══ Эксперт: {name} ═══\n{ans}")
        joined = "\n\n".join(f"[{name}]:\n{ans}" for name, ans in answers)
        synth_msgs = base_messages(sys_extra=(
            "Ты — модератор группы экспертов. Ниже — их ответы на задачу. "
            "Дай итоговый согласованный ответ: укажи согласие, разногласия "
            "и финальный вывод."))
        synth_msgs.append({"role": "user", "content":
                           f"Задача:\n{task}\n\nОтветы экспертов:\n{joined}"})
        out.append("═══ Синтез (модератор) ═══\n" + call(synth_msgs))

    else:
        raise ValueError(f"Неизвестная стратегия: {strategy}")

    elapsed = time.perf_counter() - started

    # --- строка параметров для заголовков и сводки
    if provider == "deepseek":
        mode = (f"think:{cfg['effort']}" if cfg["thinking"]
                else f"T={cfg['temperature']:.2f}")
        cost_str = f"${totals['cost']:.6f}"
    else:
        parts = [f"T={cfg['temperature']:.2f}"]
        if cfg.get("top_p") is not None:
            parts.append(f"top_p={cfg['top_p']}")
        if cfg.get("rep_penalty") is not None:
            parts.append(f"rp={cfg['rep_penalty']}")
        if cfg.get("gc_effort"):
            parts.append(f"effort={cfg['gc_effort']}")
        mode = " ".join(parts)
        cost_str = "0 ₽ (freemium)"

    title = (f"{model} | {STRATEGIES[strategy]} | {mode} | "
             f"{cfg['max_tokens']} tok")
    if provider == "deepseek":
        title += f" | {cfg['fmt']}"

    metrics = {
        "params": mode,
        "elapsed": elapsed,
        "prompt": totals["prompt"],
        "completion": totals["completion"],
        "cost_str": cost_str,
    }
    footer = (f"\n\n─── Время: {elapsed:.1f} с | "
              f"Токены: prompt={totals['prompt']}, completion={totals['completion']} | "
              f"Стоимость: {cost_str} ───")
    return title, "\n\n".join(out) + footer, metrics


# ------------------------------------------------------------- вкладка варианта

class VariantTab(ttk.Frame):
    """Один набор параметров = один параллельный ответ."""

    def __init__(self, master, saved=None):
        super().__init__(master, padding=8)
        saved = saved or {}
        self.columnconfigure(1, weight=1)
        self.columnconfigure(3, weight=1)

        # --- строка 0: модель / формат ответа
        ttk.Label(self, text="Модель:").grid(row=0, column=0, sticky="w")
        self.model = ttk.Combobox(self, values=ALL_MODELS, state="readonly",
                                  width=22)
        self.model.set(saved.get("model", ALL_MODELS[0]))
        self.model.grid(row=0, column=1, sticky="w", padx=(4, 16))
        self.model.bind("<<ComboboxSelected>>",
                        lambda e: self._toggle_provider())

        self.fmt_label = ttk.Label(self, text="Формат ответа:")
        self.fmt_label.grid(row=0, column=2, sticky="w")
        self.fmt = ttk.Combobox(self, values=("text", "json_object"),
                                state="readonly", width=12)
        self.fmt.set(saved.get("fmt", "text"))
        self.fmt.grid(row=0, column=3, sticky="w", padx=4)

        # --- строка 1: длина ответа / температура
        ttk.Label(self, text="Длина ответа (max_tokens):").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.max_tokens = tk.Spinbox(self, from_=TOKENS_MIN, to=TOKENS_MAX,
                                     increment=256, width=8)
        self.max_tokens.delete(0, "end")
        self.max_tokens.insert(0, str(saved.get("max_tokens", 4096)))
        self.max_tokens.grid(row=1, column=1, sticky="w", padx=(4, 16), pady=(6, 0))

        self.temp_label = ttk.Label(self, text="Температура (0–2):")
        self.temp_label.grid(row=1, column=2, sticky="w", pady=(6, 0))
        self.temp_entry = ttk.Entry(self, width=8)
        self.temp_entry.insert(0, str(saved.get("temperature", 1.0)))
        self.temp_entry.grid(row=1, column=3, sticky="w", padx=4, pady=(6, 0))
        enable_clipboard(self.temp_entry)

        # --- строка 2: параметры, специфичные для провайдера (взаимозаменяемые)
        # DeepSeek: thinking вкл -> усилие; выкл -> температура.
        self.ds_frame = ttk.Frame(self)
        self.thinking_var = tk.BooleanVar(value=saved.get("thinking", True))
        self.thinking_cb = ttk.Checkbutton(
            self.ds_frame, text="Thinking mode", variable=self.thinking_var,
            command=self._toggle_thinking)
        self.thinking_cb.grid(row=0, column=0, sticky="w")
        self.effort_label = ttk.Label(self.ds_frame, text="Reasoning effort:")
        self.effort_label.grid(row=0, column=1, sticky="w", padx=(24, 0))
        self.effort = ttk.Combobox(self.ds_frame, values=EFFORTS,
                                   state="readonly", width=8)
        self.effort.set(saved.get("effort", "high"))
        self.effort.grid(row=0, column=2, sticky="w", padx=4)

        # GigaChat: top_p / repetition_penalty / reasoning_effort (low/medium/high).
        self.gc_frame = ttk.Frame(self)
        ttk.Label(self.gc_frame, text="top_p:").grid(row=0, column=0, sticky="w")
        self.top_p = ttk.Entry(self.gc_frame, width=6)
        if saved.get("top_p") is not None:
            self.top_p.insert(0, str(saved["top_p"]))
        self.top_p.grid(row=0, column=1, sticky="w", padx=(4, 12))
        enable_clipboard(self.top_p)

        ttk.Label(self.gc_frame, text="Штраф повторов:").grid(row=0, column=2, sticky="w")
        self.rep_penalty = ttk.Entry(self.gc_frame, width=6)
        if saved.get("rep_penalty") is not None:
            self.rep_penalty.insert(0, str(saved["rep_penalty"]))
        self.rep_penalty.grid(row=0, column=3, sticky="w", padx=(4, 12))
        enable_clipboard(self.rep_penalty)

        ttk.Label(self.gc_frame, text="Reasoning effort:").grid(row=0, column=4, sticky="w")
        self.gc_effort = ttk.Combobox(self.gc_frame,
                                      values=("по умолчанию",) + GC_EFFORTS,
                                      state="readonly", width=12)
        self.gc_effort.set(saved.get("gc_effort") or "по умолчанию")
        self.gc_effort.grid(row=0, column=5, sticky="w", padx=4)

        self.ds_frame.grid(row=2, column=0, columnspan=4, sticky="w", pady=(6, 0))
        self.gc_frame.grid(row=2, column=0, columnspan=4, sticky="w", pady=(6, 0))

        # --- строка 3: системный промпт
        ttk.Label(self, text="Системный промпт:").grid(row=3, column=0, sticky="nw", pady=(6, 0))
        self.system = tk.Text(self, height=2, wrap="word")
        self.system.grid(row=3, column=1, columnspan=3, sticky="ew", padx=4, pady=(6, 0))
        enable_clipboard(self.system)
        if saved.get("system"):
            self.system.insert("1.0", saved["system"])

        # --- строка 4: стратегия
        s_frame = ttk.LabelFrame(self, text="Режим ответа", padding=6)
        s_frame.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        self.strategy = tk.StringVar(value=saved.get("strategy", "direct"))
        for i, (key, label) in enumerate(STRATEGIES.items()):
            ttk.Radiobutton(s_frame, text=label, value=key,
                            variable=self.strategy,
                            command=self._toggle).grid(row=0, column=i, sticky="w", padx=(0, 14))

        # --- строка 5: инструкция (для режима «Ответ с инструкцией»)
        self.instr_frame = ttk.LabelFrame(self, text="Инструкция (задаётся пользователем)", padding=6)
        self.instr_frame.columnconfigure(0, weight=1)
        self.instruction = tk.Text(self.instr_frame, height=3, wrap="word")
        self.instruction.grid(row=0, column=0, sticky="ew")
        enable_clipboard(self.instruction)
        if saved.get("instruction"):
            self.instruction.insert("1.0", saved["instruction"])

        # --- строка 6: таблица экспертов (для режима «Группа экспертов»)
        self.exp_frame = ttk.LabelFrame(self, text="Роли экспертов", padding=6)
        self.exp_frame.columnconfigure(0, weight=1)
        self.exp_tree = ttk.Treeview(self.exp_frame, columns=("name", "desc"),
                                     show="headings", height=3)
        self.exp_tree.heading("name", text="Роль")
        self.exp_tree.heading("desc", text="Описание / системный промпт роли")
        self.exp_tree.column("name", width=140, stretch=False)
        self.exp_tree.column("desc", width=420)
        self.exp_tree.grid(row=0, column=0, rowspan=3, sticky="ew", padx=(0, 6))
        for name, desc in saved.get("experts", [("Аналитик", "Анализируй задачу структурно и по фактам."),
                                                ("Критик", "Найди слабые места и риски в решении.")]):
            self.exp_tree.insert("", "end", values=(name, desc))

        self.exp_name = ttk.Entry(self.exp_frame, width=18)
        self.exp_name.insert(0, "Роль")
        self.exp_name.grid(row=0, column=1, sticky="ew")
        enable_clipboard(self.exp_name)
        self.exp_desc = ttk.Entry(self.exp_frame, width=30)
        self.exp_desc.insert(0, "Описание роли")
        self.exp_desc.grid(row=1, column=1, sticky="ew", pady=2)
        enable_clipboard(self.exp_desc)
        btns = ttk.Frame(self.exp_frame)
        btns.grid(row=2, column=1, sticky="w")
        ttk.Button(btns, text="+ Добавить", command=self._add_expert).pack(side="left")
        ttk.Button(btns, text="− Удалить", command=self._del_expert).pack(side="left", padx=4)

        self.instr_frame.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        self.exp_frame.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        self._toggle_provider()
        self._toggle()

    # ---------------------------------------------------------- переключатели

    def _provider(self):
        return MODEL_PROVIDER[self.model.get()]

    def _toggle_provider(self):
        """Показывает только параметры, релевантные выбранному провайдеру."""
        if self._provider() == "gigachat":
            self.ds_frame.grid_remove()
            self.gc_frame.grid()
            # у GigaChat нет thinking-переключателя и простого json_object
            self.fmt.set("text")
            self.fmt.config(state="disabled")
            self.fmt_label.config(state="disabled")
            self.temp_entry.config(state="normal")
            self.temp_label.config(state="normal")
        else:
            self.gc_frame.grid_remove()
            self.ds_frame.grid()
            self.fmt.config(state="readonly")
            self.fmt_label.config(state="normal")
            self._toggle_thinking()

    def _toggle_thinking(self):
        """DeepSeek: thinking вкл -> доступно усилие; выкл -> температура."""
        on = self.thinking_var.get()
        self.effort.config(state="readonly" if on else "disabled")
        self.effort_label.config(state="normal" if on else "disabled")
        temp_state = "disabled" if on else "normal"
        self.temp_entry.config(state=temp_state)
        self.temp_label.config(state=temp_state)

    def _toggle(self):
        s = self.strategy.get()
        instr_on = (s == "instruct")
        exp_on = (s == "experts")
        self.instruction.config(state="normal" if instr_on else "disabled")
        for w in (self.exp_tree, self.exp_name, self.exp_desc,
                  *self.exp_frame.winfo_children()[-1].winfo_children()):
            try:
                w.config(state="normal" if exp_on else "disabled")
            except tk.TclError:
                pass
        self.instr_frame.grid() if instr_on else self.instr_frame.grid_remove()
        self.exp_frame.grid() if exp_on else self.exp_frame.grid_remove()

    # ---------------------------------------------------------- эксперты

    def _add_expert(self):
        name, desc = self.exp_name.get().strip(), self.exp_desc.get().strip()
        if not name or name == "Роль":
            return
        self.exp_tree.insert("", "end", values=(name, desc))
        self.exp_name.delete(0, "end")
        self.exp_desc.delete(0, "end")

    def _del_expert(self):
        for item in self.exp_tree.selection():
            self.exp_tree.delete(item)

    # ---------------------------------------------------------- конфигурация

    @staticmethod
    def _float_or_none(entry, lo=None, hi=None):
        raw = entry.get().strip().replace(",", ".")
        if not raw:
            return None
        try:
            v = float(raw)
        except ValueError:
            return None
        if lo is not None:
            v = max(lo, v)
        if hi is not None:
            v = min(hi, v)
        return v

    def get_config(self):
        try:
            mt = int(self.max_tokens.get())
        except ValueError:
            mt = 4096
        mt = max(TOKENS_MIN, min(TOKENS_MAX, mt))
        try:
            temp = float(self.temp_entry.get().strip().replace(",", "."))
        except ValueError:
            temp = 1.0
        temp = max(0.0, min(2.0, temp))
        gc_effort = self.gc_effort.get()
        return {
            "model": self.model.get(),
            "fmt": self.fmt.get(),
            "thinking": bool(self.thinking_var.get()),
            "effort": self.effort.get(),
            "max_tokens": mt,
            "temperature": temp,
            "top_p": self._float_or_none(self.top_p, 0.0, 1.0),
            "rep_penalty": self._float_or_none(self.rep_penalty),
            "gc_effort": None if gc_effort == "по умолчанию" else gc_effort,
            "system": self.system.get("1.0", "end").strip(),
            "strategy": self.strategy.get(),
            "instruction": self.instruction.get("1.0", "end").strip(),
            "experts": [self.exp_tree.item(i, "values")
                        for i in self.exp_tree.get_children()],
        }


# ------------------------------------------------------------------ главное окно

class AgentApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LLM Agent — DeepSeek + GigaChat")
        self.geometry("1060x880")

        self.keys = {
            "deepseek": os.environ.get("DEEPSEEK_API_KEY", "").strip(),
            "gigachat": os.environ.get("GIGACHAT_CREDENTIALS", "").strip(),
            "gigachat_scope": os.environ.get("GIGACHAT_SCOPE", "").strip(),
        }
        self.queue = queue.Queue()
        self.workers = 0
        self.run_counter = 0

        paned = ttk.PanedWindow(self, orient="vertical")
        paned.pack(fill="both", expand=True, padx=6, pady=6)

        # --- панель 1: задача + глобальные параметры
        top = ttk.Frame(paned, padding=4)
        top.columnconfigure(0, weight=1)
        paned.add(top, weight=0)

        t_head = ttk.Frame(top)
        t_head.grid(row=0, column=0, sticky="ew")
        ttk.Label(t_head, text="Задача:").pack(side="left")
        ttk.Button(t_head, text="Вставить из буфера",
                   command=lambda: self._paste_into(self.task)).pack(side="right")
        self.task = ScrolledText(top, height=4, wrap="word")
        self.task.grid(row=1, column=0, sticky="ew", padx=(0, 8))
        enable_clipboard(self.task)

        ctrl = ttk.Frame(top)
        ctrl.grid(row=0, column=1, rowspan=2, sticky="ns")
        for label, present in (
            ("DEEPSEEK_API_KEY", bool(self.keys["deepseek"])),
            ("GIGACHAT_CREDENTIALS", bool(self.keys["gigachat"])),
        ):
            ttk.Label(ctrl, text=f"{label}: {'найден' if present else 'НЕ найден'}",
                      foreground=("green" if present else "red")).pack(anchor="w")
        ttk.Label(ctrl, text="Параллельных ответов:").pack(anchor="w", pady=(8, 0))
        self.count = tk.Spinbox(ctrl, from_=1, to=MAX_PARALLEL, width=4,
                                command=self._rebuild_tabs)
        self.count.delete(0, "end")
        self.count.insert(0, "1")
        self.count.pack(anchor="w")
        any_key = self.keys["deepseek"] or self.keys["gigachat"]
        self.run_btn = ttk.Button(ctrl, text="▶ Запустить", command=self.run,
                                  state=("normal" if any_key else "disabled"))
        self.run_btn.pack(anchor="w", pady=(10, 2), fill="x")
        ttk.Button(ctrl, text="Очистить результаты",
                   command=self._clear_results).pack(anchor="w", fill="x")
        self.status = ttk.Label(ctrl, text="Готов.", foreground="gray")
        self.status.pack(anchor="w", pady=(10, 0))

        # --- панель 2: наборы параметров (вкладки)
        mid = ttk.LabelFrame(paned, text="Наборы параметров (каждый = один ответ)", padding=2)
        paned.add(mid, weight=1)
        self.tabs = ttk.Notebook(mid)
        self.tabs.pack(fill="both", expand=True)

        # --- панель 3: результаты (вкладки) + сводная таблица
        bottom = ttk.LabelFrame(paned, text="Результаты", padding=2)
        paned.add(bottom, weight=1)
        self.results = ttk.Notebook(bottom)
        self.results.pack(fill="both", expand=True)
        self._build_summary_tab()

        self._rebuild_tabs()
        self.after(100, self._poll_queue)

    # ------------------------------------------------------------- сводка

    def _build_summary_tab(self):
        frame = ttk.Frame(self.results, padding=4)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        cols = ("run", "variant", "model", "strategy", "params",
                "elapsed", "prompt", "completion", "cost")
        headers = ("Запуск", "Набор", "Модель", "Режим", "Параметры",
                   "Время, с", "Prompt", "Compl.", "Стоимость")
        widths = (60, 50, 150, 130, 150, 70, 70, 70, 110)
        self.summary = ttk.Treeview(frame, columns=cols, show="headings")
        for c, h, w in zip(cols, headers, widths):
            self.summary.heading(c, text=h)
            self.summary.column(c, width=w, stretch=(c in ("model", "params")))
        vsb = ttk.Scrollbar(frame, orient="vertical",
                            command=self.summary.yview)
        self.summary.configure(yscrollcommand=vsb.set)
        self.summary.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        def copy_rows():
            sel = self.summary.selection()
            if not sel:
                return
            text = "\n".join(
                "\t".join(str(v) for v in self.summary.item(i, "values"))
                for i in sel)
            self.summary.clipboard_clear()
            self.summary.clipboard_append(text)

        self._copy_summary_rows = copy_rows
        menu = tk.Menu(self.summary, tearoff=0)
        menu.add_command(label="Копировать строку", command=copy_rows)
        self.summary.bind("<Button-3>",
                          lambda e: menu.tk_popup(e.x_root, e.y_root))
        for key in ("<Control-c>", "<Control-C>", "<<Copy>>"):
            self.summary.bind(key, lambda e: (copy_rows(), "break")[1])
        self.summary_tab = frame
        self.results.add(frame, text=" Сводка ")

    def _summary_add(self, run_no, index, title_model, strategy, metrics):
        self.summary.insert("", "end", values=(
            run_no, f"#{index + 1}", title_model, strategy, metrics["params"],
            f"{metrics['elapsed']:.1f}", metrics["prompt"],
            metrics["completion"], metrics["cost_str"],
        ))
        self.summary.yview_moveto(1.0)

    # ------------------------------------------------------------- вкладки

    def _paste_into(self, widget):
        try:
            data = self.clipboard_get()
        except tk.TclError:
            self.status.config(text="Буфер обмена пуст или не содержит текста.",
                               foreground="red")
            return
        widget.insert("insert", data)
        widget.see("insert")

    def _rebuild_tabs(self):
        saved = []
        for tab_id in self.tabs.tabs():
            w = self.nametowidget(tab_id)
            if isinstance(w, VariantTab):
                saved.append(w.get_config())
        for tab_id in self.tabs.tabs():
            self.tabs.forget(tab_id)
        try:
            n = max(1, min(MAX_PARALLEL, int(self.count.get())))
        except ValueError:
            n = 1
        for i in range(n):
            cfg = saved[i] if i < len(saved) else None
            tab = VariantTab(self.tabs, saved=cfg)
            self.tabs.add(tab, text=f" Ответ {i + 1} ")

    # ------------------------------------------------------------- запуск

    def run(self):
        task = self.task.get("1.0", "end").strip()
        if not task:
            messagebox.showwarning("LLM Agent", "Задача не задана.")
            return
        configs = []
        for tab_id in self.tabs.tabs():
            w = self.nametowidget(tab_id)
            if isinstance(w, VariantTab):
                configs.append(w.get_config())
        if not configs:
            return

        self.run_counter += 1
        run_no = self.run_counter
        self.run_btn.config(state="disabled")
        self.status.config(text=f"Выполняется запросов: {len(configs)}…",
                           foreground="blue")
        self.workers = len(configs)

        def job(i, cfg):
            try:
                title, text, metrics = run_variant(cfg, task, self.keys)
                self.queue.put((i, run_no, cfg, title, text, metrics, None))
            except Exception as e:
                self.queue.put((i, run_no, cfg, "Ошибка", str(e), None, e))

        def spawn():
            with ThreadPoolExecutor(max_workers=len(configs)) as pool:
                for i, cfg in enumerate(configs):
                    pool.submit(job, i, cfg)

        threading.Thread(target=spawn, daemon=True).start()

    def _poll_queue(self):
        try:
            while True:
                i, run_no, cfg, title, text, metrics, err = self.queue.get_nowait()
                self._add_result(i, run_no, cfg, title, text, metrics,
                                 is_error=err is not None)
                self.workers -= 1
                if self.workers <= 0:
                    self.run_btn.config(state="normal")
                    self.status.config(text="Готов.", foreground="gray")
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    # ------------------------------------------------------------- результаты

    def _add_result(self, index, run_no, cfg, title, text, metrics,
                    is_error=False):
        frame = ttk.Frame(self.results, padding=4)
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)
        area = ScrolledText(frame, wrap="word", state="normal")
        area.insert("1.0", text)
        area.config(state="disabled")
        copy_all = enable_copy(area, lambda: area.get("1.0", "end-1c"))

        head = ttk.Frame(frame)
        head.grid(row=0, column=0, sticky="ew", pady=(0, 2))
        header = ttk.Label(head, text=title,
                           foreground=("red" if is_error else "darkgreen"))
        header.pack(side="left")
        ttk.Button(head, text="Копировать всё",
                   command=copy_all).pack(side="right")
        area.grid(row=1, column=0, sticky="nsew")
        short = title if len(title) <= 46 else title[:43] + "…"
        self.results.add(frame, text=f"#{index + 1} {short}")
        self.results.select(frame)

        if metrics and not is_error:
            self._summary_add(run_no, index, cfg["model"],
                              STRATEGIES[cfg["strategy"]], metrics)

    def _clear_results(self):
        for tab_id in self.results.tabs():
            if self.nametowidget(tab_id) is not self.summary_tab:
                self.results.forget(tab_id)
        for item in self.summary.get_children():
            self.summary.delete(item)


if __name__ == "__main__":
    AgentApp().mainloop()
