#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Локальный агент для DeepSeek API.

Ключ берётся из переменной окружения DEEPSEEK_API_KEY.
Зависимости: pip install requests
Запуск:       python3 deepseek_agent.py
"""

import json
import os
import queue
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText

import requests

API_URL = "https://api.deepseek.com/chat/completions"
# Актуальные модели по документации DeepSeek (api-docs.deepseek.com, 2026):
# старые алиасы deepseek-chat / deepseek-reasoner отключены 2026-07-24.
MODELS = ("deepseek-v4-flash", "deepseek-v4-pro")
MAX_PARALLEL = 4          # максимум параллельных наборов параметров
TOKENS_MIN, TOKENS_MAX = 64, 65536   # у V4 max output — 384K; в UI ограничено 64K

STRATEGIES = {
    "direct":    "Прямой ответ",
    "instruct":  "Ответ с инструкцией",
    "preprompt": "Предварительный промпт",
    "experts":   "Группа экспертов",
}

# Thinking mode (api-docs.deepseek.com/guides/thinking_mode):
# переключатель {"thinking": {"type": "enabled"/"disabled"}},
# усилие {"reasoning_effort": "low"/"high"/"max"} (умолчание API: enabled + high).
# В режиме thinking temperature/top_p/penalties не поддерживаются (игнорируются),
# поэтому в UI они взаимоисключающие: thinking вкл -> усилие, выкл -> температура.
EFFORTS = ("low", "high", "max")


# --------------------------------------------------------------------------- API

def call_api(messages, *, model, temperature, max_tokens, fmt, api_key,
             thinking=True, effort="high", timeout=180):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "thinking": {"type": "enabled" if thinking else "disabled"},
        "stream": False,
    }
    if thinking:
        payload["reasoning_effort"] = effort
    else:
        payload["temperature"] = temperature
    if fmt == "json_object":
        payload["response_format"] = {"type": "json_object"}

    r = requests.post(
        API_URL,
        json=payload,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        timeout=timeout,
    )
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:600]}")
    data = r.json()
    usage = data.get("usage") or {}
    content = data["choices"][0]["message"]["content"]
    return content, usage


# ------------------------------------------------------------------- стратегии

def run_variant(cfg, task, api_key):
    """Выполняет один набор параметров. Возвращает (заголовок, текст_результата)."""
    model = cfg["model"]
    fmt = cfg["fmt"]
    common = dict(model=model, temperature=cfg["temperature"],
                  max_tokens=cfg["max_tokens"], fmt=fmt, api_key=api_key,
                  thinking=cfg["thinking"], effort=cfg["effort"])

    json_hint = ""
    if fmt == "json_object":
        # DeepSeek требует, чтобы слово "json" встречалось в сообщениях
        json_hint = "Ответь строго в формате JSON."

    def base_messages(sys_extra=""):
        parts = [p for p in (cfg["system"].strip(), sys_extra.strip(), json_hint) if p]
        msgs = []
        if parts:
            msgs.append({"role": "system", "content": "\n\n".join(parts)})
        msgs.append({"role": "user", "content": task})
        return msgs

    usage_total = {"prompt_tokens": 0, "completion_tokens": 0}

    def call(msgs):
        content, usage = call_api(msgs, **common)
        for k in usage_total:
            usage_total[k] += int(usage.get(k, 0) or 0)
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
        answer = call(base_messages(sys_extra="")[:-1] +
                      [{"role": "user", "content": built.strip()}])
        out.append(answer)

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
        # синтез мнений
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

    if cfg["thinking"]:
        mode = f"think:{cfg['effort']}"
    else:
        mode = f"T={cfg['temperature']:.2f}"
    title = (f"{model} | {STRATEGIES[strategy]} | {mode} | "
             f"{cfg['max_tokens']} tok | {fmt}")
    footer = (f"\n\n─── Токены: prompt={usage_total['prompt_tokens']}, "
              f"completion={usage_total['completion_tokens']} ───")
    return title, "\n\n".join(out) + footer


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
    widget.bind("<Control-v>",
                lambda e: (widget.event_generate("<<Paste>>"), "break")[1])
    widget.bind("<Control-V>",
                lambda e: (widget.event_generate("<<Paste>>"), "break")[1])


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
        self.model = ttk.Combobox(self, values=MODELS, state="readonly", width=20)
        self.model.set(saved.get("model", MODELS[0]))
        self.model.grid(row=0, column=1, sticky="w", padx=(4, 16))

        ttk.Label(self, text="Формат ответа:").grid(row=0, column=2, sticky="w")
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

        # --- строка 2: thinking mode / reasoning effort
        # Взаимоисключение по документации: thinking вкл -> выбор усилия,
        # thinking выкл -> температура (усилие недоступно).
        self.thinking_var = tk.BooleanVar(value=saved.get("thinking", True))
        self.thinking_cb = ttk.Checkbutton(
            self, text="Thinking mode", variable=self.thinking_var,
            command=self._toggle_thinking)
        self.thinking_cb.grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

        self.effort_label = ttk.Label(self, text="Reasoning effort:")
        self.effort_label.grid(row=2, column=2, sticky="w", pady=(6, 0))
        self.effort = ttk.Combobox(self, values=EFFORTS, state="readonly", width=8)
        self.effort.set(saved.get("effort", "high"))
        self.effort.grid(row=2, column=3, sticky="w", padx=4, pady=(6, 0))

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

        # --- строка 4: инструкция (для режима «Ответ с инструкцией»)
        self.instr_frame = ttk.LabelFrame(self, text="Инструкция (задаётся пользователем)", padding=6)
        self.instr_frame.columnconfigure(0, weight=1)
        self.instruction = tk.Text(self.instr_frame, height=3, wrap="word")
        self.instruction.grid(row=0, column=0, sticky="ew")
        enable_clipboard(self.instruction)
        if saved.get("instruction"):
            self.instruction.insert("1.0", saved["instruction"])

        # --- строка 5: таблица экспертов (для режима «Группа экспертов»)
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
        self._toggle_thinking()
        self._toggle()

    def _toggle_thinking(self):
        """thinking вкл -> доступно усилие; выкл -> доступна температура."""
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
        state_instr = "normal" if instr_on else "disabled"
        self.instruction.config(state=state_instr)
        for w in (self.exp_tree, self.exp_name, self.exp_desc,
                  *self.exp_frame.winfo_children()[-1].winfo_children()):
            try:
                w.config(state=state_instr if False else ("normal" if exp_on else "disabled"))
            except tk.TclError:
                pass
        self.instr_frame.grid() if instr_on else self.instr_frame.grid_remove()
        self.exp_frame.grid() if exp_on else self.exp_frame.grid_remove()

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

    def get_config(self):
        try:
            mt = int(self.max_tokens.get())
        except ValueError:
            mt = 4096
        mt = max(TOKENS_MIN, min(TOKENS_MAX, mt))
        try:
            # допускаем запятую как десятичный разделитель
            temp = float(self.temp_entry.get().strip().replace(",", "."))
        except ValueError:
            temp = 1.0
        temp = max(0.0, min(2.0, temp))
        return {
            "model": self.model.get(),
            "fmt": self.fmt.get(),
            "thinking": bool(self.thinking_var.get()),
            "effort": self.effort.get(),
            "max_tokens": mt,
            "temperature": temp,
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
        self.title("DeepSeek Agent — локальный клиент")
        self.geometry("1020x860")

        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        self.queue = queue.Queue()
        self.workers = 0

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
        key_ok = bool(self.api_key)
        ttk.Label(ctrl, text=("Ключ DEEPSEEK_API_KEY: найден" if key_ok
                              else "Ключ DEEPSEEK_API_KEY: НЕ найден"),
                  foreground=("green" if key_ok else "red")).pack(anchor="w")
        ttk.Label(ctrl, text="Параллельных ответов:").pack(anchor="w", pady=(8, 0))
        self.count = tk.Spinbox(ctrl, from_=1, to=MAX_PARALLEL, width=4,
                                command=self._rebuild_tabs)
        self.count.delete(0, "end")
        self.count.insert(0, "1")
        self.count.pack(anchor="w")
        self.run_btn = ttk.Button(ctrl, text="▶ Запустить", command=self.run,
                                  state=("normal" if key_ok else "disabled"))
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

        # --- панель 3: результаты (вкладки)
        bottom = ttk.LabelFrame(paned, text="Результаты", padding=2)
        paned.add(bottom, weight=1)
        self.results = ttk.Notebook(bottom)
        self.results.pack(fill="both", expand=True)

        self._rebuild_tabs()
        self.after(100, self._poll_queue)

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
            messagebox.showwarning("DeepSeek Agent", "Задача не задана.")
            return
        configs = []
        for tab_id in self.tabs.tabs():
            w = self.nametowidget(tab_id)
            if isinstance(w, VariantTab):
                configs.append(w.get_config())
        if not configs:
            return

        self.run_btn.config(state="disabled")
        self.status.config(text=f"Выполняется запросов: {len(configs)}…",
                           foreground="blue")
        self.workers = len(configs)

        def job(i, cfg):
            try:
                title, text = run_variant(cfg, task, self.api_key)
                self.queue.put((i, title, text, None))
            except Exception as e:
                self.queue.put((i, f"Ошибка", str(e), e))

        def spawn():
            with ThreadPoolExecutor(max_workers=len(configs)) as pool:
                for i, cfg in enumerate(configs):
                    pool.submit(job, i, cfg)

        import threading
        threading.Thread(target=spawn, daemon=True).start()

    def _poll_queue(self):
        try:
            while True:
                i, title, text, err = self.queue.get_nowait()
                self._add_result(i, title, text, is_error=err is not None)
                self.workers -= 1
                if self.workers <= 0:
                    self.run_btn.config(state="normal")
                    self.status.config(text="Готов.", foreground="gray")
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    # ------------------------------------------------------------- результаты

    def _add_result(self, index, title, text, is_error=False):
        frame = ttk.Frame(self.results, padding=4)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        header = ttk.Label(frame, text=title,
                           foreground=("red" if is_error else "darkgreen"))
        header.grid(row=0, column=0, sticky="w", pady=(0, 2))
        area = ScrolledText(frame, wrap="word", state="normal")
        area.grid(row=1, column=0, sticky="nsew")
        frame.rowconfigure(1, weight=1)
        area.insert("1.0", text)
        area.config(state="disabled")
        short = title if len(title) <= 46 else title[:43] + "…"
        self.results.add(frame, text=f"#{index + 1} {short}")
        self.results.select(frame)

    def _clear_results(self):
        for tab_id in self.results.tabs():
            self.results.forget(tab_id)


if __name__ == "__main__":
    AgentApp().mainloop()
