"""
GUI-агент подбора домашнего питомца (DeepSeek API, tkinter).

Авторизация: ключ из переменной окружения DEEPSEEK_API_KEY.

Сценарий:
1. Экран настроек: модель, длина ответа, максимум вопросов, число вариантов,
   формат ответа (JSON / свободный).
   Любой параметр можно не задавать — тогда модель решает сама.
2. Большая кнопка «А теперь мы подберем тебе питомца».
3. Экран опроса: агент задаёт вопросы по одному, пользователь отвечает.
4. Экран результата: рекомендация в выбранном формате.

Зависимости: pip install openai
Запуск:      python pet_picker_agent.py
"""

import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

from openai import OpenAI

BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"

NOT_SET = "не задано"

MODELS = [NOT_SET, "deepseek-v4-flash", "deepseek-v4-pro", "deepseek-v4-flash-vision-exp"]
LENGTHS = [NOT_SET, "краткий", "средний", "подробный"]
MAX_QUESTIONS = [NOT_SET, "3", "5", "7", "10"]
VARIANTS = [NOT_SET, "1", "2", "3", "5"]
FORMATS = [NOT_SET, "JSON", "свободная форма"]

FINISH_MARKER = "[ОПРОС_ЗАВЕРШЁН]"


# ---------------------------------------------------------------------------
# Сборка промптов из параметров
# ---------------------------------------------------------------------------

def build_interview_prompt(settings: dict) -> str:
    prompt = (
        "Ты — эксперт по домашним питомцам. Твоя задача — подобрать "
        "пользователю оптимального питомца. Задавай вопросы ПО ОДНОМУ, "
        "кратко и по делу: про жилищные условия, бюджет, свободное время, "
        "аллергии, опыт, предпочтения.\n"
    )
    if settings["max_questions"] != NOT_SET:
        prompt += (
            f"Задай НЕ БОЛЕЕ {settings['max_questions']} вопросов, затем заверши опрос.\n"
        )
    else:
        prompt += "Сам реши, сколько вопросов достаточно (обычно 3–7).\n"
    prompt += (
        f"Когда информации достаточно, выведи отдельной строкой маркер {FINISH_MARKER} "
        "и больше не задавай вопросов.\n"
        "Начни с первого вопроса прямо сейчас."
    )
    return prompt


def build_result_prompt(settings: dict) -> tuple[str, bool]:
    """Возвращает (system_prompt, use_json_format)."""
    fmt = settings["format"]
    use_json = fmt == "JSON"  # при NOT_SET — свободная форма, но с той же структурой данных

    variants = settings["variants"]
    if variants != NOT_SET:
        n_variants = f"ровно {variants} вариант(а/ов)"
    else:
        n_variants = "2–3 варианта на твоё усмотрение"

    length = settings["length"]
    length_hint = {
        "краткий": "Отвечай очень кратко: 1–2 предложения на вариант.",
        "средний": "Отвечай умеренно подробно: 3–5 предложений на вариант.",
        "подробный": "Отвечай подробно: обоснуй каждый вариант, укажи плюсы и минусы.",
    }.get(length, "")

    base = (
        "Ты — эксперт по домашним питомцам. На основе диалога ниже сформируй "
        f"итоговую рекомендацию: {n_variants} питомцев. "
        "Для КАЖДОГО варианта обязательно укажи: название (вид/порода), "
        "стоимость покупки в рублях, стоимость содержания в рублях в месяц, "
        "интервал средней продолжительности жизни в годах. "
        "Цены — ориентировочные, актуальные для России. "
        f"{length_hint}\n"
    )

    if use_json:
        base += (
            'Ответь СТРОГО в формате JSON по схеме:\n'
            '{\n'
            '  "message": "общий вывод",\n'
            '  "pets": [\n'
            '    {\n'
            '      "name": "вид/порода",\n'
            '      "purchase_cost_rub": "стоимость покупки",\n'
            '      "maintenance_cost_rub_per_month": "содержание в месяц",\n'
            '      "lifespan_years": "интервал жизни, например \\"10-15\\"",\n'
            '      "why": "почему подходит пользователю"\n'
            '    }\n'
            '  ]\n'
            '}\n'
            "Только JSON, без текста вне JSON."
        )
    else:
        base += "Формат — связный текст с нумерованным списком вариантов."
    return base, use_json


# ---------------------------------------------------------------------------
# Приложение
# ---------------------------------------------------------------------------

class PetPickerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Подбор питомца")
        self.geometry("720x620")

        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            messagebox.showerror(
                "Нет ключа",
                "Переменная окружения DEEPSEEK_API_KEY не задана.\n"
                "Установите её и перезапустите приложение.",
            )
            self.destroy()
            return
        self.client = OpenAI(api_key=api_key, base_url=BASE_URL)

        self.container = ttk.Frame(self)
        self.container.pack(fill=tk.BOTH, expand=True)

        self.frames = {}
        for cls in (SettingsFrame, InterviewFrame, ResultFrame):
            frame = cls(self.container, self)
            self.frames[cls] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.show(SettingsFrame)

    def show(self, cls):
        self.frames[cls].tkraise()

    # --- фоновый вызов API ---
    def api_call(self, model, messages, use_json, on_ok):
        def worker():
            try:
                kwargs = dict(model=model, messages=messages, temperature=0.7)
                if use_json:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = self.client.chat.completions.create(**kwargs)
                self.after(0, on_ok, resp.choices[0].message.content)
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror(
                    "Ошибка API", f"{type(exc).__name__}: {exc}"))

        threading.Thread(target=worker, daemon=True).start()


# ---------------------------------------------------------------------------
# Экран 1: настройки + большая кнопка
# ---------------------------------------------------------------------------

class SettingsFrame(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, padding=20)
        self.app = app

        ttk.Label(self, text="Параметры подбора (можно не задавать)",
                  font=("TkDefaultFont", 13, "bold")).pack(anchor="w", pady=(0, 12))

        self.vars = {}
        rows = [
            ("Тип модели", "model", MODELS),
            ("Длина ответа", "length", LENGTHS),
            ("Максимальное число вопросов", "max_questions", MAX_QUESTIONS),
            ("Число вариантов питомцев", "variants", VARIANTS),
            ("Формат ответа", "format", FORMATS),
        ]
        form = ttk.Frame(self)
        form.pack(fill=tk.X)
        for i, (label, key, values) in enumerate(rows):
            ttk.Label(form, text=label + ":").grid(row=i, column=0, sticky="w", pady=4)
            var = tk.StringVar(value=NOT_SET)
            ttk.Combobox(form, textvariable=var, values=values,
                         state="readonly", width=32).grid(row=i, column=1, sticky="w", padx=10, pady=4)
            self.vars[key] = var

        ttk.Label(self,
                  text="Параметры «не задано» модель определяет по своему усмотрению.",
                  foreground="gray").pack(anchor="w", pady=(8, 0))

        # большая кнопка
        btn = tk.Button(
            self,
            text="А теперь мы подберем тебе питомца!",
            font=("TkDefaultFont", 16, "bold"),
            bg="#2e7d32", fg="white",
            activebackground="#1b5e20", activeforeground="white",
            relief=tk.RAISED, bd=4, cursor="hand2",
            command=self.start,
        )
        btn.pack(fill=tk.X, ipady=18, pady=(24, 0))

    def collect(self) -> dict:
        return {
            "model": self.vars["model"].get(),
            "length": self.vars["length"].get(),
            "max_questions": self.vars["max_questions"].get(),
            "variants": self.vars["variants"].get(),
            "format": self.vars["format"].get(),
        }

    def start(self):
        settings = self.collect()
        interview = self.app.frames[InterviewFrame]
        interview.configure_session(settings)
        self.app.show(InterviewFrame)


# ---------------------------------------------------------------------------
# Экран 2: последовательный опрос
# ---------------------------------------------------------------------------

class InterviewFrame(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, padding=15)
        self.app = app
        self.settings = None
        self.messages = []
        self.questions_asked = 0
        self.max_q = None
        self.waiting = False

        ttk.Label(self, text="Агент задаёт вопросы — отвечайте коротко",
                  font=("TkDefaultFont", 12, "bold")).pack(anchor="w")
        self.counter_var = tk.StringVar()
        ttk.Label(self, textvariable=self.counter_var, foreground="gray").pack(anchor="w", pady=(0, 6))

        self.chat = scrolledtext.ScrolledText(self, wrap=tk.WORD, height=18,
                                              state="disabled", font=("TkDefaultFont", 11))
        self.chat.pack(fill=tk.BOTH, expand=True)
        self.chat.tag_config("agent", foreground="#1565c0")
        self.chat.tag_config("user", foreground="#2e7d32")

        entry_row = ttk.Frame(self)
        entry_row.pack(fill=tk.X, pady=8)
        self.entry = ttk.Entry(entry_row, font=("TkDefaultFont", 11))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry.bind("<Return>", lambda e: self.send_answer())
        self.send_btn = ttk.Button(entry_row, text="Ответить", command=self.send_answer)
        self.send_btn.pack(side=tk.LEFT, padx=6)

        bottom = ttk.Frame(self)
        bottom.pack(fill=tk.X)
        ttk.Button(bottom, text="Завершить опрос и получить результат",
                   command=self.finish).pack(side=tk.RIGHT)

    def configure_session(self, settings: dict):
        self.settings = settings
        self.messages = [{"role": "system", "content": build_interview_prompt(settings)}]
        self.questions_asked = 0
        mq = settings["max_questions"]
        self.max_q = int(mq) if mq != NOT_SET else None
        self.chat.config(state="normal")
        self.chat.delete("1.0", tk.END)
        self.chat.config(state="disabled")
        self._update_counter()
        # первый вопрос агента
        self._ask_agent()

    def _model(self):
        m = self.settings["model"]
        return m if m != NOT_SET else DEFAULT_MODEL

    def _update_counter(self):
        limit = f" из {self.max_q}" if self.max_q else ""
        self.counter_var.set(f"Вопросов задано: {self.questions_asked}{limit}")

    def _append(self, who, text, tag):
        self.chat.config(state="normal")
        self.chat.insert(tk.END, f"{who}: {text}\n\n", tag)
        self.chat.config(state="disabled")
        self.chat.see(tk.END)

    def _ask_agent(self):
        self.waiting = True
        self.send_btn.config(state="disabled")

        def on_ok(text):
            self.waiting = False
            self.send_btn.config(state="normal")
            if FINISH_MARKER in text:
                self.finish()
                return
            self.messages.append({"role": "assistant", "content": text})
            self.questions_asked += 1
            self._update_counter()
            self._append("Агент", text, "agent")

        self.app.api_call(self._model(), list(self.messages), False, on_ok)

    def send_answer(self):
        if self.waiting:
            return
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, tk.END)
        self._append("Вы", text, "user")
        self.messages.append({"role": "user", "content": text})

        if self.max_q is not None and self.questions_asked >= self.max_q:
            self.finish()  # лимит вопросов исчерпан
        else:
            self._ask_agent()

    def finish(self):
        result = self.app.frames[ResultFrame]
        result.configure_session(self.settings, self.messages)
        self.app.show(ResultFrame)


# ---------------------------------------------------------------------------
# Экран 3: результат
# ---------------------------------------------------------------------------

class ResultFrame(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, padding=15)
        self.app = app

        ttk.Label(self, text="Рекомендация", font=("TkDefaultFont", 13, "bold")).pack(anchor="w")
        self.output = scrolledtext.ScrolledText(self, wrap=tk.WORD,
                                                font=("TkDefaultFont", 11))
        self.output.pack(fill=tk.BOTH, expand=True, pady=8)

        bottom = ttk.Frame(self)
        bottom.pack(fill=tk.X)
        ttk.Button(bottom, text="Новый подбор",
                   command=lambda: app.show(SettingsFrame)).pack(side=tk.LEFT)

    def configure_session(self, settings, interview_messages):
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, "Формирую рекомендацию…\n")

        dialogue = "\n".join(
            f"{'Агент' if m['role'] == 'assistant' else 'Пользователь'}: {m['content']}"
            for m in interview_messages if m["role"] != "system"
        )
        system, use_json = build_result_prompt(settings)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Диалог опроса:\n{dialogue}\n\nСформируй итоговую рекомендацию."},
        ]
        model = settings["model"] if settings["model"] != NOT_SET else DEFAULT_MODEL

        def on_ok(text):
            self.output.delete("1.0", tk.END)
            self._render(text)

        self.app.api_call(model, messages, use_json, on_ok)

    def _render(self, text):
        # только сырой ответ нейросети (JSON или свободный текст)
        self.output.insert(tk.END, text + "\n")
        self.output.see("1.0")


if __name__ == "__main__":
    app = PetPickerApp()
    if app.winfo_exists():
        app.mainloop()
