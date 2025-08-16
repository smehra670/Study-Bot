
import math
from flask import Flask, jsonify, request, redirect, url_for, session
import json, os, re, random, difflib
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash


from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools import tool


load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
MODEL_ID = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "dev-secret-change-me")
USERS_PATH = "users.json"


QUIZ_CACHE_PATH = "quiz_cache.json"
MAX_CACHE_PER_KEY = 200

def _load_quiz_cache():
    if os.path.exists(QUIZ_CACHE_PATH):
        try:
            with open(QUIZ_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_quiz_cache(cache):
    try:
        with open(QUIZ_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _cache_key(topic, difficulty):
    base = re.sub(r"\s+", " ", (topic or "").strip().lower())
    base = re.sub(r"[^\w\s]", "", base)
    return f"{base}::{difficulty.lower()}"

def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _similar(a: str, b: str, thresh: float = 0.90) -> bool:
    return difflib.SequenceMatcher(a=_norm(a), b=_norm(b)).ratio() >= thresh

def _read_users():
    if not os.path.exists(USERS_PATH):
        return []
    try:
        with open(USERS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    users = []
    if isinstance(data, list):
        users = [u for u in data if isinstance(u, dict) and "username" in u]
    elif isinstance(data, dict):
        if "users" in data and isinstance(data["users"], list):
            users = [u for u in data["users"] if isinstance(u, dict) and "username" in u]
        else:
            for name, v in data.items():
                if isinstance(v, dict):
                    users.append({"username": name, **v})
                else:
                    users.append({"username": name, "password_hash": str(v)})
    return users

def _write_users(users):
    cleaned = []
    for u in users:
        if isinstance(u, dict) and "username" in u:
            cleaned.append({
                "username": str(u["username"]).strip(),
                "email": str(u.get("email", "")).strip(),
                "password_hash": str(u.get("password_hash", "")),
                "created_at": u.get("created_at", datetime.now().isoformat(timespec="seconds")),
            })
    with open(USERS_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

def _find_user(username):
    uname = (username or "").strip().lower()
    for u in _read_users():
        if (u.get("username", "") or "").strip().lower() == uname:
            return u
    return None

def login_required(fn):
    from functools import wraps
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapper

@tool
def homework_help_tool(question: str) -> str:
    return question

@tool
def summary(_: str) -> str:
    return ""

@tool
def quiz(easy,medium,hard) -> str:
    return ""

def make_agent(role, tools, instructions):
    try:
        return Agent(
            role=role,
            model=Groq(MODEL_ID),
            tools=tools,
            instructions=instructions,
            show_tool_calls=False,
            markdown=True,
        )
    except Exception as e:
        print(f"❌ Failed to create agent '{role}':", e)
        return None

helper = make_agent(
    "You are an intelligent homework helper and tutor.",
    [homework_help_tool],
    """You are a helpful homework assistant that guides students to learn by providing hints, explanations, and step-by-step guidance.
    Keep responses concise but comprehensive, typically 2-4 sentences with clear guidance."""
)

sum_agent = make_agent(
    "You are a homework helper.",
    [summary],
    "Analyze the user's extract. Provide structured bullet points: key ideas, quotations, techniques, and high-impact notes. Keep it concise and scannable.",
)
mcq_agent = make_agent(
    "You are a quiz generator.",
    [quiz],
    "Give mcq questions based on the level chosen by the user 'easy','medium' or 'hard'. Easy is primary school level, medium is secondary level and hard is A-level type hard questions. Clear options, 1 correct answer, and a suitable length explanation.",
)

def _to_text(out) -> str:
    text = getattr(out, "content", None)
    if not text:
        text = str(out or "")
    return text.strip()



_ansi_re = re.compile(r"\x1B\[[0-9;]*[mK]")
_ctrl_re = re.compile(r"[\x00-\x08\x0B-\x1F\x7F]")

def _strip_ansi(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = _ansi_re.sub("", s)
    s = _ctrl_re.sub("", s)
    return s

def ai_hint(question: str) -> str:
    if helper is None:
        return "Error: AI agent not initialized (check GROQ_API_KEY)."
    try:
        enhanced_prompt = f"""
        Student Question: {question}

        Provide a helpful response that:
        - Guides the student toward understanding
        - Gives clear hints without revealing the complete answer
        - Uses step-by-step thinking when appropriate
        - Is encouraging and educational
        """
        text = _to_text(helper.run(enhanced_prompt))
        cleaned_text = _strip_ansi(text)
        if not cleaned_text or len(cleaned_text.strip()) < 20:
            return "I'd be happy to help! Could you share which step or idea is confusing so I can target the hint?"
        return cleaned_text
    except Exception as e:
        print(f"Error in ai_hint: {e}")
        return "I'm having trouble processing that right now. Try rephrasing or breaking it into smaller parts."

def _clean_json(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE)
    a, b = s.find("{"), s.rfind("}")
    return s[a:b+1] if a != -1 and b != -1 and b > a else s


def _guess_subject(s: str) -> str:
    s = _norm(s)
    if any(k in s for k in (
        "algebra","equation","quadratic","slope","pythag","function","factor",
        "derivative","integral","calculus","matrix","probability","geometry","angle"
    )):
        return "math"
    if any(k in s for k in (
        "velocity","acceleration","force","newton","momentum","energy","work","power",
        "displacement","kinemat","projectile","ohm","current","voltage","resistance"
    )):
        return "physics"
    if any(k in s for k in (
        "atom","mole","molar","acid","base","salt","periodic","ion","bond",
        "reaction","stoichiometry","equilibrium","gas"
    )):
        return "chem"
    if any(k in s for k in (
        "cell","photosynthesis","respiration","enzyme","dna","protein","osmosis","diffusion","ecosystem"
    )):
        return "bio"
    return "generic"

_BAD_PHRASES = [
    "which statement is true about", "key fact about", "which option relates",
    "commonly associated with", "correct description of", "best describes",
    "nothing to do with", "brand of soft drink", "unrelated to"
]
def _is_trivial_q(text: str) -> bool:
    t = _norm(text)
    if len(t) < 25:
        return True
    return any(p in t for p in _BAD_PHRASES)

def _bad_option(opt: str) -> bool:
    o = _norm(opt)
    return any(p in o for p in ("brand of soft drink", "nothing to do with", "unrelated"))

_rng = random.Random()

def _fallback_math(count: int, difficulty: str) -> list:
    qs = []
    used = set()
    def push(q, opts, ans_idx, exp):
        key = _norm(q)
        if key in used:
            return
        used.add(key)
        qs.append({"q": q, "options": opts, "answer_index": ans_idx, "explanation": exp})

    while len(qs) < count:
        kind = _rng.choice(["linear","slope","derivative","pythag","quadratic_sum"])
        if difficulty == "hard":
            kind = _rng.choice(["derivative","quadratic_sum","slope","linear","pythag"])

        if kind == "linear":
            a = _rng.randint(2,9); s = _rng.randint(-6,6)
            b = _rng.randint(-10,10); c = a*s + b
            q = f"Solve for x: {a}x + {b} = {c}"
            correct = s
            distractors = list({s+_rng.choice([-2,-1,1,2]), s+_rng.choice([-3,3]), s+_rng.choice([-4,4])})
            opts = [str(correct)] + [str(d) for d in distractors][:3]
            _rng.shuffle(opts)
            push(q, opts, opts.index(str(correct)), "Subtract the constant then divide by the coefficient of x.")

        elif kind == "slope":
            x1, y1 = _rng.randint(-5,4), _rng.randint(-5,4)
            x2 = x1 + _rng.choice([-4,-3,-2,2,3,4]); y2 = y1 + _rng.randint(-5,5)
            slope = (y2 - y1) / (x2 - x1)
            q = f"Find the slope of the line through ({x1}, {y1}) and ({x2}, {y2})."
            correct = round(slope, 2)
            opts = [correct,
                    round(correct + _rng.choice([-1, -0.5, 0.5, 1]), 2),
                    round(-correct, 2),
                    round(correct + _rng.choice([-0.25, 0.25, 0.75, -0.75]), 2)]
            opts = [str(o) for o in opts]
            _rng.shuffle(opts)
            push(q, opts, opts.index(str(correct)), "Slope = (y₂−y₁)/(x₂−x₁).")

        elif kind == "derivative":
            a = _rng.randint(2,6); n = _rng.randint(2,5)
            q = f"Find d/dx of {a}x^{n}."
            correct = f"{a*n}x^{n-1}" if n-1 != 1 else f"{a*n}x"
            wrongs = [
                f"{a*(n-1)}x^{n-2}" if n-2>1 else f"{a*(n-1)}x",
                f"{a}x^{n-1}",
                f"{a*n}x^{n+1}"
            ]
            opts = [correct] + wrongs
            _rng.shuffle(opts)
            push(q, opts, opts.index(correct), "Power rule: d/dx(xⁿ)=n·xⁿ⁻¹ and multiply by the coefficient.")

        elif kind == "pythag":
            triple = _rng.choice([(3,4,5),(5,12,13),(8,15,17)])
            a,b,c = triple
            if _rng.random() < 0.5:
                q = f"A right triangle has legs {a} and {b}. What is the hypotenuse length?"
                correct = c
                opts = [c, c+_rng.choice([-2,2]), a+b, max(a,b)+2]
                exp = "c = √(a²+b²)."
            else:
                q = f"A right triangle has hypotenuse {c} and one leg {a}. What is the other leg?"
                leg = int(round(math.sqrt(c*c - a*a)))
                correct = leg
                opts = [leg, leg+_rng.choice([-2,2]), c-a, a+1]
                exp = "b = √(c²−a²)."
            opts = [str(o) for o in opts]
            _rng.shuffle(opts)
            push(q, opts, opts.index(str(correct)), exp)

        else:
            r1 = _rng.randint(-5,5); r2 = _rng.randint(-5,5)
            while r1 == r2:
                r2 = _rng.randint(-5,5)
            q = f"For the quadratic x² - ({r1+r2})x + {r1*r2} = 0, what is the sum of the roots?"
            correct = r1 + r2
            opts = [str(correct), str(correct+_rng.choice([-2,-1,1,2])), str(r1), str(r2)]
            _rng.shuffle(opts)
            push(q, opts, opts.index(str(correct)), "Sum of roots = -b (for ax²+bx+c=0) when a=1.")

    return qs[:count]

def _fallback_physics(topic: str, count: int, difficulty: str) -> list:
    qs = []
    used = set()
    def push(q, opts, ans_idx, exp):
        key = _norm(q)
        if key in used: return
        used.add(key)
        qs.append({"q": q, "options": opts, "answer_index": ans_idx, "explanation": exp})

    while len(qs) < count:
        kind = _rng.choice(["v=dx/dt","a=(v-u)/t","s=ut+0.5at^2","vector_scalar","units"])
        if difficulty == "hard":
            kind = _rng.choice(["s=ut+0.5at^2","a=(v-u)/t","v=dx/dt"])

        if kind == "v=dx/dt":
            d = _rng.randint(60, 240); t = _rng.randint(4, 12)
            v = round(d/t, 2)
            q = f"An object travels {d} m in {t} s at constant velocity. What is its velocity?"
            opts = [f"{v} m/s", f"{round(v+_rng.choice([-1.5,-1,-0.5,0.5,1,1.5]),2)} m/s", f"{d} m/s", f"{t} m/s"]
            _rng.shuffle(opts)
            push(q, opts, opts.index(f"{v} m/s"), "v = Δx/Δt")

        elif kind == "a=(v-u)/t":
            u = _rng.randint(2, 12); a = _rng.randint(1, 5); t = _rng.randint(3, 8)
            v = u + a*t
            q = f"A body accelerates from {u} m/s to {v} m/s in {t} s. What is the acceleration?"
            correct = f"{round((v-u)/t,2)} m/s²"
            opts = [correct, f"{v/t:.2f} m/s²", f"{u/t:.2f} m/s²", f"{(v+u)/t:.2f} m/s²"]
            _rng.shuffle(opts)
            push(q, opts, opts.index(correct), "a = (v−u)/t")

        elif kind == "s=ut+0.5at^2":
            u = _rng.randint(0, 10); a = _rng.randint(1, 4); t = _rng.randint(2, 6)
            s = u*t + 0.5*a*t*t
            q = f"With u={u} m/s and a={a} m/s², distance in {t} s?"
            correct = f"{s:.1f} m"
            opts = [correct, f"{(u*t):.1f} m", f"{(a*t*t):.1f} m", f"{(u*t + a*t*t):.1f} m"]
            _rng.shuffle(opts)
            push(q, opts, opts.index(correct), "s = ut + ½at²")

        elif kind == "vector_scalar":
            q = "Which quantity is a vector?"
            correct = "Displacement"
            opts = [correct, "Speed", "Temperature (magnitude)", "Mass"]
            _rng.shuffle(opts)
            push(q, opts, opts.index(correct), "Vectors have magnitude and direction.")

        else:
            q = "SI unit of force?"
            correct = "Newton (N)"
            opts = [correct, "Joule (J)", "Pascal (Pa)", "Watt (W)"]
            _rng.shuffle(opts)
            push(q, opts, opts.index(correct), "Force → newton (N).")

    return qs[:count]

def _fallback_generic(topic: str, count: int, difficulty: str) -> list:
    base = [
        (f"Which statement best defines {topic}?",
         [f"A precise definition of {topic}.", "A random fact not defining it.", "An unrelated description.", "A vague description."],
         0, f"The correct option states the core definition of {topic}."),
        (f"Which example illustrates {topic} in practice?",
         [f"A scenario where {topic} is applied correctly.", "A scenario with no connection.", "A pop-culture reference.", "A contradictory scenario."],
         0, f"The correct option shows {topic} being used.")
    ]
    out = []
    for q, opts, idx, exp in base[:count]:
        out.append({"q": q, "options": opts, "answer_index": idx, "explanation": exp})
    return out

def _fallback_make_questions(topic: str, needed: int, difficulty: str) -> list:
    subject = _guess_subject(topic)
    if subject == "math":
        return _fallback_math(needed, difficulty)
    if subject == "physics":
        return _fallback_physics(topic, needed, difficulty)
    return _fallback_generic(topic or "the topic", needed, difficulty)

def ai_quiz(topic_or_text: str, count: int = 10, difficulty: str = "medium") -> dict:
    if mcq_agent is None:
        return {"title": "Quiz", "questions": []}

    cache = _load_quiz_cache()
    key = _cache_key(topic_or_text, difficulty)
    prev_list = cache.get(key, [])
    prev_recent = prev_list[-50:]
    seed = random.randint(1, 10**9)

    schema = {
        "title": "Short quiz title",
        "questions": [{
            "q": "Self-contained, non-trivial question",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "answer_index": 0,
            "explanation": "Short, specific why-correct explanation."
        }]
    }

    difficulty_descriptions = {
        "easy": "Primary school level - basic recall or one-step.",
        "medium": "Secondary/GCSE - multi-step reasoning or application.",
        "hard": "A-level - deeper reasoning, multi-step, calculations or interpretation."
    }
    difficulty_desc = difficulty_descriptions.get(difficulty, difficulty_descriptions["medium"])

    def _prompt(seed_val: int, tighten=False):
        tighten_txt = ""
        if tighten:
            tighten_txt = """
STRICT QUALITY RULES:
- NO generic stems like "Which statement is true about...", "key fact about...", etc.
- Avoid trivial distractors.
- Prefer calculation / scenario / concept-application.
- Provide specific explanations.
"""
        avoid_block = "\n".join(f"- {q}" for q in prev_recent[:15]) if prev_recent else "None"
        return f"""
Create a {count}-question multiple-choice quiz from the user's input below.

DIFFICULTY: {difficulty_desc}
FORMATS TO MIX: definition, calculation, worked example, error-spotting, compare/contrast.
RIGOR:
- Each question must be self-contained and specific.
- Avoid reusing or paraphrasing these previous questions:
{avoid_block}
- RANDOM_SEED: {seed_val}
{tighten_txt}
Return ONLY valid JSON in exactly this schema (no prose, no backticks):
{json.dumps(schema, indent=2)}

User input:
\"\"\"{topic_or_text}\"\"\""""

    all_qs = []
    used_norms = set(_norm(x) for x in prev_list)
    attempts = 0
    while len(all_qs) < count and attempts < 3:
        attempts += 1
        try:
            raw = _to_text(mcq_agent.run(_prompt(seed + attempts*999, tighten=attempts >= 2)))
        except Exception:
            raw = ""
        cleaned = _clean_json(raw)
        try:
            data = json.loads(cleaned)
        except Exception:
            data = {"questions": []}

        for item in (data.get("questions") or []):
            qtext = str(item.get("q", "")).strip()
            if not qtext or _is_trivial_q(qtext):
                continue
            if any(_similar(qtext, p) for p in prev_list) or _norm(qtext) in used_norms:
                continue

            opts_raw = [str(x).strip() for x in (item.get("options") or []) if str(x).strip()]
            if len(opts_raw) < 4 or any(_bad_option(o) for o in opts_raw):
                continue
            opts = opts_raw[:4]

            try:
                ai_idx = int(item.get("answer_index", 0))
            except Exception:
                ai_idx = 0
            if ai_idx not in (0,1,2,3):
                continue

            exp = (item.get("explanation") or "").strip()
            if len(exp) < 8:
                continue

            used_norms.add(_norm(qtext))
            all_qs.append({"q": qtext, "options": opts, "answer_index": ai_idx, "explanation": exp})
            if len(all_qs) >= count:
                break

    if len(all_qs) < count:
        needed = count - len(all_qs)
        all_qs.extend(_fallback_make_questions(topic_or_text, needed, difficulty))

    random.shuffle(all_qs)
    all_qs = all_qs[:count]

    new_questions = [q["q"] for q in all_qs]
    cache.setdefault(key, [])
    cache[key].extend(new_questions)
    cache[key] = cache[key][-MAX_CACHE_PER_KEY:]
    _save_quiz_cache(cache)

    title = "Interactive Quiz" if not new_questions else (f"Interactive Quiz ({difficulty.capitalize()})" if difficulty != "medium" else "Interactive Quiz")
    return {"title": title, "questions": all_qs}


def _html(page_body: str, title: str = "StudyBot"):
    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>{title}</title>
<script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 min-h-screen">
{page_body}
</body></html>"""


@app.route("/")
def root():
    return redirect(url_for("app_home") if session.get("user") else url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        u = _find_user(username)
        if u and u.get("password_hash") and check_password_hash(u["password_hash"], password):
            session["user"] = {"username": u["username"], "email": u.get("email","")}
            return redirect(url_for("app_home"))
        msg = "Invalid username or password."
    else:
        msg = ""

    body = f"""
<div class="flex items-center justify-center pt-16 px-4">
  <div class="w-full max-w-md bg-white rounded-2xl shadow-xl p-8">
    <div class="flex items-center gap-3 mb-6">
      <div class="w-10 h-10 rounded-xl bg-gradient-to-r from-pink-500 to-purple-600"></div>
      <div>
        <h1 class="text-xl font-bold text-gray-800">StudyBot</h1>
        <p class="text-sm text-gray-500">Learning Assistant</p>
      </div>
    </div>
    <h2 class="text-2xl font-semibold text-gray-800 mb-1">Welcome back</h2>
    <p class="text-sm text-gray-500 mb-6">Log in to continue</p>
    {f'<div class="mb-4 text-sm text-red-600">{msg}</div>' if msg else ''}
    <form method="POST" class="space-y-4">
      <input name="username" placeholder="Username" required
             class="w-full px-4 py-3 rounded-lg border focus:ring-2 focus:ring-pink-500 outline-none"/>
      <input name="password" type="password" placeholder="Password" required
             class="w-full px-4 py-3 rounded-lg border focus:ring-2 focus:ring-pink-500 outline-none"/>
      <button class="w-full py-3 rounded-lg bg-gradient-to-r from-pink-500 to-pink-600 text-white font-medium hover:from-pink-600 hover:to-pink-700">
        Log in
      </button>
    </form>
    <p class="text-sm text-gray-500 mt-6">No account? <a class="text-pink-600 hover:underline" href="/signup">Sign up</a></p>
  </div>
</div>"""
    return _html(body, "Login • StudyBot")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        email = (request.form.get("email") or "").strip()
        password = request.form.get("password") or ""
        if not username or not password:
            msg = "Username and password are required."
        elif _find_user(username):
            msg = "That username is already taken."
        else:
            users = _read_users()
            users.append({
                "username": username,
                "email": email,
                "password_hash": generate_password_hash(password),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            })
            _write_users(users)
            session["user"] = {"username": username, "email": email}
            return redirect(url_for("app_home"))
    else:
        msg = ""

    body = f"""
<div class="flex items-center justify-center pt-16 px-4">
  <div class="w-full max-w-md bg-white rounded-2xl shadow-xl p-8">
    <div class="flex items-center gap-3 mb-6">
      <div class="w-10 h-10 rounded-xl bg-gradient-to-r from-pink-500 to-purple-600"></div>
      <div>
        <h1 class="text-xl font-bold text-gray-800">StudyBot</h1>
        <p class="text-sm text-gray-500">Create your account</p>
      </div>
    </div>
    <h2 class="text-2xl font-semibold text-gray-800 mb-1">Sign up</h2>
    <p class="text-sm text-gray-500 mb-6">It's quick and free</p>
    {f'<div class="mb-4 text-sm text-red-600">{msg}</div>' if msg else ''}
    <form method="POST" class="space-y-4">
      <input name="username" placeholder="Username" required
             class="w-full px-4 py-3 rounded-lg border focus:ring-2 focus:ring-pink-500 outline-none"/>
      <input name="email" type="email" placeholder="Email (optional)"
             class="w-full px-4 py-3 rounded-lg border focus:ring-2 focus:ring-pink-500 outline-none"/>
      <input name="password" type="password" placeholder="Password" required
             class="w-full px-4 py-3 rounded-lg border focus:ring-2 focus:ring-pink-500 outline-none"/>
      <button class="w-full py-3 rounded-lg bg-gradient-to-r from-pink-500 to-pink-600 text-white font-medium hover:from-pink-600 hover:to-pink-700">
        Create account
      </button>
    </form>
    <p class="text-sm text-gray-500 mt-6">Already have an account? <a class="text-pink-600 hover:underline" href="/login">Log in</a></p>
  </div>
</div>"""
    return _html(body, "Sign up • StudyBot")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/app")
@login_required
def app_home():
    user = session.get("user", {})
    username = user.get("username", "You")

    page = f"""
<div id="app">Loading…</div>
<script src="https://cdn.tailwindcss.com"></script>
<style>
.opt input {{ accent-color:#ec4899; }}
.message-user {{ background: linear-gradient(135deg, #3b82f6, #1d4ed8); }}
.message-helper {{ background: #f8fafc; border: 1px solid #e2e8f0; }}
.homework-container {{ 
  background: linear-gradient(135deg, #f0f9ff 0%, #e0e7ff 100%);
  min-height: 100vh;
}}
.chat-container {{
  max-width: 800px;
  margin: 0 auto;
  padding: 2rem;
}}
.helper-avatar {{
  background: linear-gradient(135deg, #ec4899, #be185d);
  width: 40px;
  height: 40px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-weight: bold;
  flex-shrink: 0;
}}
.card {{
  perspective: 1000px;
}}
.card-inner {{
  transition: transform .4s;
  transform-style: preserve-3d;
}}
.card.flipped .card-inner {{
  transform: rotateY(180deg);
}}
.card-face {{
  backface-visibility: hidden;
}}
.card-back {{
  transform: rotateY(180deg);
}}
</style>
<script>
  let state = {{
    question: "",
    activeTab: "homework",
    sidebarOpen: true,
    currentQuestion: "",
    conversation: [],
    loading: false,
    essayLoading: false,
    quizLoading: false,
    quizStage: "setup",
    quizTitle: "Interactive Quiz",
    quizQuestions: [],
    quizAnswers: [],
    quizScore: 0,
    essayText: "",
    essaySummary: "",
    quizSource: "",
    quizCount: 5,
    currentQuestionIndex: 0,
    quizDifficulty: "medium",

    // SIMPLE FLASHCARDS (client-only)
    fcFront: "",
    fcBack: "",
    fcCards: []  // {{id, front, back, flipped:false}}
  }};

  const username = {json.dumps(username)};

  function esc(s) {{
    return String(s).replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}})[c]);
  }}

  function render() {{
    const app = document.getElementById('app');

    const sidebarHTML = `
      <div class="${{state.sidebarOpen ? 'w-64' : 'w-16'}} bg-white border-r transition-all duration-300 flex flex-col shadow-lg">
        <div class="p-4 border-b">
          <div class="flex items-center gap-3">
            <div class="w-8 h-8 rounded-lg bg-gradient-to-r from-pink-500 to-purple-600 flex items-center justify-center">
              <span class="text-white text-sm font-bold">S</span>
            </div>
            ${{state.sidebarOpen ? '<span class="font-semibold text-gray-800">StudyBot</span>' : ''}}
          </div>
        </div>
        <nav class="flex-1 p-4">
          <div class="space-y-2">
            <button onclick="setTab('homework')" class="w-full flex items-center gap-3 px-3 py-3 rounded-lg transition-colors font-medium ${{state.activeTab==='homework'?'bg-pink-100 text-pink-700 shadow-sm':'text-gray-600 hover:bg-gray-100'}}">
              <span class="text-lg">🎓</span>${{state.sidebarOpen?'<span>Homework Helper</span>':''}}
            </button>
            <button onclick="setTab('essays')" class="w-full flex items-center gap-3 px-3 py-3 rounded-lg transition-colors font-medium ${{state.activeTab==='essays'?'bg-pink-100 text-pink-700 shadow-sm':'text-gray-600 hover:bg-gray-100'}}">
              <span class="text-lg">📝</span>${{state.sidebarOpen?'<span>Essay Analysis</span>':''}}
            </button>
            <button onclick="setTab('quizzes')" class="w-full flex items-center gap-3 px-3 py-3 rounded-lg transition-colors font-medium ${{state.activeTab==='quizzes'?'bg-pink-100 text-pink-700 shadow-sm':'text-gray-600 hover:bg-gray-100'}}">
              <span class="text-lg">🎯</span>${{state.sidebarOpen?'<span>Interactive Quizzes</span>':''}}
            </button>
            <button onclick="setTab('flashcards')" class="w-full flex items-center gap-3 px-3 py-3 rounded-lg transition-colors font-medium ${{state.activeTab==='flashcards'?'bg-pink-100 text-pink-700 shadow-sm':'text-gray-600 hover:bg-gray-100'}}">
              <span class="text-lg">🧠</span>${{state.sidebarOpen?'<span>Flashcards</span>':''}}
            </button>
          </div>
        </nav>
        <div class="p-4 border-t">
          <div class="flex items-center gap-3">
            <div class="w-8 h-8 bg-gradient-to-r from-blue-400 to-blue-600 rounded-full flex items-center justify-center">
              <span class="text-white text-sm font-medium">${{username.charAt(0).toUpperCase()}}</span>
            </div>
            ${{state.sidebarOpen ? `<div class="flex-1"><div class="text-sm font-medium text-gray-800">${{esc(username)}}</div><a href="/logout" class="text-xs text-gray-500 hover:underline">Sign out</a></div>` : ''}}
          </div>
        </div>
      </div>`;

    let contentHTML = '';

    // ===== HOMEWORK =====
    if (state.activeTab === 'homework') {{
      const conversationHTML = state.conversation.map((msg) => {{
        if (msg.type === 'user') {{
          return `
            <div class="flex justify-end mb-6">
              <div class="max-w-lg">
                <div class="message-user text-white px-6 py-4 rounded-2xl rounded-br-md shadow-lg">
                  <div class="whitespace-pre-wrap break-words">${{esc(msg.content)}}</div>
                </div>
              </div>
            </div>`;
        }} else {{
          return `
            <div class="flex gap-4 mb-6">
              <div class="helper-avatar">H</div>
              <div class="flex-1 max-w-2xl">
                <div class="message-helper px-6 py-4 rounded-2xl rounded-bl-md shadow-sm">
                  <div class="whitespace-pre-wrap break-words text-gray-800 leading-relaxed">${{esc(msg.content)}}</div>
                </div>
              </div>
            </div>`;
        }}
      }}).join('');

      contentHTML = `
        <div class="homework-container flex-1 flex flex-col">
          <div class="bg-white border-b px-6 py-6 shadow-sm">
            <div class="chat-container">
              <div class="text-center">
                <h1 class="text-2xl font-bold text-gray-800 mb-2">🎓 Homework Assistant</h1>
                <p class="text-gray-600">Ask me a homework question and I'll give you a helpful hint!</p>
              </div>
            </div>
          </div>
          
          <div class="flex-1 overflow-y-auto">
            <div class="chat-container py-8">
              ${{state.conversation.length === 0 ? `
                <div class="text-center py-16">
                  <div class="helper-avatar mx-auto mb-6" style="width: 80px; height: 80px; font-size: 2rem;">H</div>
                  <h2 class="text-xl font-semibold text-gray-800 mb-2">Hello! I'm here to help</h2>
                  <p class="text-gray-600 mb-8">Ask me any homework question and I'll guide you through it step by step!</p>
                </div>
              ` : conversationHTML}}
              
              ${{state.loading ? `
                <div class="flex gap-4 mb-6">
                  <div class="helper-avatar">H</div>
                  <div class="flex-1 max-w-2xl">
                    <div class="message-helper px-6 py-4 rounded-2xl rounded-bl-md shadow-sm">
                      <div class="flex items-center gap-2">
                        <div class="animate-spin rounded-full h-4 w-4 border-2 border-pink-500 border-t-transparent"></div>
                        <span class="text-gray-600">Thinking about your question...</span>
                      </div>
                    </div>
                  </div>
                </div>
              ` : ''}}
            </div>
          </div>
          
          <div class="bg-white border-t shadow-lg">
            <div class="chat-container py-6">
              <div class="flex gap-3">
                <input type="text" 
                  placeholder="What homework question do you need help with?" 
                  value="${{state.currentQuestion}}" 
                  onkeydown="handleKeyDown(event)" 
                  oninput="updateQuestion(event)" 
                  class="flex-1 px-6 py-4 border-2 rounded-2xl focus:ring-2 focus:ring-pink-500 focus:border-pink-500 outline-none text-lg"/>
                <button onclick="askQuestion()" 
                  ${{state.loading ? 'disabled' : ''}} 
                  class="px-8 py-4 bg-gradient-to-r from-pink-500 to-pink-600 text-white rounded-2xl hover:from-pink-600 hover:to-pink-700 disabled:opacity-50 font-medium shadow-lg transition-all">
                  ${{state.loading ? '...' : 'Send'}}
                </button>
                <button onclick="resetChat()" 
                  class="px-6 py-4 border-2 rounded-2xl text-gray-700 hover:bg-gray-50 font-medium transition-colors">
                  Reset
                </button>
              </div>
            </div>
          </div>
        </div>`;
    }}

    // ===== ESSAYS =====
    else if (state.activeTab === 'essays') {{
      contentHTML = `
        <div class="flex-1 p-6">
          <div class="max-w-4xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div class="space-y-4">
              <h3 class="text-lg font-medium text-gray-800">Paste Your Extract</h3>
              <textarea placeholder="Paste your essay extract, study notes, or any text you want analyzed..." value="${{state.essayText}}" oninput="updateEssayText(event)" class="w-full h-64 px-4 py-3 border rounded-lg focus:ring-2 focus:ring-pink-500 outline-none resize-none"></textarea>
              <button onclick="analyzeEssay()" ${{state.essayLoading?'disabled':''}} class="w-full py-3 bg-pink-500 text-white rounded-lg hover:bg-pink-600 disabled:opacity-50">
                ${{state.essayLoading?'Analyzing...':'Analyze Extract'}}
              </button>
            </div>
            <div class="space-y-4">
              <h3 class="text-lg font-medium text-gray-800">Analysis</h3>
              <div class="h-64 p-4 border rounded-lg bg-gray-50 overflow-y-auto whitespace-pre-wrap">
                ${{state.essaySummary ? esc(state.essaySummary) : '<p class="text-gray-500">Analysis will appear here...</p>'}}
              </div>
            </div>
          </div>
        </div>`;
    }}

    // ===== QUIZZES =====
    else if (state.activeTab === 'quizzes') {{
      if (state.quizStage === 'setup') {{
        contentHTML = `
          <div class="flex-1 p-6">
            <div class="max-w-2xl mx-auto space-y-6">
              <div class="text-center">
                <h3 class="text-2xl font-semibold text-gray-800 mb-2">Create Your Quiz</h3>
                <p class="text-gray-600">Enter a topic or paste your study notes, then choose your difficulty level</p>
              </div>
              
              <div class="space-y-6">
                <div>
                  <label class="block text-sm font-medium text-gray-700 mb-2">Topic or Study Material</label>
                  <textarea 
                    placeholder="Enter a topic (e.g., 'photosynthesis') or paste your study notes..." 
                    value="${{state.quizSource}}" 
                    oninput="updateQuizSource(event)" 
                    class="w-full h-32 px-4 py-3 border rounded-lg focus:ring-2 focus:ring-pink-500 outline-none resize-none">
                  </textarea>
                </div>
                
                <div>
                  <label class="block text-sm font-medium text-gray-700 mb-3">Difficulty Level</label>
                  <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                    <button onclick="updateQuizDifficulty('easy')" class="p-4 text-left border-2 rounded-lg transition-all ${{state.quizDifficulty === 'easy' ? 'border-green-500 bg-green-50' : 'border-gray-200 hover:border-green-300'}}">
                      <div class="font-medium ${{state.quizDifficulty === 'easy' ? 'text-green-700' : 'text-gray-800'}}">Easy</div>
                      <div class="text-xs text-gray-600">Primary school level</div>
                    </button>
                    <button onclick="updateQuizDifficulty('medium')" class="p-4 text-left border-2 rounded-lg transition-all ${{state.quizDifficulty === 'medium' ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-blue-300'}}">
                      <div class="font-medium ${{state.quizDifficulty === 'medium' ? 'text-blue-700' : 'text-gray-800'}}">Medium</div>
                      <div class="text-xs text-gray-600">Secondary school level</div>
                    </button>
                    <button onclick="updateQuizDifficulty('hard')" class="p-4 text-left border-2 rounded-lg transition-all ${{state.quizDifficulty === 'hard' ? 'border-red-500 bg-red-50' : 'border-gray-200 hover:border-red-300'}}">
                      <div class="font-medium ${{state.quizDifficulty === 'hard' ? 'text-red-700' : 'text-gray-800'}}">Hard</div>
                      <div class="text-xs text-gray-600">A-level difficulty</div>
                    </button>
                  </div>
                </div>
                
                <div class="flex items-center gap-4">
                  <label class="text-sm font-medium text-gray-700">Number of questions:</label>
                  <select value="${{state.quizCount}}" onchange="updateQuizCount(event)" class="px-3 py-2 border rounded-lg focus:ring-2 focus:ring-pink-500 outline-none">
                    <option value="5">5 questions</option>
                    <option value="10">10 questions</option>
                    <option value="15">15 questions</option>
                  </select>
                </div>
                
                <button onclick="generateQuiz()" ${{state.quizLoading ? 'disabled' : ''}} class="w-full py-3 bg-pink-500 text-white rounded-lg hover:bg-pink-600 disabled:opacity-50 font-medium shadow-lg">
                  ${{state.quizLoading ? 'Generating Quiz...' : `Generate ${{state.quizDifficulty.charAt(0).toUpperCase() + state.quizDifficulty.slice(1)}} Quiz`}}
                </button>
              </div>
            </div>
          </div>`;
      }} else if (state.quizStage === 'taking') {{
        const q = state.quizQuestions[state.currentQuestionIndex];
        const progress = ((state.currentQuestionIndex + 1) / state.quizQuestions.length) * 100;
        contentHTML = `
          <div class="flex-1 p-6">
            <div class="max-w-2xl mx-auto">
              <div class="mb-6">
                <div class="flex justify-between items-center mb-2">
                  <span class="text-sm text-gray-600">Question ${{state.currentQuestionIndex + 1}} of ${{state.quizQuestions.length}}</span>
                  <span class="text-sm text-gray-600">${{Math.round(progress)}}% Complete</span>
                </div>
                <div class="w-full bg-gray-200 rounded-full h-2"><div class="bg-pink-500 h-2 rounded-full" style="width:${{progress}}%"></div></div>
              </div>
              <div class="bg-white rounded-lg p-6 shadow-sm border">
                <h3 class="text-lg font-medium text-gray-800 mb-4">${{esc(q.q)}}</h3>
                <div class="space-y-3">
                  ${{q.options.map((o,i)=>`
                    <label class="flex items-center p-3 border rounded-lg cursor-pointer hover:bg-gray-50 opt">
                      <input type="radio" name="answer" value=${{i}} onchange="selectAnswer(${{i}})" class="mr-3"/>
                      <span>${{esc(o)}}</span>
                    </label>`).join('')}}
                </div>
                <div class="mt-6 flex justify-between">
                  <button onclick="previousQuestion()" ${{state.currentQuestionIndex===0?'disabled':''}} class="px-4 py-2 text-gray-600 border rounded-lg hover:bg-gray-50 disabled:opacity-50">Previous</button>
                  <button onclick="nextQuestion()" class="px-6 py-2 bg-pink-500 text-white rounded-lg hover:bg-pink-600">
                    ${{state.currentQuestionIndex===state.quizQuestions.length-1?'Finish Quiz':'Next'}}
                  </button>
                </div>
              </div>
            </div>
          </div>`;
      }} else {{
        const pct = Math.round((state.quizScore / state.quizQuestions.length) * 100);
        contentHTML = `
          <div class="flex-1 p-6">
            <div class="max-w-2xl mx-auto">
              <div class="text-center mb-6">
                <h3 class="text-2xl font-semibold text-gray-800 mb-2">Quiz Complete!</h3>
                <div class="text-4xl font-bold text-pink-500 mb-2">${{state.quizScore}}/${{state.quizQuestions.length}}</div>
                <div class="text-xl text-gray-600">${{pct}}% Score</div>
              </div>
              <div class="space-y-4 mb-6">
                ${{state.quizQuestions.map((q,i)=>`
                  <div class="bg-white rounded-lg p-4 border">
                    <div class="font-medium text-gray-800 mb-2">${{esc(q.q)}}</div>
                    <div class="text-sm text-gray-600 mb-2">
                      Your answer: ${{esc(q.options[state.quizAnswers[i]??0])}}
                      ${{state.quizAnswers[i]!==q.answer_index?`<br>Correct: ${{esc(q.options[q.answer_index])}}`:''}}
                    </div>
                    <div class="text-sm text-gray-500">${{esc(q.explanation)}}</div>
                  </div>`).join('')}}
              </div>
              <div class="flex gap-3">
                <button onclick="retakeQuiz()" class="flex-1 py-2 border text-gray-700 rounded-lg hover:bg-gray-50">Retake Quiz</button>
                <button onclick="newQuiz()" class="flex-1 py-2 bg-pink-500 text-white rounded-lg hover:bg-pink-600">New Quiz</button>
              </div>
            </div>
          </div>`;
      }}
    }}

    // ===== FLASHCARDS (client-only) =====
    else if (state.activeTab === 'flashcards') {{
      const cards = state.fcCards.map((c, i) => `
        <div class="card ${{c.flipped ? 'flipped' : ''}}" onclick="toggleCard(${{i}})">
          <div class="card-inner relative h-full">
            <div class="card-face absolute inset-0 bg-white border rounded-xl p-4 flex items-center justify-center">
              <div class="text-center text-gray-800 font-medium whitespace-pre-wrap break-words">${{esc(c.front)}}</div>
            </div>
            <div class="card-face card-back absolute inset-0 bg-gray-900 text-white border rounded-xl p-4 flex items-center justify-center">
              <div class="text-center font-medium whitespace-pre-wrap break-words">${{esc(c.back)}}</div>
            </div>
          </div>
        </div>
      `).join('');

      contentHTML = `
        <div class="flex-1 p-6">
          <div class="max-w-5xl mx-auto">
            <h2 class="text-2xl font-semibold text-gray-800 mb-2">🧠 Flashcards</h2>
            <p class="text-gray-600 mb-6">Add cards, then tap a card to flip between front/back.</p>

            <div class="bg-white border rounded-xl p-4 mb-6">
              <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                <input placeholder="Front (question / prompt)" value="${{state.fcFront}}" oninput="state.fcFront=this.value" class="px-4 py-3 border rounded-lg focus:ring-2 focus:ring-pink-500 outline-none"/>
                <input placeholder="Back (answer)" value="${{state.fcBack}}" oninput="state.fcBack=this.value" class="px-4 py-3 border rounded-lg focus:ring-2 focus:ring-pink-500 outline-none"/>
              </div>
              <div class="mt-3 flex gap-3">
                <button onclick="addCard()" class="px-5 py-2 bg-pink-500 text-white rounded-lg hover:bg-pink-600">Add card</button>
                <button onclick="clearCards()" class="px-5 py-2 border rounded-lg text-gray-700 hover:bg-gray-50">Clear all</button>
                <div class="text-sm text-gray-500 ml-auto self-center">Total: ${{state.fcCards.length}}</div>
              </div>
            </div>

            <div class="h-[60vh] overflow-y-auto">
              <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                ${{ cards || '<div class="text-gray-500">No cards yet. Add a card above.</div>' }}
              </div>
            </div>
          </div>
        </div>`;
    }}

    app.innerHTML = `<div class="flex h-screen">${{sidebarHTML}}<div class="flex-1 flex flex-col">${{contentHTML}}</div></div>`;

    // size cards nicely
    if (state.activeTab === 'flashcards') {{
      document.querySelectorAll('.card').forEach(el => {{
        el.style.height = '180px';
      }});
    }}
  }}

  // ---- helpers
  function setTab(t){{state.activeTab=t; render();}}
  function updateQuestion(e){{state.currentQuestion=e.target.value;}}
  function handleKeyDown(e){{if(e.key==='Enter'&&!state.loading)askQuestion();}}

  // ---- homework
  async function askQuestion(){{
    if(!state.currentQuestion.trim()||state.loading)return;
    const q=state.currentQuestion.trim();
    state.conversation.push({{type:'user',content:q}});
    state.currentQuestion=''; 
    state.loading=true; 
    render();
    try{{
      const r = await fetch('/api/hint', {{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{question:q}})}});
      const d = await r.json();
      state.conversation.push({{type:'helper', content: d.hint || 'I could not generate a hint right now.'}});
    }}catch(e){{
      state.conversation.push({{type:'helper', content:'Sorry, I had trouble. Please try again.'}});
    }}
    state.loading=false; render();
  }}
  function resetChat(){{state.conversation=[];state.currentQuestion='';render();}}

  // ---- essays
  function updateEssayText(e){{state.essayText=e.target.value;}}
  async function analyzeEssay(){{
    if(!state.essayText.trim()||state.essayLoading)return;
    state.essayLoading=true; render();
    try{{
      const r=await fetch('/api/summary',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{text:state.essayText}})}}); 
      const d=await r.json(); state.essaySummary=d.summary||'No analysis returned.';
    }}catch{{ state.essaySummary='Sorry, I had trouble analyzing that text.'; }}
    state.essayLoading=false; render();
  }}

  // ---- quizzes (FIXED generateQuiz)
  function updateQuizSource(e){{state.quizSource=e.target.value;}}
  function updateQuizCount(e){{state.quizCount=parseInt(e.target.value);}}
  function updateQuizDifficulty(level) {{ state.quizDifficulty = level; render(); }}
  async function generateQuiz(){{
    if (!state.quizSource.trim() || state.quizLoading) return;
    state.quizLoading = true;
    render();

    try {{
      const r = await fetch('/api/quiz', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{
          source: state.quizSource,
          count: state.quizCount,
          difficulty: state.quizDifficulty
        }})
      }});
      const d = await r.json();
      const qs = Array.isArray(d.questions) ? d.questions : [];
      if (!qs.length) {{
        alert('No questions generated. Please try again.');
        state.quizLoading = false;
        render();
        return;
      }}
      state.quizTitle = d.title || 'Interactive Quiz';
      state.quizQuestions = qs;
      state.quizAnswers = new Array(qs.length).fill(null);
      state.currentQuestionIndex = 0;
      state.quizStage = 'taking';
    }} catch (e) {{
      alert('Error generating quiz. Please try again.');
    }}

    state.quizLoading = false;
    render();
  }}
  function selectAnswer(i){{state.quizAnswers[state.currentQuestionIndex]=i;}}
  function previousQuestion(){{if(state.currentQuestionIndex>0){{state.currentQuestionIndex--;render();}}}}
  function nextQuestion(){{if(state.currentQuestionIndex<state.quizQuestions.length-1){{state.currentQuestionIndex++;render();}}else{{finishQuiz();}}}}
  function finishQuiz(){{let s=0;state.quizAnswers.forEach((a,i)=>{{if(a===state.quizQuestions[i].answer_index)s++;}});state.quizScore=s;state.quizStage='results';render();}}
  function retakeQuiz(){{state.quizAnswers=new Array(state.quizQuestions.length).fill(null);state.currentQuestionIndex=0;state.quizStage='taking';render();}}
  function newQuiz(){{state.quizStage='setup';state.quizSource='';state.quizQuestions=[];state.quizAnswers=[];state.currentQuestionIndex=0;render();}}

  // ---- flashcards (client-only)
  function addCard(){{
    const f = state.fcFront.trim();
    const b = state.fcBack.trim();
    if(!f || !b) return;
    state.fcCards.push({{id: Date.now() + Math.random(), front: f, back: b, flipped:false}});
    state.fcFront = ""; state.fcBack = "";
    render();
  }}
  function toggleCard(i){{state.fcCards[i].flipped = !state.fcCards[i].flipped; render();}}
  function clearCards(){{state.fcCards = []; render();}}

  // initial render
  render();
</script>
"""
    return _html(page, "StudyBot")

@app.route("/api/hint", methods=["POST"])
@login_required
def api_hint_route():
    try:
        data = request.get_json(force=True) or {}
        q = (data.get("question") or "").strip()
        if not q:
            return jsonify({"hint": "Please enter a question and I'll help guide you through it! 🙂"}), 400
        if len(q) < 3:
            return jsonify({"hint": "Could you provide a bit more detail in your question? I'd love to help you better!"}), 400

        try:
            enhanced = ai_hint(q)
        except Exception:
            enhanced = "I'm having trouble; try rephrasing your question."
        return jsonify({"hint": enhanced})
    except Exception:
        return jsonify({"hint": "I'm experiencing some technical difficulties. Please try again in a moment."}), 500

@app.route("/api/summary", methods=["POST"])
@login_required
def api_summary_route():
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"summary": "Please paste your extract 🙂"}), 400
    try:
        out = _to_text(sum_agent.run(text))
        summary_text = _strip_ansi(out) or "I couldn't analyze that—try a different extract."
    except Exception:
        summary_text = "I hit a snag analyzing that text. Please try again."
    return jsonify({"summary": summary_text})

@app.route("/api/quiz", methods=["POST"])
@login_required
def api_quiz_route():
    data = request.get_json(force=True) or {}
    src = (data.get("source") or "").strip()
    count = int(data.get("count") or 10)
    difficulty = data.get("difficulty", "medium")
    if not src:
        return jsonify({"title": "Quiz", "questions": []}), 400
    quiz = ai_quiz(src, count=count, difficulty=difficulty)
    return jsonify(quiz)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
