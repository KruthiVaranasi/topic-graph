"""The ~40 topics the synthetic eval set is built from.

Topics span typical assistant use. Some pairs are near-neighbours on purpose
(same domain, shared vocabulary) to test over-merging. `keywords` feed the
offline template generator; the LLM generator only needs `name` and `brief`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Topic:
    id: str
    name: str
    domain: str
    brief: str
    keywords: tuple[str, ...]
    neighbours: tuple[str, ...] = field(default=())


def _t(id, name, domain, brief, keywords, neighbours=()):
    return Topic(id, name, domain, brief, tuple(keywords.split()), tuple(neighbours))


TOPICS: list[Topic] = [
    # coding
    _t("python_errors", "Python errors", "coding", "debugging a Python traceback or exception",
       "python traceback exception keyerror typeerror stacktrace debugger variable function indentation nonetype attribute",
       ["python_packaging"]),
    _t("python_packaging", "Python packaging", "coding", "pip, virtualenvs, dependencies and publishing Python packages",
       "python pip virtualenv venv requirements dependency pyproject wheel poetry install version conflict",
       ["python_errors"]),
    _t("sql_queries", "SQL queries", "coding", "writing and fixing SQL queries",
       "sql query join groupby select where subquery aggregate rows postgres table count",
       ["database_design"]),
    _t("database_design", "Database schema design", "coding", "designing tables, keys and normalization",
       "schema table normalization foreign primary key index relationship postgres entity migration",
       ["sql_queries"]),
    _t("git_workflow", "Git workflow", "coding", "branches, merges, rebases and fixing git mistakes",
       "git branch merge rebase commit conflict stash remote origin checkout cherrypick"),
    _t("react_frontend", "React frontend", "coding", "React components, state and hooks",
       "react component useeffect usestate props render jsx hook frontend css typescript"),
    _t("docker_deploy", "Docker deployment", "coding", "containers, images and deploying services",
       "docker container image dockerfile compose port volume deploy kubernetes registry build"),
    _t("regex_help", "Regular expressions", "coding", "writing regex patterns",
       "regex pattern match capture group lookahead escape wildcard quantifier anchor substitution"),
    # career
    _t("resume_writing", "Resume writing", "career", "improving a CV or resume",
       "resume cv bullet achievements recruiter ats format experience skills quantify template",
       ["cover_letter"]),
    _t("cover_letter", "Cover letters", "career", "writing a cover letter for a job application",
       "cover letter recruiter application hiring manager paragraph motivation company role experience",
       ["resume_writing"]),
    _t("interview_prep", "Interview preparation", "career", "preparing for job interviews",
       "interview behavioral star method questions hiring panel recruiter mock answers weakness",
       ["salary_negotiation"]),
    _t("salary_negotiation", "Salary negotiation", "career", "negotiating an offer or a raise",
       "salary offer negotiation compensation counteroffer equity bonus recruiter raise ctc",
       ["interview_prep"]),
    _t("career_switch", "Career switch", "career", "changing careers or roles",
       "career switch transition pivot transferable skills product manager role bootcamp mentor industry"),
    # travel
    _t("schengen_visa", "Schengen visa", "travel", "applying for a Schengen visa",
       "schengen visa embassy appointment documents itinerary insurance germany france application biometrics",
       ["us_visa"]),
    _t("us_visa", "US visa", "travel", "US visa interviews and applications",
       "visa b1b2 f1 interview consulate ds160 appointment documents embassy waiver application",
       ["schengen_visa"]),
    _t("flight_booking", "Flight booking", "travel", "finding and booking flights",
       "flight airline fare layover booking airport baggage refund cheapest economy stopover"),
    _t("japan_itinerary", "Japan trip itinerary", "travel", "planning a trip to Japan",
       "japan tokyo kyoto osaka itinerary shinkansen railpass temples ryokan sightseeing days"),
    _t("packing_list", "Packing list", "travel", "what to pack for a trip",
       "packing luggage carryon clothes toiletries adapter backpack checklist weather layers"),
    # health
    _t("sleep_issues", "Sleep problems", "health", "insomnia and sleep habits",
       "sleep insomnia melatonin bedtime caffeine circadian awake tired nap routine"),
    _t("running_training", "Running training", "health", "training for a run or race",
       "running marathon pace km training plan tempo intervals shoes race mileage",
       ["strength_training"]),
    _t("strength_training", "Strength training", "health", "gym and lifting programs",
       "gym strength squat deadlift bench reps sets progressive overload muscle training plan",
       ["running_training"]),
    _t("back_pain", "Back pain", "health", "lower back pain and posture",
       "back pain lower posture stretches spine physiotherapy sciatica desk ergonomic"),
    _t("nutrition_protein", "Protein and nutrition", "health", "diet, macros and protein intake",
       "protein nutrition macros calories diet grams carbs fat intake supplements whey"),
    # finance
    _t("income_tax_india", "Indian income tax", "finance", "Indian income tax filing and regimes",
       "tax itr regime deduction 80c hra filing income slab refund form16",
       ["mutual_funds"]),
    _t("mutual_funds", "Mutual funds", "finance", "investing in mutual funds and SIPs",
       "mutual funds sip index fund nav expense ratio portfolio equity returns elss",
       ["income_tax_india"]),
    _t("credit_card_debt", "Credit card debt", "finance", "paying off credit card debt",
       "credit card debt interest balance payoff minimum emi apr statement outstanding",
       ["budgeting"]),
    _t("budgeting", "Personal budgeting", "finance", "monthly budgets and saving",
       "budget expenses savings monthly spending categories emergency fund income tracking rule",
       ["credit_card_debt"]),
    _t("home_loan", "Home loan", "finance", "mortgages and home loans",
       "home loan mortgage emi interest rate tenure prepayment downpayment bank floating"),
    # cooking
    _t("indian_curry", "Indian curry", "cooking", "cooking Indian curries and dal",
       "curry masala onion tomato garam spices dal paneer tadka ginger garlic",
       ["meal_prep"]),
    _t("baking_bread", "Baking bread", "cooking", "baking bread and sourdough",
       "bread sourdough flour dough yeast proof knead oven crust starter hydration"),
    _t("meal_prep", "Weekly meal prep", "cooking", "planning and batch-cooking meals for the week",
       "meal prep batch containers week lunches recipes fridge freezer grocery portions",
       ["indian_curry"]),
    _t("vegan_recipes", "Vegan recipes", "cooking", "plant-based cooking",
       "vegan plant tofu chickpeas lentils dairy substitute recipe cashew tempeh"),
    # study
    _t("calculus_help", "Calculus", "study", "derivatives, integrals and limits",
       "calculus derivative integral limit chain rule substitution differentiation integration function",
       ["physics_mechanics"]),
    _t("physics_mechanics", "Physics mechanics", "study", "Newtonian mechanics problems",
       "physics force velocity acceleration momentum newton friction energy projectile mass",
       ["calculus_help"]),
    _t("essay_writing", "Essay writing", "study", "structuring and writing essays",
       "essay thesis paragraph argument introduction conclusion citations outline draft structure"),
    _t("gre_prep", "GRE preparation", "study", "studying for the GRE",
       "gre verbal quant vocabulary score practice test analytical writing prep"),
    _t("ml_concepts", "Machine learning concepts", "study", "understanding ML models and training",
       "machine learning model overfitting gradient descent neural network training loss regularization"),
    # misc
    _t("wedding_speech", "Wedding speech", "life", "writing a wedding toast",
       "wedding speech toast bride groom best man story funny sister emotional"),
    _t("dog_training", "Dog training", "life", "training a puppy or dog",
       "dog puppy training leash bark treats crate commands potty walk"),
    _t("apartment_rent", "Renting an apartment", "life", "leases, landlords and rent",
       "apartment rent lease landlord deposit tenant agreement broker flat notice"),
]

BY_ID = {t.id: t for t in TOPICS}


def near_pairs() -> list[tuple[str, str]]:
    return sorted({tuple(sorted((t.id, n))) for t in TOPICS for n in t.neighbours})
