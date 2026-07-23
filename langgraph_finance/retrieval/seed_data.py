"""Seed documents loaded into ChromaDB on first run.

`CATEGORIZATION_EXAMPLES` gives the categorization agent labeled few-shot
examples to retrieve by similarity instead of guessing from an empty prompt.
`KNOWLEDGE_SNIPPETS` gives the analysis agent grounded budgeting benchmarks
to cite instead of relying purely on LLM memory.
"""

CATEGORIZATION_EXAMPLES = [
    {"description": "WHOLE FOODS MARKET #1234", "category": "Groceries"},
    {"description": "TRADER JOE'S #56", "category": "Groceries"},
    {"description": "SAFEWAY STORE 0091", "category": "Groceries"},
    {"description": "KROGER FUEL CENTER", "category": "Groceries"},
    {"description": "STARBUCKS STORE 4021", "category": "Dining"},
    {"description": "CHIPOTLE ONLINE ORDER", "category": "Dining"},
    {"description": "DOORDASH*MCDONALDS", "category": "Dining"},
    {"description": "UBER EATS", "category": "Dining"},
    {"description": "OLIVE GARDEN #245", "category": "Dining"},
    {"description": "SHELL OIL 57443210", "category": "Transportation"},
    {"description": "CHEVRON GAS STATION", "category": "Transportation"},
    {"description": "UBER TRIP HELP.UBER.COM", "category": "Transportation"},
    {"description": "LYFT RIDE THU 4PM", "category": "Transportation"},
    {"description": "BART CLIPPER CARD RELOAD", "category": "Transportation"},
    {"description": "RENT PAYMENT - ZELLE", "category": "Housing"},
    {"description": "PROPERTY MANAGEMENT CO MONTHLY RENT", "category": "Housing"},
    {"description": "MORTGAGE PAYMENT WELLS FARGO", "category": "Housing"},
    {"description": "PG&E ELECTRIC BILL", "category": "Utilities"},
    {"description": "COMCAST XFINITY INTERNET", "category": "Utilities"},
    {"description": "WATER DEPT UTILITY PAYMENT", "category": "Utilities"},
    {"description": "AT&T WIRELESS BILL", "category": "Utilities"},
    {"description": "NETFLIX.COM", "category": "Subscriptions"},
    {"description": "SPOTIFY PREMIUM", "category": "Subscriptions"},
    {"description": "AMAZON PRIME MEMBERSHIP", "category": "Subscriptions"},
    {"description": "APPLE.COM/BILL ICLOUD STORAGE", "category": "Subscriptions"},
    {"description": "AMC THEATRES ONLINE", "category": "Entertainment"},
    {"description": "STEAM GAMES PURCHASE", "category": "Entertainment"},
    {"description": "TICKETMASTER CONCERT TICKETS", "category": "Entertainment"},
    {"description": "CVS PHARMACY #4455", "category": "Healthcare"},
    {"description": "KAISER PERMANENTE COPAY", "category": "Healthcare"},
    {"description": "WALGREENS PRESCRIPTION", "category": "Healthcare"},
    {"description": "AMAZON.COM PURCHASE", "category": "Shopping"},
    {"description": "TARGET STORE T-1122", "category": "Shopping"},
    {"description": "BEST BUY ELECTRONICS", "category": "Shopping"},
    {"description": "DELTA AIR LINES TICKET", "category": "Travel"},
    {"description": "MARRIOTT HOTEL RESERVATION", "category": "Travel"},
    {"description": "EXPEDIA.COM BOOKING", "category": "Travel"},
    {"description": "GEICO AUTO INSURANCE PREMIUM", "category": "Insurance"},
    {"description": "STATE FARM INSURANCE PAYMENT", "category": "Insurance"},
    {"description": "COURSERA SUBSCRIPTION", "category": "Education"},
    {"description": "UNIVERSITY TUITION PAYMENT", "category": "Education"},
    {"description": "GREAT CLIPS HAIRCUT", "category": "Personal Care"},
    {"description": "PLANET FITNESS MEMBERSHIP", "category": "Personal Care"},
    {"description": "OVERDRAFT FEE", "category": "Fees & Charges"},
    {"description": "ATM WITHDRAWAL FEE", "category": "Fees & Charges"},
    {"description": "MONTHLY MAINTENANCE FEE", "category": "Fees & Charges"},
    {"description": "PAYROLL DIRECT DEPOSIT", "category": "Income"},
    {"description": "ACME CORP SALARY", "category": "Income"},
    {"description": "FREELANCE PAYMENT RECEIVED", "category": "Income"},
    {"description": "TAX REFUND IRS", "category": "Income"},
    {"description": "DIVIDEND PAYMENT", "category": "Income"},
    {"description": "INTEREST PAYMENT", "category": "Income"},
    {"description": "CASHBACK REWARD", "category": "Income"},
    {"description": "VENMO CASH IN", "category": "Income"},
]

KNOWLEDGE_SNIPPETS = [
    {
        "topic": "50/30/20 rule",
        "text": (
            "The 50/30/20 budgeting rule suggests allocating roughly 50% of after-tax "
            "income to needs (housing, utilities, groceries), 30% to wants (dining, "
            "entertainment, shopping), and 20% to savings or debt repayment."
        ),
    },
    {
        "topic": "housing benchmark",
        "text": (
            "Financial planners commonly recommend keeping housing costs (rent or "
            "mortgage) at or below 30% of gross monthly income to avoid being "
            "'house poor'."
        ),
    },
    {
        "topic": "dining out benchmark",
        "text": (
            "Households that spend more than 15% of their total monthly expenditures "
            "on dining out and food delivery are generally considered to have room to "
            "cut back toward the 10-15% range typical of grocery-first households."
        ),
    },
    {
        "topic": "subscription creep",
        "text": (
            "Recurring subscription charges (streaming, software, memberships) are a "
            "common source of 'subscription creep' — the average household underestimates "
            "its combined subscription spend by 2-3x versus what recurring-charge audits reveal."
        ),
    },
    {
        "topic": "transportation benchmark",
        "text": (
            "Transportation costs (fuel, rideshare, transit, parking) exceeding 20% of "
            "monthly spend are considered high relative to the typical 15-18% U.S. average."
        ),
    },
    {
        "topic": "emergency fund",
        "text": (
            "A commonly cited emergency-fund target is 3-6 months of essential monthly "
            "expenditures held in accessible savings, scaled by income volatility."
        ),
    },
    {
        "topic": "month-over-month volatility",
        "text": (
            "A month-over-month spending increase greater than 15% in a single category "
            "is generally flagged as worth investigating, since it often signals a one-off "
            "large purchase rather than a sustained trend."
        ),
    },
    {
        "topic": "fees and charges",
        "text": (
            "Bank fees (overdraft, ATM, maintenance) are considered avoidable spend; "
            "even small recurring fees ($5-35/mo) compound to a meaningful annual cost "
            "and are usually flagged regardless of their share of total spend."
        ),
    },
]
