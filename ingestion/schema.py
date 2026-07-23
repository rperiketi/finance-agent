"""
Shared column aliases and name normalisation for transaction CSVs.
"""

import re

DATE_ALIASES = [
    "date", "transaction_date", "trans_date", "txn_date", "value_date",
    "posting_date", "posted_date", "booked_date", "settlement_date",
    "cleared_date", "process_date", "dt", "time", "datetime",
]

DESC_ALIASES = [
    "description", "merchant", "narration", "details", "particulars",
    "transaction_details", "trans_desc", "memo", "payee", "name",
    "counterparty", "vendor", "title", "detail", "note", "notes",
]

# Single-column amount (net or signed)
AMOUNT_ALIASES = [
    "amount", "transaction_amount", "trans_amount", "amt", "value",
    "net_amount", "total", "sum", "transaction_value", "payment_amount",
    "withdrawal_amt", "deposit_amt", "debit", "credit",
    "withdrawalamt", "depositamt", "withdrawal", "deposit",
]

DEBIT_COLUMN_ALIASES = [
    "withdrawal_amt", "withdrawalamt", "withdrawal", "debit", "dr",
    "debit_amt", "outflow", "payment_out",
]

CREDIT_COLUMN_ALIASES = [
    "deposit_amt", "depositamt", "deposit", "credit", "cr",
    "credit_amt", "inflow", "payment_in",
]

TYPE_ALIASES = [
    "type", "transaction_type", "trans_type", "txn_type", "dc",
    "debit_credit", "dr_cr",
]

CATEGORY_ALIASES = [
    "category", "categories", "cat", "transaction_category", "txn_category",
    "spend_category", "merchant_category", "classification", "category_name",
]


def _normalise_col_name(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"[.\-/]", "", name)
    name = re.sub(r"[\s_]+", "_", name)
    return name.rstrip("_")
