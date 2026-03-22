"""
=====================================================================
DARCRAYS  —  Step 1 : Raw Transaction Dataset Generator
=====================================================================
Generates REALISTIC raw bank statement transactions for 1 lakh users
covering 1–2 years (12–24 months) of banking history.

Output files
────────────
  data/raw_transactions.csv   — every transaction row (100 M+ rows)
  data/user_profiles.csv      — static user metadata

Run
───
  python step1_generate_raw_data.py

After this run Step 2 → Step 3.
=====================================================================
"""

import numpy as np
import pandas as pd
from numpy.random import default_rng
from datetime import datetime
import calendar, os, time

os.makedirs("data", exist_ok=True)

# ─────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────
N_USERS      = 100_000          # 1 lakh users
MONTHS_MIN   = 12               # minimum history months
MONTHS_MAX   = 24               # maximum history months
BASE_YEAR    = 2023             # history starts Jan 2023
CHUNK_SIZE   = 5_000            # write to CSV every N users (memory control)

USER_TYPES   = ["salaried_private", "salaried_govt", "shopkeeper",
                "businessman", "self_employed"]
TYPE_WEIGHTS = [0.35, 0.15, 0.20, 0.18, 0.12]
BANDS        = ["A", "B", "C", "D"]
BAND_WEIGHTS = {
    "salaried_private": [0.30, 0.35, 0.25, 0.10],
    "salaried_govt":    [0.45, 0.35, 0.15, 0.05],
    "shopkeeper":       [0.20, 0.30, 0.30, 0.20],
    "businessman":      [0.22, 0.28, 0.28, 0.22],
    "self_employed":    [0.25, 0.32, 0.27, 0.16],
}

# Transaction categories — debit (DR) or credit (CR)
CATEGORIES = {
    # ── Credits ──────────────────────────────────────────────────
    "SALARY_CREDIT":        "CR",
    "BUSINESS_CREDIT":      "CR",
    "POS_CREDIT":           "CR",   # shopkeeper incoming
    "UPI_CREDIT":           "CR",
    "CHEQUE_CREDIT":        "CR",
    "TRANSFER_CREDIT":      "CR",
    "INTEREST_CREDIT":      "CR",
    "REFUND":               "CR",
    # ── Debits ───────────────────────────────────────────────────
    "NACH_DEBIT":           "DR",   # EMI auto-debit
    "CHEQUE_BOUNCE_CHARGE": "DR",   # penalty when NACH fails
    "ELECTRICITY_BILL":     "DR",
    "MOBILE_BILL":          "DR",
    "BROADBAND_BILL":       "DR",
    "INSURANCE_PREMIUM":    "DR",
    "UPI_DEBIT":            "DR",
    "ATM_WITHDRAWAL":       "DR",
    "GROCERY_DEBIT":        "DR",
    "ENTERTAINMENT_DEBIT":  "DR",
    "ONLINE_SHOPPING":      "DR",
    "BNPL_REPAYMENT":       "DR",
    "RD_FD_INSTALLMENT":    "DR",
    "GST_PAYMENT":          "DR",   # business only
    "TRANSFER_DEBIT":       "DR",
}


# ─────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────
def _date(year: int, month: int, day: int) -> str:
    month = ((month - 1) % 12) + 1
    year  = year + ((month - 1) // 12)       # handle overflow month
    _, max_day = calendar.monthrange(year, month)
    day = min(day, max_day)
    return f"{year}-{month:02d}-{day:02d}"


def _resolve_month(base_year: int, offset_month: int):
    """Convert offset month (1–24) to (year, month)."""
    year  = base_year + (offset_month - 1) // 12
    month = ((offset_month - 1) % 12) + 1
    return year, month


# ─────────────────────────────────────────────────────────────────
#  USER PROFILE
# ─────────────────────────────────────────────────────────────────
def build_profile(user_id: int, user_type: str, band: str, rng) -> dict:
    bp = {
        "A": {"qhi": (0.78, 1.00), "qlo": (0.00, 0.18)},
        "B": {"qhi": (0.55, 0.82), "qlo": (0.12, 0.40)},
        "C": {"qhi": (0.28, 0.58), "qlo": (0.35, 0.68)},
        "D": {"qhi": (0.00, 0.32), "qlo": (0.60, 1.00)},
    }[band]
    qhi = float(rng.uniform(*bp["qhi"]))
    qlo = float(rng.uniform(*bp["qlo"]))

    income_range = {
        "salaried_private": (18_000, 200_000),
        "salaried_govt":    (25_000, 180_000),
        "shopkeeper":       (15_000, 120_000),
        "businessman":      (30_000, 800_000),
        "self_employed":    (20_000, 300_000),
    }[user_type]
    monthly_income = float(rng.uniform(*income_range))

    n_months = int(rng.integers(MONTHS_MIN, MONTHS_MAX + 1))

    # Salary credit day (consistent for good bands, random for bad)
    sal_day = int(rng.integers(1, 10))   # base: early month
    sal_day_jitter = 0 if band in ["A","B"] else int(rng.integers(0, 6))

    has_emi    = bool(rng.random() > (0.1 if band == "D" else 0.3))
    emi_amount = monthly_income * float(rng.uniform(0.08, 0.45)) if has_emi else 0.0
    emi_count  = int(rng.integers(1, 5)) if has_emi else 0
    emi_ontime_prob = 0.45 + qhi * 0.55   # prob of paying EMI on time

    is_business = user_type in ["shopkeeper", "businessman"]
    monthly_sales = monthly_income / float(rng.uniform(0.12, 0.45)) if is_business else 0.0

    return {
        "user_id":            user_id,
        "user_type":          user_type,
        "true_band":          band,
        "age":                int(rng.integers(22, 65)),
        "n_months":           n_months,
        "monthly_income":     monthly_income,
        "sal_day":            sal_day,
        "sal_day_jitter":     sal_day_jitter,
        "has_emi":            has_emi,
        "emi_amount":         emi_amount,
        "emi_count":          emi_count,
        "emi_ontime_prob":    emi_ontime_prob,
        "is_business":        is_business,
        "monthly_sales":      monthly_sales,
        "employer_tenure":    int(rng.integers(1, 20)),
        "account_vintage":    int(rng.integers(6, 240)),
        "kyc_score":          float(0.5 + qhi * 0.5),
        "co_applicant":       int(rng.integers(0, 2)),
        "qhi":                qhi,
        "qlo":                qlo,
    }


# ─────────────────────────────────────────────────────────────────
#  TRANSACTION GENERATOR  (per user)
# ─────────────────────────────────────────────────────────────────
def generate_user_transactions(p: dict, rng) -> list:
    uid     = p["user_id"]
    utype   = p["user_type"]
    inc     = p["monthly_income"]
    qhi     = p["qhi"]
    qlo     = p["qlo"]
    n_mon   = p["n_months"]
    is_biz  = p["is_business"]
    sales   = p["monthly_sales"]

    rows = []

    def add(year, month, day, category, amount):
        rows.append({
            "user_id":   uid,
            "date":      _date(year, month, day),
            "year":      year,
            "month":     month,
            "category":  category,
            "direction": CATEGORIES[category],
            "amount":    round(max(float(amount), 1.0), 2),
        })

    util_ontime_prob = 0.35 + qhi * 0.65   # probability utility paid this month
    bnpl_prob        = 0.05 + qlo * 0.40   # bad bands use BNPL more

    for mo in range(1, n_mon + 1):
        yr, mn = _resolve_month(BASE_YEAR, mo)
        _, max_d = calendar.monthrange(yr, mn)

        def rday(lo=1, hi=None):
            hi = hi or max_d
            return int(rng.integers(lo, hi + 1))

        # ── INCOME CREDITS ────────────────────────────────────────
        if utype in ["salaried_private", "salaried_govt"]:
            # Salary on consistent day ± jitter
            jitter = int(rng.integers(-p["sal_day_jitter"], p["sal_day_jitter"] + 1)) if p["sal_day_jitter"] > 0 else 0
            sal_d  = max(1, min(max_d, p["sal_day"] + jitter))
            # Govt always gets salary; private may miss for bad band
            if utype == "salaried_govt" or rng.random() > 0.08 * (1 + qlo):
                add(yr, mn, sal_d, "SALARY_CREDIT", inc * float(rng.uniform(0.97, 1.03)))

        elif is_biz:
            # Multiple POS credits (daily cash register type)
            n_pos = int(rng.integers(15, 60))
            for _ in range(n_pos):
                add(yr, mn, rday(), "POS_CREDIT",
                    sales / n_pos * float(rng.uniform(0.4, 1.8)))
            # Occasional large business transfer
            if rng.random() > 0.45:
                add(yr, mn, rday(), "BUSINESS_CREDIT",
                    sales * float(rng.uniform(0.05, 0.25)))

        else:  # self_employed
            n_cr = int(rng.integers(3, 12))
            for _ in range(n_cr):
                add(yr, mn, rday(), "UPI_CREDIT",
                    inc / n_cr * float(rng.uniform(0.6, 1.5)))
            if rng.random() > 0.4:
                add(yr, mn, rday(), "CHEQUE_CREDIT",
                    inc * float(rng.uniform(0.15, 0.5)))

        # UPI small credits (peer transfers)
        for _ in range(int(rng.integers(0, 6))):
            add(yr, mn, rday(), "UPI_CREDIT", float(rng.uniform(100, 4000)))

        # Quarterly interest
        if mn % 3 == 0:
            add(yr, mn, max_d, "INTEREST_CREDIT", float(rng.uniform(50, 2500)))

        # ── EMI / NACH DEBIT ──────────────────────────────────────
        if p["has_emi"] and p["emi_count"] > 0:
            for _ in range(p["emi_count"]):
                emi_per = p["emi_amount"] / p["emi_count"]
                if rng.random() < p["emi_ontime_prob"]:
                    add(yr, mn, min(max_d, p["sal_day"] + 3), "NACH_DEBIT", emi_per)
                else:
                    # Bounce: charge + possible retry
                    add(yr, mn, rday(), "CHEQUE_BOUNCE_CHARGE",
                        emi_per * float(rng.uniform(1.01, 1.05)))
                    if rng.random() > 0.35:   # retry payment
                        add(yr, mn, min(max_d, rday() + 3), "NACH_DEBIT", emi_per)

        # ── UTILITY BILLS ────────────────────────────────────────
        for cat, base_frac in [
            ("ELECTRICITY_BILL", 0.04),
            ("MOBILE_BILL",      0.018),
            ("BROADBAND_BILL",   0.012),
        ]:
            if rng.random() < util_ontime_prob:
                add(yr, mn, rday(1, 10), cat,
                    inc * base_frac * float(rng.uniform(0.80, 1.25)))

        # Insurance (quarterly)
        if mn % 3 == 1 and rng.random() < (0.3 + qhi * 0.5):
            add(yr, mn, rday(), "INSURANCE_PREMIUM",
                inc * float(rng.uniform(0.02, 0.08)))

        # ── SPENDING ──────────────────────────────────────────────
        # Grocery (multiple times a month)
        grocery_budget = inc * float(rng.uniform(0.07, 0.20))
        for _ in range(int(rng.integers(4, 14))):
            add(yr, mn, rday(), "GROCERY_DEBIT",
                grocery_budget / 9 * float(rng.uniform(0.5, 1.6)))

        # ATM withdrawals
        for _ in range(int(rng.integers(0, 5))):
            add(yr, mn, rday(), "ATM_WITHDRAWAL",
                float(rng.choice([500, 1000, 2000, 3000, 5000])))

        # UPI daily payments
        spend_budget = inc * (0.30 + qlo * 0.50)
        for _ in range(int(rng.integers(8, 45))):
            add(yr, mn, rday(), "UPI_DEBIT",
                spend_budget / 25 * float(rng.uniform(0.3, 2.2)))

        # Entertainment (higher for bad bands)
        if qlo > 0.05:
            for _ in range(int(rng.integers(0, max(1, int(qlo * 8))))):
                add(yr, mn, rday(), "ENTERTAINMENT_DEBIT",
                    inc * qlo * 0.04 * float(rng.uniform(0.5, 2.0)))

        # Online shopping
        if qlo > 0.05:
            for _ in range(int(rng.integers(0, max(1, int(qlo * 5))))):
                add(yr, mn, rday(), "ONLINE_SHOPPING",
                    inc * qlo * 0.05 * float(rng.uniform(0.5, 2.5)))

        # BNPL repayment (bad bands do this more)
        if rng.random() < bnpl_prob:
            add(yr, mn, rday(), "BNPL_REPAYMENT",
                inc * float(rng.uniform(0.01, 0.10)))

        # RD / FD savings (good bands do this)
        if rng.random() < (0.10 + qhi * 0.55):
            add(yr, mn, rday(1, 5), "RD_FD_INSTALLMENT",
                inc * float(rng.uniform(0.03, 0.18)))

        # GST payment (business only)
        if is_biz and rng.random() < (0.3 + qhi * 0.6):
            add(yr, mn, rday(10, 20), "GST_PAYMENT",
                sales * float(rng.uniform(0.03, 0.12)))

    return rows


# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("  DARCRAYS — Step 1 : Raw Transaction Generator")
    print(f"  {N_USERS:,} users  |  {MONTHS_MIN}–{MONTHS_MAX} months history each")
    print("=" * 62)

    rng_main = default_rng(42)
    type_idx = rng_main.choice(len(USER_TYPES), size=N_USERS, p=TYPE_WEIGHTS)
    assignments = []
    for tidx in type_idx:
        ut   = USER_TYPES[tidx]
        band = str(rng_main.choice(BANDS, p=BAND_WEIGHTS[ut]))
        assignments.append((ut, band))

    profiles   = []
    txn_buffer = []
    first_write_txn  = True
    first_write_prof = True
    t0 = time.time()
    total_txns = 0

    for i, (utype, band) in enumerate(assignments):
        seed    = hash((42, i)) & 0xFFFFFFFF
        rng     = default_rng(seed)
        profile = build_profile(i, utype, band, rng)
        txns    = generate_user_transactions(profile, rng)

        # Keep profile (without internal params)
        profiles.append({
            "user_id":           profile["user_id"],
            "user_type":         profile["user_type"],
            "true_band":         profile["true_band"],
            "age":               profile["age"],
            "n_months":          profile["n_months"],
            "monthly_income":    round(profile["monthly_income"], 2),
            "has_emi":           int(profile["has_emi"]),
            "is_business":       int(profile["is_business"]),
            "employer_tenure":   profile["employer_tenure"],
            "account_vintage":   profile["account_vintage"],
            "kyc_score":         round(profile["kyc_score"], 4),
            "co_applicant":      profile["co_applicant"],
        })

        txn_buffer.extend(txns)
        total_txns += len(txns)

        # Flush chunk to CSV
        if (i + 1) % CHUNK_SIZE == 0 or (i + 1) == N_USERS:
            df_chunk = pd.DataFrame(txn_buffer)
            df_chunk.to_csv(
                "data/raw_transactions.csv",
                mode="w" if first_write_txn else "a",
                header=first_write_txn,
                index=False,
            )
            first_write_txn = False
            txn_buffer = []

            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            eta     = (N_USERS - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1:>7,}/{N_USERS:,}]  txns={total_txns:>10,}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    # Save profiles
    pd.DataFrame(profiles).to_csv("data/user_profiles.csv", index=False)

    elapsed = time.time() - t0
    print(f"\n✅  Done in {elapsed:.0f}s")
    print(f"   data/raw_transactions.csv  — {total_txns:,} rows")
    print(f"   data/user_profiles.csv     — {N_USERS:,} users")
    print(f"\n→  Run next: python step2_feature_engineering.py")


if __name__ == "__main__":
    main()