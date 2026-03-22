"""
=====================================================================
DARCRAYS  —  Step 2 : Feature Engineering  (UPDATED FLOW)
=====================================================================

FLOW:
  1 Lakh users → Feature Engineering → SAVE as all_users.csv
                                              ↓
                                   GMM trains on full 1 lakh  ✅
                                              ↓
                               1 lakh mein 30% missing add karo
                                              ↓
                               GMM se missing fill karo
                                              ↓
                         QA: filled vs original compare karo  ✅
                                              ↓
                         XGBoost trains on full 1 lakh  ✅
                                              ↓
                    Alag 10k NEW test users → GMM impute → Predict

Output files:
  data/all_users.csv              — 1 lakh users | complete | with labels
  data/all_users_with_missing.csv — 1 lakh users | 30% missing | with labels
  data/all_users_ground_truth.csv — original values (for QA comparison)
  data/test.csv                   — 10k NEW users | 25% missing | NO labels
  data/test_ground_truth.csv      — test original values (for final eval)

Run:
  python Feature_engineering.py
=====================================================================
"""

import numpy as np
import pandas as pd
from numpy.random import default_rng
import os, time

os.makedirs("data", exist_ok=True)

# ─────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────
# Structural zero columns — always 0 for non-business users, NEVER imputed
STRUCTURAL_ZERO_COLS = [
    "monthly_avg_business_credit",
    "pos_txn_count_monthly_avg",
    "gst_payment_count",
    "business_account_avg_balance",
    "trade_credit_utilisation",
    "receivables_turnover_days",
]

META_COLS   = ["user_id", "user_type"]
TARGET_COLS = ["credit_score", "risk_band"]

SCORE_WEIGHTS = {
    "income":   0.20,
    "balance":  0.15,
    "spending": 0.15,
    "emi":      0.20,
    "bills":    0.15,
    "savings":  0.07,
    "digital":  0.04,
    "business": 0.05,
    "relation": 0.03,
    "bnpl":     0.02,
}
WEIGHT_NORM = sum(SCORE_WEIGHTS.values())  # 1.06


# ─────────────────────────────────────────────────────────────────
#  FEATURE ENGINEERING  (per user)
# ─────────────────────────────────────────────────────────────────
def engineer_features(uid, txns, profile):
    if txns.empty:
        return None

    utype = profile["user_type"]
    n_mon = int(profile["n_months"])

    def msum(cat):
        s = txns[txns["category"] == cat].groupby("month")["amount"].sum()
        return s.reindex(range(1, n_mon + 1), fill_value=0.0)

    def mcount(cat):
        s = txns[txns["category"] == cat].groupby("month")["amount"].count()
        return s.reindex(range(1, n_mon + 1), fill_value=0)

    cr = txns[txns["direction"] == "CR"].groupby("month")["amount"].sum().reindex(range(1, n_mon+1), fill_value=0.0)
    dr = txns[txns["direction"] == "DR"].groupby("month")["amount"].sum().reindex(range(1, n_mon+1), fill_value=0.0)

    sal   = msum("SALARY_CREDIT");   pos   = msum("POS_CREDIT")
    biz   = msum("BUSINESS_CREDIT"); upi_c = msum("UPI_CREDIT")
    chq_c = msum("CHEQUE_CREDIT");   nach  = msum("NACH_DEBIT")
    bounce= mcount("CHEQUE_BOUNCE_CHARGE")
    elec  = msum("ELECTRICITY_BILL"); mob  = msum("MOBILE_BILL")
    bb    = msum("BROADBAND_BILL");   ins  = msum("INSURANCE_PREMIUM")
    groc  = msum("GROCERY_DEBIT");    atm  = msum("ATM_WITHDRAWAL")
    upi_d = msum("UPI_DEBIT");        ent  = msum("ENTERTAINMENT_DEBIT")
    shop  = msum("ONLINE_SHOPPING");  bnpl = msum("BNPL_REPAYMENT")
    rdfd  = msum("RD_FD_INSTALLMENT"); gst = msum("GST_PAYMENT")

    tcr = float(cr.sum()) + 1e-6
    tdr = float(dr.sum()) + 1e-6

    # Main income series by user type
    if utype in ["salaried_private", "salaried_govt"]:
        main_inc = sal
    elif profile["is_business"]:
        main_inc = pos + biz
    else:
        main_inc = upi_c + chq_c

    avg_inc = float(main_inc.mean()) + 1e-6

    # Salary day consistency
    stxn = txns[txns["category"] == "SALARY_CREDIT"].copy()
    if not stxn.empty:
        stxn["day"] = pd.to_datetime(stxn["date"]).dt.day
        std_d    = float(stxn["day"].std()) if len(stxn) > 1 else 0.0
        sal_cons = float(np.clip(1.0 - std_d / 10.0, 0, 1))
    else:
        sal_cons = 0.0

    h1 = float(main_inc.iloc[:n_mon//2].mean()) + 1e-6
    h2 = float(main_inc.iloc[n_mon//2:].mean()) + 1e-6
    inc_growth  = float(np.clip((h2 - h1) / h1, -0.5, 1.0))
    inc_cv      = float(main_inc.std() / avg_inc)
    sec_ratio   = float(np.clip(upi_c.sum() / tcr, 0, 0.6)) \
                  if utype in ["salaried_private", "salaried_govt"] else 0.0
    io_ratio    = float(np.clip(tcr / tdr, 0.5, 4.0))

    # Balance proxy
    bal      = (cr - dr).cumsum()
    avg_bal  = float(bal.mean())
    bal_vol  = float(bal.std())
    neg_mons = int((bal < 0).sum())
    low_mons = int((bal < 5000).sum())
    bal_util = float(np.clip(tdr / tcr, 0.05, 1.5))

    # Spending
    groc_r = float(np.clip(groc.sum() / tcr, 0, 0.5))
    util_r = float(np.clip((elec+mob+bb).sum() / tcr, 0, 0.35))
    ent_r  = float(np.clip(ent.sum()  / tcr, 0, 0.35))
    atm_r  = float(np.clip(atm.sum()  / tcr, 0, 0.6))
    shop_r = float(np.clip(shop.sum() / tcr, 0, 0.35))
    spd_r  = float(np.clip(tdr / tcr, 0.2, 1.5))
    upi_m  = float(mcount("UPI_DEBIT").mean())

    # EMI
    avg_nach  = float(nach.mean())
    emi_ratio = float(np.clip(avg_nach / avg_inc, 0, 0.8))
    b_total   = int(bounce.sum())
    nach_att  = int(txns[txns["category"].isin(["NACH_DEBIT","CHEQUE_BOUNCE_CHARGE"])].shape[0])
    nach_ok   = int(txns[txns["category"] == "NACH_DEBIT"].shape[0])
    si_rate   = float(np.clip(nach_ok / (nach_att + 1e-6), 0, 1))
    has_emi   = bool(profile["has_emi"])
    loan_trk  = float(np.clip(0.3 + float(profile["kyc_score"]) * 0.7, 0, 1)) \
                if has_emi else float(np.clip(0.7 + float(profile["kyc_score"]) * 0.3, 0, 1))

    # Bills
    elec_ok = int((elec > 0).sum()); mob_ok = int((mob > 0).sum())
    bb_ok   = int((bb   > 0).sum()); ins_ok = int((ins > 0).sum())
    util_cons = float(np.clip((elec_ok+mob_ok+bb_ok) / (n_mon * 3), 0, 1))

    # Savings
    net_sav  = float(np.clip((tcr - tdr) / tcr, -0.3, 0.7))
    rdfd_r   = float(np.clip(rdfd.sum() / tcr, 0, 0.45))
    rdfd_cnt = int(np.clip((rdfd > 0).sum() // 4, 0, 5))
    sav_txn  = int(np.clip((rdfd > 0).sum() * 2, 0, 24))

    # Digital
    qhi_p    = max(float(profile["kyc_score"]) - 0.5, 0)
    nb_days  = float(np.clip(5 + qhi_p * 40, 0, 25))
    app_sess = float(np.clip(3 + qhi_p * 114, 0, 60))
    upi_auto = int(nach_ok > 0)
    dc_txn   = float(np.clip(txns[txns["category"].isin(
        ["GROCERY_DEBIT","ENTERTAINMENT_DEBIT","ONLINE_SHOPPING"])].shape[0], 0, 200))

    # BNPL
    bnpl_mon = int((bnpl > 0).sum())
    bnpl_tot = float(bnpl.sum())
    bnpl_ir  = float(np.clip(bnpl_tot / tcr, 0, 0.5))
    bnpl_ot  = float(np.clip(1.0 - bnpl_ir * 3, 0, 1))

    # Business (structural zeros for non-business)
    is_biz = bool(profile["is_business"])
    if is_biz:
        avg_biz = float((pos + biz).mean())
        gst_cnt = int((gst > 0).sum())
        trade_u = float(np.clip(0.1 + (1 - float(profile["kyc_score"])) * 0.7, 0, 0.9))
        recv_d  = float(np.clip(5 + (1 - float(profile["kyc_score"])) * 115, 5, 120))
        pos_m   = float(mcount("POS_CREDIT").mean())
        biz_bal = float(np.clip(avg_bal * 1.5, 0, 5e6))
    else:
        avg_biz = pos_m = gst_cnt = trade_u = recv_d = biz_bal = 0.0

    vintage   = int(profile["account_vintage"])
    rel_score = float(np.clip(float(profile["kyc_score"]) * 0.8 + 0.1, 0, 1))

    return {
        # Income (9)
        "monthly_avg_income":              round(avg_inc, 2),
        "salary_credit_count":             int(stxn.shape[0]) if utype in ["salaried_private","salaried_govt"] else int(txns[txns["direction"]=="CR"].shape[0]),
        "income_variability_cv":           round(float(np.clip(inc_cv, 0, 2)), 4),
        "salary_day_consistency":          round(sal_cons, 4),
        "employer_tenure_years":           float(profile["employer_tenure"]),
        "income_growth_yoy":               round(inc_growth, 4),
        "secondary_income_ratio":          round(sec_ratio, 4),
        "total_annual_inflow":             round(float(cr.sum()), 2),
        "inflow_outflow_ratio":            round(io_ratio, 4),
        # Balance (7)
        "avg_monthly_balance":             round(max(avg_bal, 0), 2),
        "min_balance_breach_count":        int(np.clip(low_mons * 1.5, 0, 24)),
        "balance_below_5k_months":         int(np.clip(low_mons, 0, n_mon)),
        "avg_eom_balance":                 round(max(avg_bal, 0), 2),
        "balance_volatility":              round(float(np.clip(bal_vol, 0, 1e7)), 2),
        "negative_balance_months":         int(np.clip(neg_mons, 0, n_mon)),
        "balance_utilisation_ratio":       round(float(np.clip(bal_util, 0.05, 1.5)), 4),
        # Spending (10)
        "monthly_avg_debit":               round(float(dr.mean()), 2),
        "debit_txn_count_monthly":         round(float(txns[txns["direction"]=="DR"].shape[0] / n_mon), 2),
        "grocery_spend_ratio":             round(groc_r, 4),
        "utility_spend_ratio":             round(util_r, 4),
        "entertainment_spend_ratio":       round(ent_r, 4),
        "atm_withdrawal_ratio":            round(atm_r, 4),
        "upi_txn_count_monthly":           round(upi_m, 2),
        "online_shopping_ratio":           round(shop_r, 4),
        "spend_to_income_ratio":           round(float(np.clip(spd_r, 0.2, 1.5)), 4),
        "weekend_spend_ratio":             round(float(np.clip(0.20 + float(np.random.uniform(0, 0.25)), 0.1, 0.55)), 4),
        # EMI (7)
        "active_emi_count":                int(has_emi) * int(max(1, int(np.random.randint(1, 5)))) if has_emi else 0,
        "monthly_emi_obligation":          round(avg_nach, 2),
        "emi_to_income_ratio":             round(emi_ratio, 4),
        "emi_bounce_count":                int(np.clip(b_total, 0, 12)),
        "emi_paid_ontime_ratio":           round(si_rate, 4),
        "nach_mandate_active":             int(has_emi),
        "loan_repayment_track_score":      round(loan_trk, 4),
        # Bills (7)
        "electricity_paid_ontime_months":  int(np.clip(elec_ok, 0, n_mon)),
        "mobile_paid_ontime_months":       int(np.clip(mob_ok,  0, n_mon)),
        "broadband_paid_ontime_months":    int(np.clip(bb_ok,   0, n_mon)),
        "insurance_paid_months":           int(np.clip(ins_ok,  0, n_mon)),
        "utility_payment_consistency":     round(util_cons, 4),
        "cheque_bounce_count":             int(np.clip(b_total, 0, 10)),
        "standing_instruction_rate":       round(si_rate, 4),
        # Savings (5)
        "rd_fd_count_active":              int(rdfd_cnt),
        "savings_txn_count":               int(sav_txn),
        "net_savings_rate":                round(float(np.clip(net_sav, -0.3, 0.7)), 4),
        "investment_inflow_ratio":         round(rdfd_r, 4),
        "sweep_utilisation":               round(float(np.clip(float(np.random.uniform(0, 1)), 0, 1)), 4),
        # Digital (4)
        "netbanking_days_monthly":         round(nb_days, 2),
        "app_sessions_monthly":            round(app_sess, 2),
        "upi_autopay_mandates":            int(upi_auto),
        "debit_card_txn_count":            round(dc_txn, 0),
        # BNPL (3)
        "bnpl_usage_months":               int(np.clip(bnpl_mon, 0, n_mon)),
        "bnpl_repayment_ratio":            round(bnpl_ot, 4),
        "bnpl_to_income_ratio":            round(bnpl_ir, 4),
        # Business (6) — structural zeros for non-business
        "monthly_avg_business_credit":     round(float(avg_biz), 2),
        "pos_txn_count_monthly_avg":       round(float(pos_m), 2),
        "gst_payment_count":               int(gst_cnt),
        "business_account_avg_balance":    round(float(biz_bal), 2),
        "trade_credit_utilisation":        round(float(trade_u), 4),
        "receivables_turnover_days":       round(float(recv_d), 2),
        # Profile (5)
        "age":                             int(profile["age"]),
        "account_vintage_months":          int(np.clip(vintage, 6, 240)),
        "kyc_completeness_score":          round(float(np.clip(float(profile["kyc_score"]), 0.5, 1.0)), 4),
        "co_applicant_flag":               int(profile["co_applicant"]),
        "existing_relationship_score":     round(rel_score, 4),
        "history_months":                  int(n_mon),
    }


# ─────────────────────────────────────────────────────────────────
#  CREDIT SCORE FORMULA
# ─────────────────────────────────────────────────────────────────
def compute_score(f):
    def c(x): return float(np.clip(x, 0.0, 1.0))
    h = int(f["history_months"]) + 1e-6

    inc  = c(0.25*c(f["salary_day_consistency"]) +
             0.20*c(f["income_growth_yoy"]/0.40 + 0.375) +
             0.20*c(1 - f["income_variability_cv"]/1.0) +
             0.20*c((f["inflow_outflow_ratio"] - 0.85)/1.65) +
             0.15*c(f["net_savings_rate"]/0.55 + 0.18))

    bal  = c(0.30*c(1 - f["balance_below_5k_months"]/h) +
             0.25*c(1 - f["negative_balance_months"]/h) +
             0.25*c(1 - min(f["balance_utilisation_ratio"], 1.0)) +
             0.20*c(1 - f["min_balance_breach_count"]/24))

    spd  = c(0.35*c(1 - f["spend_to_income_ratio"]) +
             0.30*c(1 - f["entertainment_spend_ratio"]/0.25) +
             0.20*c(1 - f["online_shopping_ratio"]/0.30) +
             0.15*c(f["net_savings_rate"]/0.55 + 0.18))

    emi  = c(0.35*c(f["emi_paid_ontime_ratio"]) +
             0.30*c(f["loan_repayment_track_score"]) +
             0.20*c(1 - f["emi_bounce_count"]/12) +
             0.15*c(1 - min(f["emi_to_income_ratio"]/0.55, 1.0)))

    bill = c(0.22*c(f["electricity_paid_ontime_months"]/h) +
             0.18*c(f["mobile_paid_ontime_months"]/h) +
             0.22*c(f["utility_payment_consistency"]) +
             0.22*c(f["standing_instruction_rate"]) +
             0.16*c(1 - f["cheque_bounce_count"]/10))

    sav  = c(0.40*c(f["net_savings_rate"]/0.55 + 0.18) +
             0.35*c(f["investment_inflow_ratio"]/0.40) +
             0.25*c(f["rd_fd_count_active"]/5))

    dig  = c(0.50*c(f["app_sessions_monthly"]/60) +
             0.30*c(f["netbanking_days_monthly"]/25) +
             0.20*c(f["upi_autopay_mandates"]/8))

    biz  = c(0.35*c(f["gst_payment_count"]/h) +
             0.35*c(1 - f["trade_credit_utilisation"]) +
             0.30*c(1 - f["receivables_turnover_days"]/120)) \
           if f["monthly_avg_business_credit"] > 0 else 0.5

    rel  = c(0.50*c(f["account_vintage_months"]/240) +
             0.30*c(f["existing_relationship_score"]) +
             0.20*c(f["kyc_completeness_score"]))

    bnp  = c(0.60*c(f["bnpl_repayment_ratio"]) +
             0.40*c(1 - f["bnpl_to_income_ratio"]/0.3))

    comp = (SCORE_WEIGHTS["income"]   * inc  +
            SCORE_WEIGHTS["balance"]  * bal  +
            SCORE_WEIGHTS["spending"] * spd  +
            SCORE_WEIGHTS["emi"]      * emi  +
            SCORE_WEIGHTS["bills"]    * bill +
            SCORE_WEIGHTS["savings"]  * sav  +
            SCORE_WEIGHTS["digital"]  * dig  +
            SCORE_WEIGHTS["business"] * biz  +
            SCORE_WEIGHTS["relation"] * rel  +
            SCORE_WEIGHTS["bnpl"]     * bnp) / WEIGHT_NORM

    return int(np.clip(300 + comp * 600, 300, 900))


def assign_band(s):
    return "A" if s >= 750 else ("B" if s >= 650 else ("C" if s >= 550 else "D"))


# ─────────────────────────────────────────────────────────────────
#  INTRODUCE MISSINGNESS  (skip structural zeros)
# ─────────────────────────────────────────────────────────────────
def apply_missingness(df, ratio, feat_cols, seed):
    """
    Introduce missing values row by row.
    Structural zero cols (business features) are NEVER made missing.
    """
    rng       = default_rng(seed)
    df        = df.copy()
    imputable = [c for c in feat_cols if c not in STRUCTURAL_ZERO_COLS]

    for i in range(len(df)):
        n_miss    = max(1, int(len(imputable) * ratio))
        miss_cols = rng.choice(imputable, size=n_miss, replace=False)
        for col in miss_cols:
            df.iloc[i, df.columns.get_loc(col)] = np.nan
    return df


# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("  DARCRAYS — Step 2 : Feature Engineering (UPDATED)")
    print("  Flow: 1L users → GMM (full 1L) → Missing → Impute")
    print("        → QA Check → XGBoost (full 1L) → Test (10k new)")
    print("=" * 62)

    # ── Load raw data ─────────────────────────────────────────────
    print("\nLoading raw_transactions.csv …")
    t0      = time.time()
    txn     = pd.read_csv("data/raw_transactions.csv",
                          dtype={"user_id": int, "month": int, "amount": float})
    prof_df = pd.read_csv("data/user_profiles.csv")
    print(f"  Transactions : {len(txn):,} rows in {time.time()-t0:.1f}s")
    print(f"  Profiles     : {len(prof_df):,} users")

    profiles  = prof_df.set_index("user_id").to_dict("index")
    txn_grp   = txn.groupby("user_id")

    # ── Engineer features for ALL 1 lakh users ───────────────────
    print("\nEngineering features for all 1 lakh users …")
    t0   = time.time()
    rows = []

    for i, uid in enumerate(profiles.keys()):
        if uid not in txn_grp.groups:
            continue
        feat = engineer_features(uid, txn_grp.get_group(uid), profiles[uid])
        if feat is None:
            continue

        feat["user_id"]      = uid
        feat["user_type"]    = profiles[uid]["user_type"]
        feat["credit_score"] = compute_score(feat)
        feat["risk_band"]    = assign_band(feat["credit_score"])
        rows.append(feat)

        if (i + 1) % 10_000 == 0:
            print(f"  {i+1:>7,}/{len(profiles):,}  ({time.time()-t0:.0f}s)")

    df_all = pd.DataFrame(rows)

    # Column ordering
    feat_cols   = [c for c in df_all.columns if c not in META_COLS + TARGET_COLS]
    df_all      = df_all[feat_cols + META_COLS + TARGET_COLS]

    print(f"\n✅ Feature engineering done in {time.time()-t0:.0f}s")
    print(f"   Shape       : {df_all.shape}")
    print(f"   Score range : {df_all['credit_score'].min()} – {df_all['credit_score'].max()}")
    print(f"   Mean score  : {df_all['credit_score'].mean():.0f}")
    print(f"   Bands       :\n{df_all['risk_band'].value_counts().sort_index().to_string()}")
    print(f"   User types  :\n{df_all['user_type'].value_counts().to_string()}")

    # ── Save complete 1 lakh dataset ─────────────────────────────
    # This is what GMM and XGBoost will train on
    df_all.to_csv("data/all_users.csv", index=False)
    df_all.to_csv("data/all_users_ground_truth.csv", index=False)  # backup for QA
    print(f"\n✅ Saved → data/all_users.csv  ({len(df_all):,} rows, complete)")

    # ── Detect + remove leakage ───────────────────────────────────
    corr         = df_all[feat_cols + ["credit_score"]].select_dtypes(include=np.number).corr()["credit_score"].drop("credit_score")
    leak_features= corr[corr.abs() > 0.90].index.tolist()
    print(f"\n   Leak features (|corr|>0.90): {leak_features}")
    feat_cols_clean = [c for c in feat_cols if c not in leak_features]
    print(f"   Clean feature count: {len(feat_cols_clean)}")

    # ── Add 30% missingness to the SAME 1 lakh users ─────────────
    print("\nAdding 30% missingness to 1 lakh users …")
    df_with_missing = apply_missingness(df_all, 0.30, feat_cols_clean, seed=42)
    df_with_missing.to_csv("data/all_users_with_missing.csv", index=False)

    miss_avg = df_with_missing[feat_cols_clean].isnull().mean().mean() * 100
    print(f"✅ Saved → data/all_users_with_missing.csv")
    print(f"   Avg missing : {miss_avg:.1f}%")

    # ── Separate 10k NEW test users ───────────────────────────────
    # These are users XGBoost has NEVER seen — simulates new DB users
    print("\nCreating separate 10k test users from end of dataset …")
    # Last 10k users are test — they were NOT in training
    # (In real production, these would be brand new users coming to bank)
    df_test_complete = df_all.tail(10_000).reset_index(drop=True)
    df_test_gt       = df_test_complete.copy()

    # Add 25% missingness to test (simulates incomplete bank data)
    df_test_missing  = apply_missingness(df_test_complete, 0.25, feat_cols_clean, seed=300)
    # Remove labels — test set has NO credit score (to be predicted)
    df_test_nolabel  = df_test_missing.drop(columns=["credit_score", "risk_band"])

    df_test_nolabel.to_csv("data/test.csv", index=False)
    df_test_gt.to_csv("data/test_ground_truth.csv", index=False)

    test_miss_avg = df_test_nolabel[feat_cols_clean].isnull().mean().mean() * 100
    print(f"✅ Saved → data/test.csv  ({len(df_test_nolabel):,} rows, ~{test_miss_avg:.1f}% missing, no labels)")
    print(f"✅ Saved → data/test_ground_truth.csv  (for final eval)")

    # ── Save feature column list ──────────────────────────────────
    import json
    with open("data/feat_cols_clean.json", "w") as f:
        json.dump(feat_cols_clean, f)
    print(f"\n✅ Saved → data/feat_cols_clean.json  ({len(feat_cols_clean)} features)")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  SUMMARY")
    print("=" * 62)
    print(f"  data/all_users.csv              — {len(df_all):,} users | complete | GMM + XGBoost train")
    print(f"  data/all_users_with_missing.csv — {len(df_with_missing):,} users | 30% missing | GMM impute + QA")
    print(f"  data/all_users_ground_truth.csv — {len(df_all):,} users | original values for QA")
    print(f"  data/test.csv                   — {len(df_test_nolabel):,} users | 25% missing | predict only")
    print(f"  data/test_ground_truth.csv      — {len(df_test_gt):,} users | for final accuracy check")
    print(f"\n  Feature columns : {len(feat_cols_clean)}")
    print(f"\n→  Run next: jupyter notebook DARCRAYS_Complete_Pipeline.ipynb")
    print(f"   OR       : python step3_train_and_predict.py")


if __name__ == "__main__":
    main()