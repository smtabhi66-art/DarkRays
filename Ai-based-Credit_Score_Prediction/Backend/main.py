"""
DARCRAYS — Credit Scoring API
FastAPI backend that loads trained models and serves predictions.

Run with:
    uvicorn main:app --reload --port 8000

Requires: fastapi uvicorn joblib numpy pandas scikit-learn xgboost shap
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import joblib, shap, os, json
from datetime import datetime

# ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DARCRAYS Credit Scoring API",
    description="AI-Powered Alternate Credit Scoring — Barclays Hackathon",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────
#  LOAD MODELS (once at startup)
# ─────────────────────────────────────────────────────────────────
MODELS = {}

@app.on_event("startup")
def load_models():
  # 🔥 FIXED PATH (Machine_Learning folder)
  BASE_PATH = os.path.join(os.path.dirname(__file__), "../Machine_Learning")


  required = [
      "gmm_model.pkl",
      "scaler.pkl",
      "xgb_model.pkl",
      "label_encoder.pkl",
      "feature_cols.pkl"
  ]

  missing = [f for f in required if not os.path.exists(os.path.join(BASE_PATH, f))]

  if missing:
      print(f"❌ Missing model files: {missing}")
      print("👉 Check Machine_Learning folder OR run notebook again")
      return

  MODELS["gmm"]     = joblib.load(os.path.join(BASE_PATH, "gmm_model.pkl"))
  MODELS["scaler"]  = joblib.load(os.path.join(BASE_PATH, "scaler.pkl"))
  MODELS["xgb"]     = joblib.load(os.path.join(BASE_PATH, "xgb_model.pkl"))
  MODELS["le"]      = joblib.load(os.path.join(BASE_PATH, "label_encoder.pkl"))
  MODELS["feats"]   = joblib.load(os.path.join(BASE_PATH, "feature_cols.pkl"))

  MODELS["explainer"] = shap.TreeExplainer(MODELS["xgb"])

  print("✅ Models loaded successfully from Machine_Learning/")
 


# @app.on_event("startup")
# def load_models():
#     required = ["gmm_model.pkl", "scaler.pkl", "xgb_model.pkl",
#                 "label_encoder.pkl", "feature_cols.pkl"]
#     missing = [f for f in required if not os.path.exists(f)]
#     if missing:
#         print(f"⚠️  Missing model files: {missing}")
#         print("   Run the Jupyter notebook first to train and save models!")
#         return
#     MODELS["gmm"]     = joblib.load("gmm_model.pkl")
#     MODELS["scaler"]  = joblib.load("scaler.pkl")
#     MODELS["xgb"]     = joblib.load("xgb_model.pkl")
#     MODELS["le"]      = joblib.load("label_encoder.pkl")
#     MODELS["feats"]   = joblib.load("feature_cols.pkl")
#     MODELS["explainer"] = shap.TreeExplainer(MODELS["xgb"])
#     print("✅ All models loaded!")

# ─────────────────────────────────────────────────────────────────
#  IMPUTATION
# ─────────────────────────────────────────────────────────────────
def gmm_impute_single(row_dict: dict, feat_cols: list) -> dict:
    gmm    = MODELS["gmm"]
    scaler = MODELS["scaler"]
    X      = np.array([row_dict.get(c, np.nan) for c in feat_cols], dtype=float)
    mask   = np.isnan(X)
    if not mask.any():
        return row_dict

    obs      = np.where(~mask)[0]
    miss_idx = np.where(mask)[0]
    mu_orig  = scaler.inverse_transform(gmm.means_)

    if len(obs) == 0:
        X[miss_idx] = mu_orig[:, miss_idx].mean(axis=0)
    else:
        tmp = X.copy()
        tmp[miss_idx] = mu_orig[:, miss_idx].mean(axis=0)
        tmp_s = scaler.transform(tmp.reshape(1, -1))[0]
        log_w = np.log(gmm.weights_ + 1e-10)
        lp    = log_w.copy()
        for k in range(gmm.n_components):
            diff = tmp_s[obs] - gmm.means_[k, obs]
            var  = gmm.covariances_[k, obs] + 1e-6
            lp[k] += -0.5 * np.sum(diff**2 / var + np.log(2 * np.pi * var))
        lp -= lp.max()
        probs = np.exp(lp); probs /= probs.sum()
        X[miss_idx] = probs @ mu_orig[:, miss_idx]

    return {c: float(X[i]) for i, c in enumerate(feat_cols)}

def to_band(score: int) -> str:
    if score >= 750: return "A"
    if score >= 650: return "B"
    if score >= 550: return "C"
    return "D"

def band_to_decision(band: str) -> str:
    return {
        "A": "AUTO_APPROVE",
        "B": "APPROVE_WITH_CONDITIONS",
        "C": "MANUAL_REVIEW",
        "D": "REJECT"
    }[band]

def band_to_label(band: str) -> str:
    return {
        "A": "Low Risk",
        "B": "Moderate Risk",
        "C": "High Risk",
        "D": "Very High Risk"
    }[band]

# ─────────────────────────────────────────────────────────────────
#  REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────
class UserInput(BaseModel):
    # Required
    user_type: str = Field(..., description="salaried_private | salaried_govt | shopkeeper | businessman | self_employed")

    # Income features
    monthly_avg_salary_credit:     Optional[float] = None
    salary_credit_count_12m:       Optional[float] = None
    income_variability_cv:         Optional[float] = None
    salary_day_consistency:        Optional[float] = None
    employer_tenure_years:         Optional[float] = None
    income_growth_yoy:             Optional[float] = None
    secondary_income_credit_ratio: Optional[float] = None
    total_annual_inflow:           Optional[float] = None
    inflow_outflow_ratio_12m:      Optional[float] = None

    # Balance features
    avg_monthly_balance:           Optional[float] = None
    min_balance_breach_count_12m:  Optional[float] = None
    balance_below_5k_days_12m:     Optional[float] = None
    avg_eom_balance:               Optional[float] = None
    balance_volatility_std:        Optional[float] = None
    negative_balance_days_12m:     Optional[float] = None
    balance_utilisation_ratio:     Optional[float] = None

    # Spending
    monthly_avg_debit_amount:      Optional[float] = None
    debit_txn_count_monthly_avg:   Optional[float] = None
    grocery_spend_ratio:           Optional[float] = None
    utility_spend_ratio:           Optional[float] = None
    entertainment_spend_ratio:     Optional[float] = None
    atm_withdrawal_ratio:          Optional[float] = None
    upi_txn_count_monthly_avg:     Optional[float] = None
    online_shopping_spend_ratio:   Optional[float] = None
    spend_to_income_ratio:         Optional[float] = None
    weekend_spend_ratio:           Optional[float] = None

    # EMI / Loan
    active_emi_count:              Optional[float] = None
    total_emi_monthly_obligation:  Optional[float] = None
    emi_to_income_ratio:           Optional[float] = None
    emi_bounce_count_12m:          Optional[float] = None
    emi_paid_on_time_ratio:        Optional[float] = None
    nach_mandate_active:           Optional[float] = None
    loan_repayment_track_score:    Optional[float] = None

    # Bills
    electricity_bill_paid_ontime_12m:  Optional[float] = None
    mobile_bill_paid_ontime_12m:       Optional[float] = None
    broadband_paid_ontime_12m:         Optional[float] = None
    insurance_premium_paid_12m:        Optional[float] = None
    utility_payment_consistency_score: Optional[float] = None
    cheque_bounce_count_12m:           Optional[float] = None
    standing_instruction_success_rate: Optional[float] = None

    # Savings
    rd_fd_count_active:            Optional[float] = None
    savings_txn_count_12m:         Optional[float] = None
    net_savings_rate:              Optional[float] = None
    investment_credit_ratio:       Optional[float] = None
    sweep_account_utilisation:     Optional[float] = None

    # Digital
    netbanking_login_days_monthly_avg: Optional[float] = None
    mobile_app_sessions_monthly_avg:   Optional[float] = None
    upi_autopay_mandates_active:       Optional[float] = None
    debit_card_txn_count_12m:          Optional[float] = None

    # Business
    monthly_avg_business_credit:   Optional[float] = None
    pos_txn_count_monthly_avg:     Optional[float] = None
    gst_payment_count_12m:         Optional[float] = None
    business_account_avg_balance:  Optional[float] = None
    trade_credit_utilisation:      Optional[float] = None
    receivables_turnover_days:     Optional[float] = None

    # Profile
    age:                           Optional[float] = None
    account_vintage_months:        Optional[float] = None
    kyc_completeness_score:        Optional[float] = None
    co_applicant_flag:             Optional[float] = None
    existing_relationship_score:   Optional[float] = None

# ─────────────────────────────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "DARCRAYS Credit Scoring API",
        "status": "running",
        "models_loaded": len(MODELS) > 0,
        "endpoints": ["/predict", "/health", "/features", "/sample-input"]
    }

@app.get("/health")
def health():
    return {
        "status": "healthy" if MODELS else "models_not_loaded",
        "models": list(MODELS.keys()),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/features")
def get_features():
    """Returns list of all features the model uses"""
    if not MODELS:
        raise HTTPException(503, "Models not loaded. Run notebook first.")
    return {"features": MODELS["feats"], "count": len(MODELS["feats"])}

@app.get("/sample-input")
def sample_input():
    """Returns a sample input for testing"""
    return {
        "user_type": "salaried_private",
        "monthly_avg_salary_credit": 65000,
        "salary_credit_count_12m": 12,
        "emi_paid_on_time_ratio": 0.95,
        "cheque_bounce_count_12m": 0,
        "net_savings_rate": 0.28,
        "utility_payment_consistency_score": 0.90,
        "account_vintage_months": 48,
        "income_growth_yoy": 0.12,
        "balance_below_5k_days_12m": 2,
        "emi_bounce_count_12m": 0,
        "savings_txn_count_12m": 15,
        "kyc_completeness_score": 1.0
    }

@app.post("/predict")
def predict(user: UserInput):
    """
    Main prediction endpoint.
    Pass any subset of features — missing ones are auto-imputed via GMM.
    """
    if not MODELS:
        raise HTTPException(503, "Models not loaded. Run the Jupyter notebook first to train models.")

    feat_cols = MODELS["feats"]
    user_dict = user.dict()
    user_type = user_dict.pop("user_type")

    # Track which fields were provided vs missing
    provided = {k: v for k, v in user_dict.items() if v is not None}
    missing  = [k for k, v in user_dict.items() if v is None and k in feat_cols]

    # Build feature row
    row = {c: user_dict.get(c, np.nan) for c in feat_cols}

    # GMM Imputation
    row_imputed = gmm_impute_single(row, feat_cols)

    # Encode user type
    try:
        utype_enc = int(MODELS["le"].transform([user_type])[0])
    except:
        utype_enc = 0

    # Build feature df
    X = pd.DataFrame([row_imputed])[feat_cols].copy()
    X["user_type_enc"] = utype_enc

    # Predict
    raw_score = float(MODELS["xgb"].predict(X)[0])
    score     = int(np.clip(round(raw_score), 300, 900))
    band      = to_band(score)
    decision  = band_to_decision(band)
    label     = band_to_label(band)

    # SHAP explanation
    sv      = MODELS["explainer"].shap_values(X)
    sv_s    = pd.Series(sv[0], index=X.columns)
    pos_top = sv_s.nlargest(5).to_dict()
    neg_top = sv_s.nsmallest(5).to_dict()

    # Cluster membership
    x_scaled      = MODELS["scaler"].transform(pd.DataFrame([row_imputed])[feat_cols].values)
    cluster_probs = MODELS["gmm"].predict_proba(x_scaled)[0]
    cluster_mem   = {f"C{i}": round(float(p), 3) for i, p in enumerate(cluster_probs)}

    # Score breakdown (for gauge chart in frontend)
    score_pct = (score - 300) / 600 * 100

    return {
        "credit_score":     score,
        "score_percentage": round(score_pct, 1),
        "risk_band":        band,
        "risk_label":       label,
        "loan_decision":    decision,
        "user_type":        user_type,
        "features_provided": len(provided),
        "features_imputed":  len(missing),
        "imputed_features":  missing[:10],
        "shap_explanation": {
            "positive_factors": {k: round(v, 3) for k, v in pos_top.items()},
            "negative_factors": {k: round(v, 3) for k, v in neg_top.items()},
        },
        "cluster_membership": cluster_mem,
        "band_thresholds": {
            "A_min": 750, "B_min": 650, "C_min": 550, "D_min": 300
        },
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict/batch")
def predict_batch(users: list[UserInput]):
    """Batch predict for multiple users"""
    if not MODELS:
        raise HTTPException(503, "Models not loaded.")
    if len(users) > 100:
        raise HTTPException(400, "Max 100 users per batch request.")
    return [predict(u) for u in users]


@app.get("/stats")
def get_stats():
    """Returns model statistics for dashboard"""
    return {
        "model_info": {
            "type": "XGBoost Regressor",
            "training_samples": "2,00,000",
            "features": 57,
            "gmm_clusters": MODELS["gmm"].n_components if MODELS else "N/A",
            "r2_score": 0.9994,
            "mae": 2.2,
            "band_accuracy": 98.6,
        },
        "score_bands": {
            "A": {"range": "750-900", "label": "Low Risk",       "decision": "Auto Approve",           "color": "#22c55e"},
            "B": {"range": "650-749", "label": "Moderate Risk",  "decision": "Approve with Conditions","color": "#f59e0b"},
            "C": {"range": "550-649", "label": "High Risk",      "decision": "Manual Review",          "color": "#f97316"},
            "D": {"range": "300-549", "label": "Very High Risk", "decision": "Reject",                 "color": "#ef4444"},
        },
        "user_types": [
            "salaried_private", "salaried_govt",
            "shopkeeper", "businessman", "self_employed"
        ]
    }





# """
# =====================================================================
# DARCRAYS — FastAPI Backend (Real Architecture)
# =====================================================================
# Endpoints:
#   GET  /                        — API info
#   GET  /health                  — Health check
#   GET  /customer/{id}           — Fetch customer from DB
#   POST /predict/customer/{id}   — Score existing customer from DB
#   POST /predict/manual          — Score new user (manual form input)
#   GET  /customers               — List all customers (paginated)
#   GET  /stats                   — Model stats for dashboard

# Run:
#   uvicorn main:app --reload --port 8000
# =====================================================================
# """

# from fastapi import FastAPI, HTTPException, Query
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Optional
# import numpy as np
# import pandas as pd
# import joblib, shap, sqlite3, os
# from datetime import datetime

# # Import our feature engineering pipeline
# from feature_engineering import engineer_features, get_connection

# # ─────────────────────────────────────────────────────────────
# app = FastAPI(
#     title="DARCRAYS Credit Scoring API",
#     description="Real Bank Architecture — Raw DB → Feature Engineering → ML Scoring",
#     version="2.0.0",
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# MODELS = {}
# ML_FEATURES = []   # ordered list of features the model expects


# @app.on_event("startup")
# def load_models():
#     required = ["gmm_model.pkl","scaler.pkl","xgb_model.pkl",
#                 "label_encoder.pkl","feature_cols.pkl"]
#     missing = [f for f in required if not os.path.exists(f)]
#     if missing:
#         print(f"⚠️  Missing model files: {missing}")
#         return
#     MODELS["gmm"]      = joblib.load("gmm_model.pkl")
#     MODELS["scaler"]   = joblib.load("scaler.pkl")
#     MODELS["xgb"]      = joblib.load("xgb_model.pkl")
#     MODELS["le"]       = joblib.load("label_encoder.pkl")
#     MODELS["feats"]    = joblib.load("feature_cols.pkl")
#     MODELS["explainer"]= shap.TreeExplainer(MODELS["xgb"])
#     global ML_FEATURES
#     ML_FEATURES = MODELS["feats"]
#     print(f"✅ Models loaded | Features: {len(ML_FEATURES)}")

#     if not os.path.exists("bank_data.db"):
#         print("⚠️  bank_data.db not found. Run setup_database.py first.")
#     else:
#         conn = sqlite3.connect("bank_data.db")
#         n = pd.read_sql("SELECT COUNT(*) as c FROM customers", conn).iloc[0]["c"]
#         conn.close()
#         print(f"✅ DB connected | {n} customers")


# # ─────────────────────────────────────────────────────────────
# #  HELPERS
# # ─────────────────────────────────────────────────────────────
# def to_band(score):
#     if score >= 750: return "A"
#     if score >= 650: return "B"
#     if score >= 550: return "C"
#     return "D"

# BAND_META = {
#     "A": {"label":"Low Risk",       "decision":"AUTO_APPROVE",            "color":"#22c55e"},
#     "B": {"label":"Moderate Risk",  "decision":"APPROVE_WITH_CONDITIONS", "color":"#f59e0b"},
#     "C": {"label":"High Risk",      "decision":"MANUAL_REVIEW",           "color":"#f97316"},
#     "D": {"label":"Very High Risk", "decision":"REJECT",                  "color":"#ef4444"},
# }


# def gmm_impute_row(row_dict: dict) -> dict:
#     """Impute missing values using GMM clusters"""
#     feat_cols = ML_FEATURES
#     gmm       = MODELS["gmm"]
#     scaler    = MODELS["scaler"]

#     X    = np.array([row_dict.get(c, np.nan) for c in feat_cols], dtype=float)
#     mask = np.isnan(X)
#     if not mask.any():
#         return row_dict

#     obs      = np.where(~mask)[0]
#     miss_idx = np.where(mask)[0]
#     mu_orig  = scaler.inverse_transform(gmm.means_)

#     if len(obs) == 0:
#         X[miss_idx] = mu_orig[:, miss_idx].mean(axis=0)
#     else:
#         tmp = X.copy()
#         tmp[miss_idx] = mu_orig[:, miss_idx].mean(axis=0)
#         tmp_s = scaler.transform(tmp.reshape(1, -1))[0]
#         lp    = np.log(gmm.weights_ + 1e-10)
#         for k in range(gmm.n_components):
#             diff = tmp_s[obs] - gmm.means_[k, obs]
#             var  = gmm.covariances_[k, obs] + 1e-6
#             lp[k] += -0.5 * np.sum(diff**2 / var + np.log(2*np.pi*var))
#         lp -= lp.max()
#         probs = np.exp(lp); probs /= probs.sum()
#         X[miss_idx] = probs @ mu_orig[:, miss_idx]

#     return {c: float(X[i]) for i, c in enumerate(feat_cols)}


# def run_scoring(feat_dict: dict, user_type: str) -> dict:
#     """
#     Core scoring logic:
#       feat_dict  — ML features (may have NaN → GMM fills)
#       user_type  — user type string
#     Returns full prediction result dict.
#     """
#     # GMM imputation
#     feat_imputed = gmm_impute_row(feat_dict)
#     imputed_cols = [k for k in ML_FEATURES if np.isnan(feat_dict.get(k, np.nan))]

#     # Encode user type
#     try:
#         utype_enc = int(MODELS["le"].transform([user_type])[0])
#     except:
#         utype_enc = 0

#     # Build model input
#     X = pd.DataFrame([feat_imputed])[ML_FEATURES].copy()
#     X["user_type_enc"] = utype_enc

#     # Predict
#     raw   = float(MODELS["xgb"].predict(X)[0])
#     score = int(np.clip(round(raw), 300, 900))
#     band  = to_band(score)
#     meta  = BAND_META[band]

#     # SHAP
#     sv     = MODELS["explainer"].shap_values(X)
#     sv_s   = pd.Series(sv[0], index=X.columns)
#     pos    = sv_s.nlargest(5).round(3).to_dict()
#     neg    = sv_s.nsmallest(5).round(3).to_dict()

#     # Cluster membership
#     x_sc   = MODELS["scaler"].transform(pd.DataFrame([feat_imputed])[ML_FEATURES].values)
#     c_prob = MODELS["gmm"].predict_proba(x_sc)[0]
#     clusters = {f"C{i}": round(float(p), 3) for i, p in enumerate(c_prob)}

#     return {
#         "credit_score":       score,
#         "score_percentage":   round((score - 300) / 600 * 100, 1),
#         "risk_band":          band,
#         "risk_label":         meta["label"],
#         "loan_decision":      meta["decision"],
#         "band_color":         meta["color"],
#         "user_type":          user_type,
#         "features_imputed":   len(imputed_cols),
#         "imputed_features":   imputed_cols[:10],
#         "shap_explanation": {
#             "positive_factors": pos,
#             "negative_factors": neg,
#         },
#         "cluster_membership": clusters,
#         "timestamp":          datetime.now().isoformat(),
#     }


# # ─────────────────────────────────────────────────────────────
# #  ENDPOINTS
# # ─────────────────────────────────────────────────────────────

# @app.get("/")
# def root():
#     return {
#         "name":    "DARCRAYS Credit Scoring API v2",
#         "status":  "running",
#         "models":  len(MODELS) > 0,
#         "db":      os.path.exists("bank_data.db"),
#         "architecture": "Raw Transactions DB → Feature Engineering → GMM Imputation → XGBoost → Score",
#     }


# @app.get("/health")
# def health():
#     db_ok = os.path.exists("bank_data.db")
#     return {
#         "status":        "healthy" if (MODELS and db_ok) else "degraded",
#         "models_loaded": len(MODELS) > 0,
#         "db_connected":  db_ok,
#         "timestamp":     datetime.now().isoformat(),
#     }


# @app.get("/customers")
# def list_customers(
#     page:      int = Query(1, ge=1),
#     page_size: int = Query(20, ge=1, le=100),
#     user_type: Optional[str] = None,
#     risk_band: Optional[str] = None,
#     search:    Optional[str] = None,
# ):
#     """List customers from DB with pagination + filters"""
#     if not os.path.exists("bank_data.db"):
#         raise HTTPException(503, "Database not found. Run setup_database.py first.")

#     conn = get_connection()
#     query  = "SELECT customer_id, full_name, age, city, user_type, risk_band, kyc_status FROM customers WHERE 1=1"
#     params = []

#     if user_type:
#         query += " AND user_type = ?"; params.append(user_type)
#     if risk_band:
#         query += " AND risk_band = ?"; params.append(risk_band)
#     if search:
#         query += " AND (customer_id LIKE ? OR full_name LIKE ?)"; params += [f"%{search}%",f"%{search}%"]

#     total = pd.read_sql(f"SELECT COUNT(*) as c FROM ({query})", conn, params=params).iloc[0]["c"]
#     offset = (page - 1) * page_size
#     query += f" LIMIT {page_size} OFFSET {offset}"

#     df = pd.read_sql(query, conn, params=params)
#     conn.close()

#     return {
#         "total":     int(total),
#         "page":      page,
#         "page_size": page_size,
#         "pages":     int(np.ceil(total / page_size)),
#         "customers": df.to_dict(orient="records"),
#     }


# @app.get("/customer/{customer_id}")
# def get_customer(customer_id: str):
#     """Get customer profile + account summary from DB"""
#     if not os.path.exists("bank_data.db"):
#         raise HTTPException(503, "Database not found.")

#     conn = get_connection()
#     cust = pd.read_sql("SELECT * FROM customers WHERE customer_id = ?", conn, params=(customer_id,))
#     if cust.empty:
#         conn.close()
#         raise HTTPException(404, f"Customer {customer_id} not found")

#     acc  = pd.read_sql("SELECT * FROM accounts WHERE customer_id = ?", conn, params=(customer_id,))
#     txn_summary = pd.read_sql("""
#         SELECT COUNT(*) as total_txns,
#                SUM(CASE WHEN txn_type='CREDIT' THEN amount ELSE 0 END) as total_credits,
#                SUM(CASE WHEN txn_type='DEBIT'  THEN amount ELSE 0 END) as total_debits,
#                MIN(txn_date) as first_txn_date,
#                MAX(txn_date) as last_txn_date
#         FROM transactions WHERE customer_id = ?
#     """, conn, params=(customer_id,))
#     emi_summary = pd.read_sql("""
#         SELECT COUNT(*) as total_emis,
#                SUM(CASE WHEN status='BOUNCED' THEN 1 ELSE 0 END) as bounced
#         FROM emi_payments WHERE customer_id = ?
#     """, conn, params=(customer_id,))
#     conn.close()

#     return {
#         "customer":    cust.iloc[0].to_dict(),
#         "account":     acc.iloc[0].to_dict() if not acc.empty else {},
#         "txn_summary": txn_summary.iloc[0].to_dict(),
#         "emi_summary": emi_summary.iloc[0].to_dict(),
#     }


# @app.post("/predict/customer/{customer_id}")
# def predict_from_db(customer_id: str):
#     """
#     Score an EXISTING customer from the database.
#     Fetches raw 3-year transaction data → engineers features → predicts.
#     """
#     if not MODELS:
#         raise HTTPException(503, "Models not loaded. Run Jupyter notebook first.")
#     if not os.path.exists("bank_data.db"):
#         raise HTTPException(503, "Database not found. Run setup_database.py first.")

#     try:
#         # Step 1: Feature engineering from raw DB data
#         feats = engineer_features(customer_id)
#     except ValueError as e:
#         raise HTTPException(404, str(e))
#     except Exception as e:
#         raise HTTPException(500, f"Feature engineering failed: {str(e)}")

#     # Separate metadata from ML features
#     user_type = feats.pop("_user_type", "salaried_private")
#     meta_keys = [k for k in feats if k.startswith("_")]
#     customer_meta = {k: feats.pop(k) for k in meta_keys}

#     # Step 2: Score
#     result = run_scoring(feats, user_type)

#     # Merge customer info into response
#     result["customer_id"]   = customer_id
#     result["customer_name"] = customer_meta.get("_full_name", "")
#     result["city"]          = customer_meta.get("_city", "")
#     result["total_txns"]    = customer_meta.get("_total_txns", 0)
#     result["data_months"]   = customer_meta.get("_data_months", 36)
#     result["source"]        = "DATABASE"

#     return result


# # ─────────────────────────────────────────────────────────────
# #  MANUAL ENTRY (new user not in DB)
# # ─────────────────────────────────────────────────────────────
# class ManualInput(BaseModel):
#     user_type: str = "salaried_private"
#     # Pass any features you know — rest are GMM-imputed
#     monthly_avg_salary_credit:         Optional[float] = None
#     salary_credit_count_12m:           Optional[float] = None
#     income_variability_cv:             Optional[float] = None
#     salary_day_consistency:            Optional[float] = None
#     employer_tenure_years:             Optional[float] = None
#     income_growth_yoy:                 Optional[float] = None
#     secondary_income_credit_ratio:     Optional[float] = None
#     total_annual_inflow:               Optional[float] = None
#     inflow_outflow_ratio_12m:          Optional[float] = None
#     avg_monthly_balance:               Optional[float] = None
#     min_balance_breach_count_12m:      Optional[float] = None
#     balance_below_5k_days_12m:         Optional[float] = None
#     avg_eom_balance:                   Optional[float] = None
#     balance_volatility_std:            Optional[float] = None
#     negative_balance_days_12m:         Optional[float] = None
#     balance_utilisation_ratio:         Optional[float] = None
#     monthly_avg_debit_amount:          Optional[float] = None
#     debit_txn_count_monthly_avg:       Optional[float] = None
#     grocery_spend_ratio:               Optional[float] = None
#     utility_spend_ratio:               Optional[float] = None
#     entertainment_spend_ratio:         Optional[float] = None
#     atm_withdrawal_ratio:              Optional[float] = None
#     upi_txn_count_monthly_avg:         Optional[float] = None
#     online_shopping_spend_ratio:       Optional[float] = None
#     spend_to_income_ratio:             Optional[float] = None
#     weekend_spend_ratio:               Optional[float] = None
#     active_emi_count:                  Optional[float] = None
#     total_emi_monthly_obligation:      Optional[float] = None
#     emi_to_income_ratio:               Optional[float] = None
#     emi_bounce_count_12m:              Optional[float] = None
#     emi_paid_on_time_ratio:            Optional[float] = None
#     nach_mandate_active:               Optional[float] = None
#     loan_repayment_track_score:        Optional[float] = None
#     electricity_bill_paid_ontime_12m:  Optional[float] = None
#     mobile_bill_paid_ontime_12m:       Optional[float] = None
#     broadband_paid_ontime_12m:         Optional[float] = None
#     insurance_premium_paid_12m:        Optional[float] = None
#     utility_payment_consistency_score: Optional[float] = None
#     cheque_bounce_count_12m:           Optional[float] = None
#     standing_instruction_success_rate: Optional[float] = None
#     rd_fd_count_active:                Optional[float] = None
#     savings_txn_count_12m:             Optional[float] = None
#     net_savings_rate:                  Optional[float] = None
#     investment_credit_ratio:           Optional[float] = None
#     sweep_account_utilisation:         Optional[float] = None
#     netbanking_login_days_monthly_avg: Optional[float] = None
#     mobile_app_sessions_monthly_avg:   Optional[float] = None
#     upi_autopay_mandates_active:       Optional[float] = None
#     debit_card_txn_count_12m:          Optional[float] = None
#     monthly_avg_business_credit:       Optional[float] = None
#     pos_txn_count_monthly_avg:         Optional[float] = None
#     gst_payment_count_12m:             Optional[float] = None
#     business_account_avg_balance:      Optional[float] = None
#     trade_credit_utilisation:          Optional[float] = None
#     receivables_turnover_days:         Optional[float] = None
#     age:                               Optional[float] = None
#     account_vintage_months:            Optional[float] = None
#     kyc_completeness_score:            Optional[float] = None
#     co_applicant_flag:                 Optional[float] = None
#     existing_relationship_score:       Optional[float] = None


# @app.post("/predict/manual")
# def predict_manual(data: ManualInput):
#     """
#     Score a NEW user not in the database.
#     Fill in as many features as you have from the bank statement.
#     Missing features are auto-imputed by GMM.
#     """
#     if not MODELS:
#         raise HTTPException(503, "Models not loaded.")

#     user_dict = data.dict()
#     user_type = user_dict.pop("user_type")
#     feat_dict = {k: v for k, v in user_dict.items() if k in ML_FEATURES}
#     # Replace None with NaN
#     feat_dict = {k: (np.nan if v is None else v) for k, v in feat_dict.items()}

#     result = run_scoring(feat_dict, user_type)
#     result["source"] = "MANUAL_ENTRY"
#     return result


# @app.get("/stats")
# def get_stats():
#     db_stats = {}
#     if os.path.exists("bank_data.db"):
#         conn = get_connection()
#         db_stats = {
#             "total_customers": int(pd.read_sql("SELECT COUNT(*) as c FROM customers", conn).iloc[0]["c"]),
#             "total_transactions": int(pd.read_sql("SELECT COUNT(*) as c FROM transactions", conn).iloc[0]["c"]),
#             "band_distribution": pd.read_sql("SELECT risk_band, COUNT(*) as count FROM customers GROUP BY risk_band", conn).set_index("risk_band")["count"].to_dict(),
#             "user_type_distribution": pd.read_sql("SELECT user_type, COUNT(*) as count FROM customers GROUP BY user_type", conn).set_index("user_type")["count"].to_dict(),
#         }
#         conn.close()

#     return {
#         "model_info": {
#             "type":             "XGBoost Regressor + GMM Imputation",
#             "training_samples": "2,00,000",
#             "features":         len(ML_FEATURES) if ML_FEATURES else 57,
#             "gmm_clusters":     MODELS["gmm"].n_components if MODELS else "N/A",
#             "r2_score":         0.9994,
#             "mae":              2.2,
#             "band_accuracy":    98.6,
#             "data_window":      "3 years raw transactions",
#         },
#         "score_bands": BAND_META,
#         "user_types":  ["salaried_private","salaried_govt","shopkeeper","businessman","self_employed"],
#         "db_stats":    db_stats,
#     }


# @app.get("/sample-input")
# def sample_input():
#     return {
#         "user_type":                        "salaried_private",
#         "monthly_avg_salary_credit":        65000,
#         "salary_credit_count_12m":          12,
#         "emi_paid_on_time_ratio":           0.95,
#         "cheque_bounce_count_12m":          0,
#         "net_savings_rate":                 0.28,
#         "utility_payment_consistency_score":0.90,
#         "account_vintage_months":           48,
#         "income_growth_yoy":                0.12,
#         "balance_below_5k_days_12m":        2,
#         "emi_bounce_count_12m":             0,
#         "savings_txn_count_12m":            15,
#         "kyc_completeness_score":           1.0,
#     }