import logging
import os

logger = logging.getLogger(__name__)

try:
    from psycopg2 import pool as pg_pool  # type: ignore[import]
    _PSYCOPG2_AVAILABLE = True
except ImportError:
    _PSYCOPG2_AVAILABLE = False

_pool = None

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS prediction_logs (
    id                SERIAL PRIMARY KEY,
    timestamp         TIMESTAMPTZ NOT NULL,
    gender            INT,
    senior_citizen    INT,
    partner           INT,
    dependents        INT,
    tenure            INT,
    phone_service     INT,
    multiple_lines    INT,
    internet_service  INT,
    online_security   INT,
    online_backup     INT,
    device_protection INT,
    tech_support      INT,
    streaming_tv      INT,
    streaming_movies  INT,
    contract          INT,
    paperless_billing INT,
    payment_method    INT,
    monthly_charges   FLOAT,
    total_charges     FLOAT,
    churn_probability FLOAT,
    churn_prediction  INT
);
"""

_INSERT = """
INSERT INTO prediction_logs (
    timestamp, gender, senior_citizen, partner, dependents, tenure,
    phone_service, multiple_lines, internet_service, online_security,
    online_backup, device_protection, tech_support, streaming_tv,
    streaming_movies, contract, paperless_billing, payment_method,
    monthly_charges, total_charges, churn_probability, churn_prediction
) VALUES (
    %(timestamp)s, %(gender)s, %(SeniorCitizen)s, %(Partner)s, %(Dependents)s,
    %(tenure)s, %(PhoneService)s, %(MultipleLines)s, %(InternetService)s,
    %(OnlineSecurity)s, %(OnlineBackup)s, %(DeviceProtection)s, %(TechSupport)s,
    %(StreamingTV)s, %(StreamingMovies)s, %(Contract)s, %(PaperlessBilling)s,
    %(PaymentMethod)s, %(MonthlyCharges)s, %(TotalCharges)s,
    %(churn_probability)s, %(churn_prediction)s
);
"""


def init_db():
    global _pool
    if not _PSYCOPG2_AVAILABLE:
        logger.warning("psycopg2 not installed — DB logging disabled")
        return
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.warning("DATABASE_URL not set — DB logging disabled")
        return
    try:
        _pool = pg_pool.ThreadedConnectionPool(1, 10, dsn=db_url)
        conn = _pool.getconn()
        with conn.cursor() as cur:
            cur.execute(_CREATE_TABLE)
        conn.commit()
        _pool.putconn(conn)
        logger.info("Database initialised and prediction_logs table ready")
    except Exception as e:
        logger.error(f"Database init failed: {e}")
        _pool = None


def log_prediction(timestamp, input_data: dict, churn_probability: float, churn_prediction: int):
    if _pool is None:
        return
    conn = None
    try:
        conn = _pool.getconn()
        with conn.cursor() as cur:
            cur.execute(_INSERT, {
                "timestamp": timestamp,
                **input_data,
                "churn_probability": churn_probability,
                "churn_prediction": churn_prediction,
            })
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to log prediction to DB: {e}")
    finally:
        if conn:
            _pool.putconn(conn)
