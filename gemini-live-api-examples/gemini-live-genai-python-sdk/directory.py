"""
Employee roster — maps a phone number (E.164) to the employee's first name (for
the opening identity check: "नमस्ते! क्या मेरी बात Pratik जी से हो रही है?") and
their employee ID (carried onto the call record for the assessment report).

Loaded lazily from a data file (CSV or JSON) at MEMBER_DIRECTORY_PATH
(default: data/employees.csv). Pure stdlib, no app imports, never raises — an
unknown number or a missing file simply yields "" (the agent then falls back to
the unnamed opening).

CSV shape (header required; employee_id optional):
    phone,first_name,employee_id
    +919876543210,Pratik,CNY-042

JSON shape (either form):
    {"+919876543210": "Pratik"}
    [{"phone": "+919876543210", "first_name": "Pratik", "employee_id": "CNY-042"}]

The file is PII — keep it out of version control and do not expose names on the
public /live dashboard.
"""

import csv
import json
import logging
import os
import re
import threading

logger = logging.getLogger(__name__)

_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "data", "employees.csv")
_LEGACY_PATH = os.path.join(os.path.dirname(__file__), "data", "members.csv")

_LOCK = threading.Lock()
_MAP = None            # normalized phone -> {"first_name": ..., "employee_id": ...}
_MISSING_LOGGED = False


def normalize_phone(raw):
    """Normalise a phone number for exact matching.

    Strips spaces/dashes/parens/dots, keeps a single leading '+', and defaults a
    bare 10-digit Indian mobile to +91. Returns '' for anything without digits.
    """
    if not raw:
        return ""
    s = str(raw).strip()
    plus = s.startswith("+")
    digits = re.sub(r"\D", "", s)
    if not digits:
        return ""
    if plus:
        return "+" + digits
    if len(digits) == 10:                 # bare Indian mobile
        return "+91" + digits
    if len(digits) == 12 and digits.startswith("91"):
        return "+" + digits
    return "+" + digits


def _path():
    p = os.getenv("MEMBER_DIRECTORY_PATH")
    if p:
        return p
    if not os.path.isfile(_DEFAULT_PATH) and os.path.isfile(_LEGACY_PATH):
        return _LEGACY_PATH               # pre-rename deployments keep working
    return _DEFAULT_PATH


def _record(name, employee_id=""):
    return {"first_name": str(name or "").strip(),
            "employee_id": str(employee_id or "").strip()}


def _load(path):
    """Read the roster file into a {normalized_phone: record} dict."""
    out = {}
    _, ext = os.path.splitext(path.lower())
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        if ext == ".json":
            data = json.load(f)
            if isinstance(data, dict):
                for phone, name in data.items():
                    key = normalize_phone(phone)
                    if key and name:
                        out[key] = _record(name)
            else:
                for row in data:
                    key = normalize_phone(row.get("phone"))
                    name = row.get("first_name") or row.get("name")
                    if key and name:
                        out[key] = _record(name, row.get("employee_id"))
        else:
            for row in csv.DictReader(f):
                # tolerate header variants: phone/number, first_name/name, employee_id/emp_id
                phone = row.get("phone") or row.get("number") or row.get("mobile")
                name = row.get("first_name") or row.get("name") or row.get("firstname")
                emp = row.get("employee_id") or row.get("emp_id") or row.get("id")
                key = normalize_phone(phone)
                if key and name:
                    out[key] = _record(name, emp)
    return out


def _ensure_loaded():
    global _MAP, _MISSING_LOGGED
    if _MAP is not None:
        return _MAP
    with _LOCK:
        if _MAP is not None:
            return _MAP
        path = _path()
        try:
            _MAP = _load(path)
            logger.info(f"Employee roster loaded from {path} ({len(_MAP)} employees)")
        except FileNotFoundError:
            if not _MISSING_LOGGED:
                logger.info(f"Employee roster not found at {path}; greetings will be generic")
                _MISSING_LOGGED = True
            _MAP = {}
        except Exception as e:
            logger.warning(f"Failed to load employee roster {path}: {e}")
            _MAP = {}
    return _MAP


def first_name_for(phone):
    """Return the employee's first name for a phone number, or '' if unknown.

    Exact match only (never fuzzy) — confirming the wrong name in a disciplinary
    screening is worse than no name.
    """
    key = normalize_phone(phone)
    if not key:
        return ""
    return (_ensure_loaded().get(key) or {}).get("first_name", "")


def record_for(phone):
    """Return {'first_name': ..., 'employee_id': ...} for a phone number, or {}."""
    key = normalize_phone(phone)
    if not key:
        return {}
    rec = _ensure_loaded().get(key)
    return dict(rec) if rec else {}


def reload():
    """Drop the cache so the next lookup re-reads the file (e.g. after an edit)."""
    global _MAP, _MISSING_LOGGED
    with _LOCK:
        _MAP = None
        _MISSING_LOGGED = False
