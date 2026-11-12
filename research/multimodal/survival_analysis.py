"""Conditional Survival Analysis and Follow-Up Duration Auditing Engine.

Evaluates clinical cohort eligibility for time-to-event survival modeling:
- Scans clinical schemas for verified follow-up duration (e.g. OS_months, DFS_days).
- Verifies binary vital status / event indicators (death vs alive/censored).
- Confirms time origin (date of surgery / pathological confirmation).
- Enforces strict scientific blocking: raises SurvivalAnalysisBlockedError if survival
  fields are absent, prohibiting synthetic or imputed survival modeling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("multimodal.survival")


class SurvivalAnalysisBlockedError(RuntimeError):
    """Raised when survival analysis is requested on a clinical dataset lacking valid follow-up or vital status."""
    pass


@dataclass
class SurvivalEligibilityResult:
    """Documented assessment of clinical dataset suitability for time-to-event survival analysis."""
    is_eligible: bool
    follow_up_time_field: Optional[str]
    event_indicator_field: Optional[str]
    time_origin_field: Optional[str]
    total_records_examined: int
    observed_events_count: int
    censored_count: int
    missing_requirements: List[str]
    blocker_description: str
    unblocking_requirements: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SurvivalEligibilityAuditor:
    """
    Audits a clinical DataFrame to verify whether valid survival endpoints are present.
    """

    KNOWN_TIME_FIELDS = [
        "os_months", "overall_survival_months", "survival_time", "follow_up_months",
        "pfs_months", "dfs_months", "time_to_death", "days_to_last_followup", "survival_duration"
    ]
    KNOWN_EVENT_FIELDS = [
        "vital_status", "death", "status", "event", "relapse", "recurrence", "os_event", "mortality"
    ]
    KNOWN_ORIGIN_FIELDS = [
        "surgery_date", "diagnosis_date", "admission_date", "biopsy_date", "date_of_diagnosis"
    ]

    @classmethod
    def audit_dataset_for_survival(cls, df: pd.DataFrame) -> SurvivalEligibilityResult:
        """
        Inspects dataframe columns for verified survival endpoints.
        """
        cols_lower = {c.lower().strip(): c for c in df.columns}

        found_time: Optional[str] = None
        for candidate in cls.KNOWN_TIME_FIELDS:
            if candidate in cols_lower:
                found_time = cols_lower[candidate]
                break

        found_event: Optional[str] = None
        for candidate in cls.KNOWN_EVENT_FIELDS:
            if candidate in cols_lower:
                found_event = cols_lower[candidate]
                break

        found_origin: Optional[str] = None
        for candidate in cls.KNOWN_ORIGIN_FIELDS:
            if candidate in cols_lower:
                found_origin = cols_lower[candidate]
                break

        missing_reqs: List[str] = []
        if found_time is None:
            missing_reqs.append("Verified follow-up duration (e.g. OS_months, survival_time)")
        if found_event is None:
            missing_reqs.append("Verified vital status / event indicator (e.g. death=1, alive/censored=0)")
        if found_origin is None:
            missing_reqs.append("Verified time origin (date of surgical resection or initial histopathological diagnosis)")

        is_eligible = len(missing_reqs) == 0

        if not is_eligible:
            blocker_desc = (
                f"HMU-GC-Clinical.csv contains strictly cross-sectional pathological variables "
                f"(TNM staging, Lauren classification, histological subtype, HER-2 score). "
                f"The dataset lacks {', '.join(missing_reqs)}. Inferring survival from stage or "
                f"synthesizing follow-up times is strictly prohibited by research protocol."
            )
            unblocking = (
                "Requires longitudinal clinical follow-up data from Harbin Medical University "
                "recording overall survival duration (months/days), vital status at last contact, "
                "and date of primary surgery."
            )
            return SurvivalEligibilityResult(
                is_eligible=False,
                follow_up_time_field=None,
                event_indicator_field=None,
                time_origin_field=None,
                total_records_examined=len(df),
                observed_events_count=0,
                censored_count=0,
                missing_requirements=missing_reqs,
                blocker_description=blocker_desc,
                unblocking_requirements=unblocking,
            )

        # If eligible (e.g. in test fixture mode with synthetic survival data)
        events = int(df[found_event].sum())
        censored = len(df) - events
        return SurvivalEligibilityResult(
            is_eligible=True,
            follow_up_time_field=found_time,
            event_indicator_field=found_event,
            time_origin_field=found_origin,
            total_records_examined=len(df),
            observed_events_count=events,
            censored_count=censored,
            missing_requirements=[],
            blocker_description="Dataset meets all prerequisites for time-to-event survival modeling.",
            unblocking_requirements="None (Eligible).",
        )


class ConditionalSurvivalModelRunner:
    """
    Executes survival modeling ONLY if eligibility criteria are rigorously met.
    """

    @staticmethod
    def run_kaplan_meier_analysis(
        df: pd.DataFrame,
        time_col: str = "os_months",
        event_col: str = "vital_status",
    ) -> Dict[str, Any]:
        """
        Executes Kaplan-Meier analysis if columns exist.
        Raises SurvivalAnalysisBlockedError if columns are absent.
        """
        audit = SurvivalEligibilityAuditor.audit_dataset_for_survival(df)
        if not audit.is_eligible:
            raise SurvivalAnalysisBlockedError(audit.blocker_description)

        # Standard Kaplan-Meier estimation
        times = df[time_col].to_numpy(dtype=float)
        events = df[event_col].to_numpy(dtype=int)

        unique_times = np.sort(np.unique(times))
        s_t = 1.0
        n_at_risk = len(times)
        survival_curve = []

        for t in unique_times:
            d_i = np.sum((times == t) & (events == 1))
            c_i = np.sum((times == t) & (events == 0))
            if n_at_risk > 0:
                s_t *= (1.0 - d_i / n_at_risk)
            survival_curve.append({
                "time": float(t),
                "n_at_risk": int(n_at_risk),
                "events": int(d_i),
                "censored": int(c_i),
                "survival_probability": float(s_t),
            })
            n_at_risk -= (d_i + c_i)

        return {
            "median_survival_time": float(np.median(times)),
            "total_patients": len(df),
            "total_events": int(np.sum(events)),
            "curve": survival_curve,
        }

