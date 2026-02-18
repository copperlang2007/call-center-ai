"""
INSUREitALL Commission Engine v2.0
Enterprise Medicare Agency Commission Calculator

Purpose: Process carrier data files and produce agent-level pay run reports
with chargebacks, compliance holds, liquidity netting, and hierarchy overrides.

Usage:
    engine = CommissionEngine()
    engine.load_agent_roster("agents.csv")
    pay_run = engine.run_pay_cycle("producer_toolbox_export.csv", pay_date="2026-02-13")
    pay_run.export("pay_run_2026-02-13.csv")
    pay_run.print_summary()
"""

import pandas as pd
import re
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP
import csv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CommissionEngine")

# =============================================================================
# 1. CONFIGURATION: MASTER RATE CARD
# =============================================================================

@dataclass
class PlanRateCard:
    carrier_name: str
    plan_id: str
    year: int
    street_initial_annual: float
    street_renewal_annual: float
    hra_gross: float
    marketing_kicker: float

# 2026 CMS National Max Rates: $694 initial / $347 renewal
# NOTE: Update HRA and kicker from your carrier Value-Based contracts
RATE_CARD_DB: Dict[str, PlanRateCard] = {
    # HUMANA
    "H1036-291": PlanRateCard("Humana", "H1036-291", 2026, 694.00, 347.00, 50.00, 0.0),
    "H1951-044": PlanRateCard("Humana", "H1951-044", 2026, 694.00, 347.00, 50.00, 0.0),
    "H5216-348": PlanRateCard("Humana", "H5216-348", 2026, 694.00, 347.00, 50.00, 0.0),
    "H4141-022": PlanRateCard("Humana", "H4141-022", 2026, 694.00, 347.00, 50.00, 0.0),
    "H7617-076": PlanRateCard("Humana", "H7617-076", 2026, 694.00, 347.00, 50.00, 0.0),
    "H4461-070": PlanRateCard("Humana", "H4461-070", 2026, 694.00, 347.00, 50.00, 0.0),
    # UNITEDHEALTHCARE
    "H4514-021": PlanRateCard("UnitedHealthcare", "H4514-021", 2026, 694.00, 347.00, 50.00, 0.0),
    "H5253-189": PlanRateCard("UnitedHealthcare", "H5253-189", 2026, 694.00, 347.00, 50.00, 0.0),
    "H0251-008": PlanRateCard("UnitedHealthcare", "H0251-008", 2026, 694.00, 347.00, 50.00, 0.0),
    "H1889-034": PlanRateCard("UnitedHealthcare", "H1889-034", 2026, 694.00, 347.00, 50.00, 0.0),
    # AETNA
    "H3146-006": PlanRateCard("Aetna", "H3146-006", 2026, 694.00, 347.00, 50.00, 0.0),
    "H5522-013": PlanRateCard("Aetna", "H5522-013", 2026, 694.00, 347.00, 50.00, 0.0),
    "H3239-003": PlanRateCard("Aetna", "H3239-003", 2026, 694.00, 347.00, 50.00, 0.0),
    # DEVOTED HEALTH
    "H7605-004": PlanRateCard("Devoted Health", "H7605-004", 2026, 694.00, 347.00, 50.00, 0.0),
    "H9888-006": PlanRateCard("Devoted Health", "H9888-006", 2026, 694.00, 347.00, 50.00, 0.0),
    # CIGNA
    "H4513-091": PlanRateCard("Cigna", "H4513-091", 2026, 694.00, 347.00, 50.00, 0.0),
    # WELLPOINT (ANTHEM)
    "H2593-051": PlanRateCard("Wellpoint", "H2593-051", 2026, 694.00, 347.00, 50.00, 0.0),
}

NATIONAL_FALLBACK = PlanRateCard("Unknown", "FALLBACK", 2026, 694.00, 347.00, 50.00, 0.0)

# =============================================================================
# 2. STATUS CODE MAPPING (expanded, with regex support)
# =============================================================================

# Exact match codes
STATUS_MAP_EXACT: Dict[str, str] = {
    "CHGBCK-VOL":   "CHARGEBACK_VOLUNTARY",
    "CHGBCK-INVOL": "CHARGEBACK_INVOLUNTARY",
    "ADVCX":        "CANCELLED_ADVANCE",
    "ADVREN":       "ACTIVE_RENEWAL",
    "ADVINT":       "ACTIVE_INITIAL",
    "NO CERT":      "HOLD_NO_CERT",
    "NO APPT":      "HOLD_NO_APPT",
    "DECEASED":     "CHARGEBACK_DEATH",
    "PEND":         "PENDING",
    "PEND-VERIFY":  "PENDING_VERIFICATION",
    "REINSTATE":    "ACTIVE_REINSTATEMENT",
    "DISENROLL":    "CHARGEBACK_VOLUNTARY",
    "RAPID-DIS":    "CHARGEBACK_VOLUNTARY",
    "LAPSE":        "CANCELLED_LAPSE",
    "NO NPN":       "HOLD_NO_NPN",
    "NO RTS":       "HOLD_NO_RTS",
    "NO SOA":       "HOLD_NO_SOA",
    "SUSP":         "HOLD_SUSPENDED",
    "TRANSFER-IN":  "ACTIVE_TRANSFER_IN",
    "TRANSFER-OUT": "CHARGEBACK_TRANSFER_OUT",
    "VOID":         "CANCELLED_VOID",
}

# FIX: Regex patterns for codes that vary by month/year (e.g. CX-02-26, CX-03-26)
STATUS_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"^CX-\d{2}-\d{2}$"), "CANCELLED_TERM"),
    (re.compile(r"^CHGBCK-.*$"),       "CHARGEBACK_VOLUNTARY"),
    (re.compile(r"^ADV-CX-.*$"),       "CANCELLED_ADVANCE"),
]

def resolve_status(raw_code: str) -> str:
    """Translate carrier raw status to internal financial status."""
    raw = raw_code.strip().upper()

    # Try exact match first
    if raw in STATUS_MAP_EXACT:
        return STATUS_MAP_EXACT[raw]

    # Try regex patterns
    for pattern, internal_status in STATUS_PATTERNS:
        if pattern.match(raw):
            return internal_status

    logger.warning(f"Unknown carrier status code: '{raw_code}' — defaulting to PENDING")
    return "PENDING"

# =============================================================================
# 3. HIERARCHY & OVERRIDE CONFIGURATION
# =============================================================================

@dataclass
class AgentProfile:
    agent_id: str
    agent_name: str
    agency_id: str
    agency_name: str
    team_lead_id: Optional[str] = None
    manager_id: Optional[str] = None
    director_id: Optional[str] = None
    agency_split: float = 1.0  # 1.0 = agent gets full street rate

# Override rates as percentage of street commission
OVERRIDE_RATES = {
    "team_lead": 0.05,     # 5% of agent's street commission
    "manager":   0.03,     # 3%
    "director":  0.02,     # 2%
    "agency":    0.00,     # Set per agency contract
}

# =============================================================================
# 4. LEDGER ENTRY
# =============================================================================

@dataclass
class LedgerEntry:
    agent_id: str
    enrollment_id: str
    plan_id: str
    carrier: str
    entry_type: str          # STREET_IMMEDIATE, STREET_TRUE_UP, HRA, KICKER,
                             # CHARGEBACK, OVERRIDE_TL, OVERRIDE_MGR, OVERRIDE_DIR
    amount: float
    due_date: datetime
    status: str              # PAYABLE, FROZEN, SCHEDULED, HOLD, CLAWBACK
    description: str
    risk_alert: str = "NONE"
    raw_carrier_code: str = ""

# =============================================================================
# 5. THE ENGINE
# =============================================================================

class CommissionEngine:
    def __init__(self, agent_hra_split: float = 0.5):
        self.agent_hra_split = agent_hra_split
        self.agent_roster: Dict[str, AgentProfile] = {}
        self.agent_balances: Dict[str, float] = {}
        self.ledger: List[LedgerEntry] = []
        self.hold_report: List[Dict] = []
        self.warnings: List[str] = []

    # -----------------------------------------------------------------
    # ROSTER MANAGEMENT
    # -----------------------------------------------------------------
    def load_agent_roster(self, filepath: str) -> None:
        """
        Load agent-to-agency mapping from CSV.
        Expected columns: agent_id, agent_name, agency_id, agency_name,
                          team_lead_id, manager_id, director_id, agency_split
        """
        df = pd.read_csv(filepath)
        for _, row in df.iterrows():
            profile = AgentProfile(
                agent_id=str(row["agent_id"]).strip(),
                agent_name=str(row.get("agent_name", "")).strip(),
                agency_id=str(row.get("agency_id", "DEFAULT")).strip(),
                agency_name=str(row.get("agency_name", "Unknown Agency")).strip(),
                team_lead_id=str(row.get("team_lead_id", "")).strip() or None,
                manager_id=str(row.get("manager_id", "")).strip() or None,
                director_id=str(row.get("director_id", "")).strip() or None,
                agency_split=float(row.get("agency_split", 1.0)),
            )
            self.agent_roster[profile.agent_id] = profile
        logger.info(f"Loaded {len(self.agent_roster)} agents from roster")

    def add_agent(self, profile: AgentProfile) -> None:
        """Manually add a single agent profile."""
        self.agent_roster[profile.agent_id] = profile

    def set_agent_balance(self, agent_id: str, balance: float) -> None:
        """Set carrier balance for an agent (from Producer Toolbox)."""
        self.agent_balances[agent_id] = balance

    # -----------------------------------------------------------------
    # RATE LOOKUP
    # -----------------------------------------------------------------
    def _get_rates(self, plan_id: str) -> PlanRateCard:
        rates = RATE_CARD_DB.get(plan_id)
        if not rates:
            self.warnings.append(f"Plan {plan_id} not in rate card — using national fallback")
            return PlanRateCard(
                NATIONAL_FALLBACK.carrier_name,
                plan_id,
                NATIONAL_FALLBACK.year,
                NATIONAL_FALLBACK.street_initial_annual,
                NATIONAL_FALLBACK.street_renewal_annual,
                NATIONAL_FALLBACK.hra_gross,
                NATIONAL_FALLBACK.marketing_kicker,
            )
        return rates

    # -----------------------------------------------------------------
    # LIQUIDITY NETTING (FIX: actually nets earnings against deficit)
    # -----------------------------------------------------------------
    def _apply_liquidity_netting(self, agent_id: str, gross_amount: float) -> Tuple[float, str]:
        """
        If agent has a negative carrier balance, new earnings fill the hole first.
        Returns (net_payable_amount, status_note).
        """
        current_balance = self.agent_balances.get(agent_id, 0.0)

        if current_balance >= 0:
            return gross_amount, "PAYABLE"

        # Agent is negative — offset
        net = gross_amount + current_balance  # current_balance is negative

        if net > 0:
            # Earnings exceed deficit — pay the remainder
            self.agent_balances[agent_id] = 0.0
            return round(net, 2), f"PAYABLE (offset ${abs(current_balance):.2f} deficit)"
        else:
            # Deficit exceeds earnings — absorb fully, pay nothing
            self.agent_balances[agent_id] = net  # still negative
            return 0.0, f"FROZEN (applied ${gross_amount:.2f} to ${abs(current_balance):.2f} deficit, remaining: ${abs(net):.2f})"

    # -----------------------------------------------------------------
    # SINGLE ENROLLMENT PROCESSOR
    # -----------------------------------------------------------------
    def process_enrollment(
        self,
        agent_id: str,
        enrollment_id: str,
        plan_id: str,
        effective_date: datetime,
        enrollment_date: datetime,
        carrier_raw_status: str,
        carrier_balance: float = 0.0,
        is_new_to_medicare: bool = False,
    ) -> List[LedgerEntry]:

        entries: List[LedgerEntry] = []
        internal_status = resolve_status(carrier_raw_status)
        rates = self._get_rates(plan_id)

        # Update agent balance from carrier feed
        if carrier_balance != 0.0:
            self.agent_balances[agent_id] = carrier_balance

        risk = "HIGH" if self.agent_balances.get(agent_id, 0.0) < 0 else "NONE"

        # ---- CHARGEBACKS: calculate actual reversal amount ----
        if "CHARGEBACK" in internal_status or "CANCELLED" in internal_status:
            # FIX: Calculate the actual clawback dollar amount
            # Use the rate that was originally paid
            if internal_status in ("CANCELLED_ADVANCE", "CANCELLED_VOID"):
                # Cancelled before effectuation — reverse full initial
                clawback_amount = rates.street_initial_annual
            else:
                # Chargeback after effectuation — reverse prorated amount
                months_active = max(
                    1, (datetime.now().month - effective_date.month) % 12 or 12
                )
                clawback_amount = (rates.street_initial_annual / 12) * months_active

            entries.append(LedgerEntry(
                agent_id=agent_id,
                enrollment_id=enrollment_id,
                plan_id=plan_id,
                carrier=rates.carrier_name,
                entry_type="CHARGEBACK",
                amount=round(-clawback_amount, 2),
                due_date=datetime.now(),
                status="CLAWBACK",
                description=f"Reversal: {internal_status} (carrier code: {carrier_raw_status})",
                risk_alert=risk,
                raw_carrier_code=carrier_raw_status,
            ))
            return entries

        # ---- COMPLIANCE HOLDS ----
        if "HOLD" in internal_status:
            self.hold_report.append({
                "agent_id": agent_id,
                "enrollment_id": enrollment_id,
                "plan_id": plan_id,
                "hold_type": internal_status,
                "raw_code": carrier_raw_status,
                "note": f"Agent missing credential ({carrier_raw_status}) — payments frozen",
            })
            entries.append(LedgerEntry(
                agent_id=agent_id,
                enrollment_id=enrollment_id,
                plan_id=plan_id,
                carrier=rates.carrier_name,
                entry_type="COMPLIANCE_HOLD",
                amount=0.0,
                due_date=effective_date,
                status="HOLD",
                description=f"Frozen: {internal_status} ({carrier_raw_status})",
                risk_alert=risk,
                raw_carrier_code=carrier_raw_status,
            ))
            return entries

        # ---- PENDING: log but don't pay ----
        if internal_status == "PENDING" or internal_status == "PENDING_VERIFICATION":
            entries.append(LedgerEntry(
                agent_id=agent_id,
                enrollment_id=enrollment_id,
                plan_id=plan_id,
                carrier=rates.carrier_name,
                entry_type="PENDING",
                amount=0.0,
                due_date=effective_date,
                status="PENDING",
                description=f"Awaiting confirmation (carrier code: {carrier_raw_status})",
                risk_alert=risk,
                raw_carrier_code=carrier_raw_status,
            ))
            return entries

        # ================================================================
        # ACTIVE ENROLLMENT — CALCULATE COMMISSION
        # ================================================================

        # FIX: Force renewal path when status is ACTIVE_RENEWAL
        is_renewal = (internal_status == "ACTIVE_RENEWAL")
        is_initial = (internal_status == "ACTIVE_INITIAL") and is_new_to_medicare

        months_remaining = 13 - effective_date.month

        # Get agent's agency split (default 100%)
        profile = self.agent_roster.get(agent_id)
        agency_split = profile.agency_split if profile else 1.0

        # ---- Trigger A: Prorated Street Commission (Immediate) ----
        if is_renewal:
            # RENEWAL: use renewal rate only, no true-up ever
            trigger_a = (rates.street_renewal_annual / 12) * months_remaining
        else:
            # INITIAL: start with prorated renewal rate (conservative pay)
            trigger_a = (rates.street_renewal_annual / 12) * months_remaining

        trigger_a_agent = round(trigger_a * agency_split, 2)

        # Apply liquidity netting
        net_amount, pay_status = self._apply_liquidity_netting(agent_id, trigger_a_agent)

        entries.append(LedgerEntry(
            agent_id=agent_id,
            enrollment_id=enrollment_id,
            plan_id=plan_id,
            carrier=rates.carrier_name,
            entry_type="STREET_IMMEDIATE",
            amount=net_amount,
            due_date=effective_date,
            status=pay_status,
            description=f"Prorated {'Renewal' if is_renewal else 'Street'} ({months_remaining} mos @ ${rates.street_renewal_annual}/yr)",
            risk_alert=risk,
            raw_carrier_code=carrier_raw_status,
        ))

        # ---- Trigger B: Initial True-Up (ONLY for new-to-Medicare initial) ----
        if is_initial and not is_renewal:
            trigger_b = rates.street_initial_annual - trigger_a
            trigger_b_agent = round(trigger_b * agency_split, 2)
            true_up_date = effective_date + timedelta(days=30)

            net_b, status_b = self._apply_liquidity_netting(agent_id, trigger_b_agent)

            entries.append(LedgerEntry(
                agent_id=agent_id,
                enrollment_id=enrollment_id,
                plan_id=plan_id,
                carrier=rates.carrier_name,
                entry_type="STREET_TRUE_UP",
                amount=net_b,
                due_date=true_up_date,
                status=status_b,
                description=f"Initial Year True-Up (30-day lag, ${rates.street_initial_annual} - ${trigger_a:.2f})",
                risk_alert=risk,
                raw_carrier_code=carrier_raw_status,
            ))

        # ---- HRA Bonus (FIX: now actually calculated) ----
        if rates.hra_gross > 0:
            hra_agent = round(rates.hra_gross * self.agent_hra_split * agency_split, 2)
            if hra_agent > 0:
                entries.append(LedgerEntry(
                    agent_id=agent_id,
                    enrollment_id=enrollment_id,
                    plan_id=plan_id,
                    carrier=rates.carrier_name,
                    entry_type="HRA_BONUS",
                    amount=hra_agent,
                    due_date=effective_date,
                    status="PAYABLE",
                    description=f"HRA Completion Bonus (${rates.hra_gross} × {self.agent_hra_split:.0%} split)",
                    risk_alert=risk,
                    raw_carrier_code=carrier_raw_status,
                ))

        # ---- Marketing Kicker (FIX: now actually calculated) ----
        if rates.marketing_kicker > 0:
            kicker_agent = round(rates.marketing_kicker * agency_split, 2)
            entries.append(LedgerEntry(
                agent_id=agent_id,
                enrollment_id=enrollment_id,
                plan_id=plan_id,
                carrier=rates.carrier_name,
                entry_type="MARKETING_KICKER",
                amount=kicker_agent,
                due_date=effective_date,
                status="PAYABLE",
                description=f"Carrier Marketing Kicker (${rates.marketing_kicker})",
                risk_alert=risk,
                raw_carrier_code=carrier_raw_status,
            ))

        # ---- HIERARCHY OVERRIDES ----
        if profile:
            base_commission = trigger_a  # Overrides calculated on gross, pre-split
            if profile.team_lead_id:
                amt = round(base_commission * OVERRIDE_RATES["team_lead"], 2)
                if amt > 0:
                    entries.append(LedgerEntry(
                        agent_id=profile.team_lead_id,
                        enrollment_id=enrollment_id,
                        plan_id=plan_id,
                        carrier=rates.carrier_name,
                        entry_type="OVERRIDE_TEAM_LEAD",
                        amount=amt,
                        due_date=effective_date,
                        status="PAYABLE",
                        description=f"Team Lead override on {agent_id} ({OVERRIDE_RATES['team_lead']:.0%})",
                        risk_alert="NONE",
                        raw_carrier_code=carrier_raw_status,
                    ))
            if profile.manager_id:
                amt = round(base_commission * OVERRIDE_RATES["manager"], 2)
                if amt > 0:
                    entries.append(LedgerEntry(
                        agent_id=profile.manager_id,
                        enrollment_id=enrollment_id,
                        plan_id=plan_id,
                        carrier=rates.carrier_name,
                        entry_type="OVERRIDE_MANAGER",
                        amount=amt,
                        due_date=effective_date,
                        status="PAYABLE",
                        description=f"Manager override on {agent_id} ({OVERRIDE_RATES['manager']:.0%})",
                        risk_alert="NONE",
                        raw_carrier_code=carrier_raw_status,
                    ))
            if profile.director_id:
                amt = round(base_commission * OVERRIDE_RATES["director"], 2)
                if amt > 0:
                    entries.append(LedgerEntry(
                        agent_id=profile.director_id,
                        enrollment_id=enrollment_id,
                        plan_id=plan_id,
                        carrier=rates.carrier_name,
                        entry_type="OVERRIDE_DIRECTOR",
                        amount=amt,
                        due_date=effective_date,
                        status="PAYABLE",
                        description=f"Director override on {agent_id} ({OVERRIDE_RATES['director']:.0%})",
                        risk_alert="NONE",
                        raw_carrier_code=carrier_raw_status,
                    ))

        return entries

    # -----------------------------------------------------------------
    # BATCH CSV PROCESSOR
    # -----------------------------------------------------------------
    def process_carrier_file(
        self,
        filepath: str,
        col_map: Optional[Dict[str, str]] = None,
    ) -> int:
        """
        Batch process a Producer Toolbox CSV export.

        Default expected columns (override with col_map):
            agent_id, enrollment_id, plan_id, effective_date,
            enrollment_date, status, carrier_balance, is_new_to_medicare

        Returns number of enrollments processed.
        """
        default_map = {
            "agent_id":           "agent_id",
            "enrollment_id":     "enrollment_id",
            "plan_id":           "plan_id",
            "effective_date":    "effective_date",
            "enrollment_date":   "enrollment_date",
            "status":            "status",
            "carrier_balance":   "carrier_balance",
            "is_new_to_medicare": "is_new_to_medicare",
        }
        cmap = col_map or default_map

        df = pd.read_csv(filepath)
        processed = 0
        skipped = 0

        for idx, row in df.iterrows():
            try:
                agent_id = str(row[cmap["agent_id"]]).strip()
                enrollment_id = str(row.get(cmap["enrollment_id"], f"ENR-{idx}")).strip()
                plan_id = str(row[cmap["plan_id"]]).strip()

                effective_date = pd.to_datetime(row[cmap["effective_date"]])
                enrollment_date = pd.to_datetime(
                    row.get(cmap["enrollment_date"], row[cmap["effective_date"]])
                )

                raw_status = str(row[cmap["status"]]).strip()
                balance = float(row.get(cmap["carrier_balance"], 0.0) or 0.0)

                ntm_raw = row.get(cmap["is_new_to_medicare"], False)
                is_ntm = str(ntm_raw).strip().upper() in ("TRUE", "1", "YES", "Y")

                entries = self.process_enrollment(
                    agent_id=agent_id,
                    enrollment_id=enrollment_id,
                    plan_id=plan_id,
                    effective_date=effective_date.to_pydatetime(),
                    enrollment_date=enrollment_date.to_pydatetime(),
                    carrier_raw_status=raw_status,
                    carrier_balance=balance,
                    is_new_to_medicare=is_ntm,
                )
                self.ledger.extend(entries)
                processed += 1

            except Exception as e:
                skipped += 1
                self.warnings.append(f"Row {idx}: {str(e)}")
                logger.error(f"Row {idx} skipped: {e}")

        logger.info(f"Processed {processed} enrollments, skipped {skipped}")
        return processed

    # -----------------------------------------------------------------
    # PAY RUN GENERATION
    # -----------------------------------------------------------------
    def generate_pay_run(self, pay_date: datetime) -> pd.DataFrame:
        """
        Aggregate all ledger entries into a per-agent pay run summary.
        Only includes entries due on or before pay_date.
        """
        payable_entries = [
            e for e in self.ledger
            if e.due_date <= pay_date and e.status not in ("HOLD", "PENDING")
        ]

        if not payable_entries:
            logger.warning("No payable entries found for this pay date.")
            return pd.DataFrame()

        rows = []
        for e in payable_entries:
            rows.append({
                "agent_id": e.agent_id,
                "enrollment_id": e.enrollment_id,
                "plan_id": e.plan_id,
                "carrier": e.carrier,
                "type": e.entry_type,
                "amount": e.amount,
                "status": e.status,
                "description": e.description,
                "risk_alert": e.risk_alert,
            })

        detail_df = pd.DataFrame(rows)

        # Aggregate per agent
        summary = detail_df.groupby("agent_id").agg(
            gross_earnings=pd.NamedAgg(
                column="amount",
                aggfunc=lambda x: round(x[x > 0].sum(), 2)
            ),
            total_chargebacks=pd.NamedAgg(
                column="amount",
                aggfunc=lambda x: round(x[x < 0].sum(), 2)
            ),
            net_payable=pd.NamedAgg(
                column="amount",
                aggfunc=lambda x: round(x.sum(), 2)
            ),
            enrollment_count=pd.NamedAgg(
                column="enrollment_id",
                aggfunc="nunique"
            ),
            risk_flag=pd.NamedAgg(
                column="risk_alert",
                aggfunc=lambda x: "HIGH" if "HIGH" in x.values else "NONE"
            ),
        ).reset_index()

        # Add agent name and agency from roster
        def enrich(row):
            profile = self.agent_roster.get(row["agent_id"])
            if profile:
                row["agent_name"] = profile.agent_name
                row["agency"] = profile.agency_name
            else:
                row["agent_name"] = "Unknown"
                row["agency"] = "Unknown"
            return row

        summary = summary.apply(enrich, axis=1)

        # Reorder columns
        col_order = [
            "agent_id", "agent_name", "agency",
            "enrollment_count", "gross_earnings",
            "total_chargebacks", "net_payable", "risk_flag"
        ]
        summary = summary[[c for c in col_order if c in summary.columns]]
        summary = summary.sort_values("net_payable", ascending=False)

        return summary

    # -----------------------------------------------------------------
    # EXPORTS
    # -----------------------------------------------------------------
    def export_pay_run(self, pay_date: datetime, output_path: str) -> str:
        """Export the pay run summary to CSV."""
        summary = self.generate_pay_run(pay_date)
        summary.to_csv(output_path, index=False)
        logger.info(f"Pay run exported to {output_path}")
        return output_path

    def export_detail_ledger(self, output_path: str) -> str:
        """Export full ledger detail to CSV."""
        rows = []
        for e in self.ledger:
            rows.append({
                "agent_id": e.agent_id,
                "enrollment_id": e.enrollment_id,
                "plan_id": e.plan_id,
                "carrier": e.carrier,
                "entry_type": e.entry_type,
                "amount": e.amount,
                "due_date": e.due_date.strftime("%Y-%m-%d"),
                "status": e.status,
                "description": e.description,
                "risk_alert": e.risk_alert,
                "raw_carrier_code": e.raw_carrier_code,
            })
        pd.DataFrame(rows).to_csv(output_path, index=False)
        logger.info(f"Detail ledger exported to {output_path}")
        return output_path

    def export_hold_report(self, output_path: str) -> str:
        """Export compliance hold report."""
        pd.DataFrame(self.hold_report).to_csv(output_path, index=False)
        logger.info(f"Hold report exported to {output_path}")
        return output_path

    def print_summary(self, pay_date: datetime) -> None:
        """Print pay run summary to console."""
        summary = self.generate_pay_run(pay_date)
        if summary.empty:
            print("No payable entries.")
            return

        print("\n" + "=" * 80)
        print(f"  INSUREitALL COMMISSION PAY RUN — {pay_date.strftime('%B %d, %Y')}")
        print("=" * 80)
        print(f"  Agents:          {len(summary)}")
        print(f"  Enrollments:     {summary['enrollment_count'].sum()}")
        print(f"  Gross Earnings:  ${summary['gross_earnings'].sum():,.2f}")
        print(f"  Chargebacks:     ${summary['total_chargebacks'].sum():,.2f}")
        print(f"  NET PAYABLE:     ${summary['net_payable'].sum():,.2f}")
        print(f"  Risk Flags:      {(summary['risk_flag'] == 'HIGH').sum()} agents")
        print(f"  Compliance Holds: {len(self.hold_report)} enrollments")
        print("=" * 80)
        print()
        print(summary.to_string(index=False))

        if self.hold_report:
            print(f"\n  {len(self.hold_report)} enrollments on COMPLIANCE HOLD:")
            for h in self.hold_report:
                print(f"   {h['agent_id']} | {h['enrollment_id']} | {h['hold_type']} | {h['raw_code']}")

        if self.warnings:
            print(f"\n  {len(self.warnings)} warnings:")
            for w in self.warnings[:10]:
                print(f"   {w}")
            if len(self.warnings) > 10:
                print(f"   ... and {len(self.warnings) - 10} more")

# =============================================================================
# 6. COMPLETE PAY RUN WORKFLOW
# =============================================================================

def run_friday_pay_cycle(
    carrier_file: str,
    roster_file: Optional[str] = None,
    pay_date: str = "2026-02-13",
    output_prefix: str = "pay_run",
    col_map: Optional[Dict[str, str]] = None,
) -> CommissionEngine:
    """
    End-to-end pay run execution.

    Args:
        carrier_file:  Path to Producer Toolbox CSV export
        roster_file:   Path to agent roster CSV (optional)
        pay_date:      Pay date as YYYY-MM-DD string
        output_prefix:  Prefix for output files
        col_map:       Column name mapping if your CSV uses different headers

    Returns:
        The engine instance with full ledger for inspection
    """
    engine = CommissionEngine()

    # Load roster if available
    if roster_file:
        engine.load_agent_roster(roster_file)

    # Process all enrollments
    count = engine.process_carrier_file(carrier_file, col_map=col_map)

    # Generate outputs
    dt = datetime.strptime(pay_date, "%Y-%m-%d")
    engine.export_pay_run(dt, f"{output_prefix}_summary_{pay_date}.csv")
    engine.export_detail_ledger(f"{output_prefix}_detail_{pay_date}.csv")
    engine.export_hold_report(f"{output_prefix}_holds_{pay_date}.csv")
    engine.print_summary(dt)

    return engine

# =============================================================================
# 7. EXAMPLE EXECUTION (validates the fix for agent CCNQPRMTSY)
# =============================================================================

if __name__ == "__main__":

    engine = CommissionEngine()

    # Register an agent
    engine.add_agent(AgentProfile(
        agent_id="CCNQPRMTSY",
        agent_name="Test Agent",
        agency_id="AGY-001",
        agency_name="Michael's Agency",
        team_lead_id="TL-001",
        manager_id="MGR-001",
        agency_split=1.0,
    ))

    # --- Scenario 1: New-to-Medicare Initial with negative balance ---
    print("\n--- SCENARIO 1: Initial + Negative Balance ---")
    entries = engine.process_enrollment(
        agent_id="CCNQPRMTSY",
        enrollment_id="ENR-20260201-001",
        plan_id="H5216-348",
        effective_date=datetime(2026, 2, 1),
        enrollment_date=datetime(2026, 1, 15),
        carrier_raw_status="ADVINT",
        carrier_balance=-222.04,
        is_new_to_medicare=True,
    )
    engine.ledger.extend(entries)
    for e in entries:
        print(f"  {e.entry_type}: ${e.amount:.2f} | {e.status} | {e.description}")

    # --- Scenario 2: Renewal (should NOT get true-up) ---
    print("\n--- SCENARIO 2: Renewal (no true-up) ---")
    entries2 = engine.process_enrollment(
        agent_id="CCNQPRMTSY",
        enrollment_id="ENR-20260201-002",
        plan_id="H5216-348",
        effective_date=datetime(2026, 2, 1),
        enrollment_date=datetime(2026, 1, 10),
        carrier_raw_status="ADVREN",
        carrier_balance=0.0,
        is_new_to_medicare=False,
    )
    engine.ledger.extend(entries2)
    for e in entries2:
        print(f"  {e.entry_type}: ${e.amount:.2f} | {e.status} | {e.description}")

    # --- Scenario 3: Chargeback (should calculate actual $) ---
    print("\n--- SCENARIO 3: Chargeback (actual amount) ---")
    entries3 = engine.process_enrollment(
        agent_id="CCNQPRMTSY",
        enrollment_id="ENR-20260201-003",
        plan_id="H4514-021",
        effective_date=datetime(2025, 11, 1),
        enrollment_date=datetime(2025, 10, 15),
        carrier_raw_status="CHGBCK-VOL",
    )
    engine.ledger.extend(entries3)
    for e in entries3:
        print(f"  {e.entry_type}: ${e.amount:.2f} | {e.status} | {e.description}")

    # --- Scenario 4: CX with dynamic month code ---
    print("\n--- SCENARIO 4: Cancellation CX-03-26 (regex match) ---")
    entries4 = engine.process_enrollment(
        agent_id="CCNQPRMTSY",
        enrollment_id="ENR-20260301-004",
        plan_id="H3146-006",
        effective_date=datetime(2026, 1, 1),
        enrollment_date=datetime(2025, 12, 10),
        carrier_raw_status="CX-03-26",
    )
    engine.ledger.extend(entries4)
    for e in entries4:
        print(f"  {e.entry_type}: ${e.amount:.2f} | {e.status} | {e.description}")

    # --- Scenario 5: Compliance Hold ---
    print("\n--- SCENARIO 5: Compliance Hold ---")
    entries5 = engine.process_enrollment(
        agent_id="CCNQPRMTSY",
        enrollment_id="ENR-20260201-005",
        plan_id="H1036-291",
        effective_date=datetime(2026, 2, 1),
        enrollment_date=datetime(2026, 1, 20),
        carrier_raw_status="NO CERT",
    )
    engine.ledger.extend(entries5)
    for e in entries5:
        print(f"  {e.entry_type}: ${e.amount:.2f} | {e.status} | {e.description}")

    # --- Print full pay run ---
    engine.print_summary(datetime(2026, 2, 13))
