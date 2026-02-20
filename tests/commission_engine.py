from datetime import datetime

import pytest

from commission_engine import AgentProfile, CommissionEngine, resolve_status


@pytest.fixture
def engine() -> CommissionEngine:
    e = CommissionEngine()
    e.add_agent(
        AgentProfile(
            agent_id="CCNQPRMTSY",
            agent_name="Test Agent",
            agency_id="AGY-001",
            agency_name="Michael's Agency",
            team_lead_id="TL-001",
            manager_id="MGR-001",
            agency_split=1.0,
        )
    )
    return e


def test_resolve_status_exact() -> None:
    assert resolve_status("ADVINT") == "ACTIVE_INITIAL"
    assert resolve_status("ADVREN") == "ACTIVE_RENEWAL"
    assert resolve_status("CHGBCK-VOL") == "CHARGEBACK_VOLUNTARY"
    assert resolve_status("NO CERT") == "HOLD_NO_CERT"
    assert resolve_status("PEND") == "PENDING"


def test_resolve_status_regex_cx() -> None:
    assert resolve_status("CX-03-26") == "CANCELLED_TERM"
    assert resolve_status("CX-12-26") == "CANCELLED_TERM"


def test_resolve_status_unknown_defaults_to_pending() -> None:
    assert resolve_status("COMPLETELY_UNKNOWN_CODE") == "PENDING"


def test_initial_enrollment_with_negative_balance(engine: CommissionEngine) -> None:
    """New-to-Medicare initial enrollment with negative carrier balance should
    produce a STREET_IMMEDIATE entry (possibly offset), a STREET_TRUE_UP entry,
    an HRA bonus, and hierarchy overrides."""
    engine.set_agent_balance("CCNQPRMTSY", -222.04)
    entries = engine.process_enrollment(
        agent_id="CCNQPRMTSY",
        enrollment_id="ENR-001",
        plan_id="H5216-348",
        effective_date=datetime(2026, 2, 1),
        enrollment_date=datetime(2026, 1, 15),
        carrier_raw_status="ADVINT",
        carrier_balance=0.0,
        is_new_to_medicare=True,
    )
    entry_types = [e.entry_type for e in entries]
    assert "STREET_IMMEDIATE" in entry_types
    assert "STREET_TRUE_UP" in entry_types
    assert "HRA_BONUS" in entry_types
    assert "OVERRIDE_TEAM_LEAD" in entry_types
    assert "OVERRIDE_MANAGER" in entry_types

    immediate = next(e for e in entries if e.entry_type == "STREET_IMMEDIATE")
    assert immediate.risk_alert == "HIGH"


def test_renewal_no_true_up(engine: CommissionEngine) -> None:
    """Renewal enrollment should NOT produce a STREET_TRUE_UP entry."""
    entries = engine.process_enrollment(
        agent_id="CCNQPRMTSY",
        enrollment_id="ENR-002",
        plan_id="H5216-348",
        effective_date=datetime(2026, 2, 1),
        enrollment_date=datetime(2026, 1, 10),
        carrier_raw_status="ADVREN",
        carrier_balance=0.0,
        is_new_to_medicare=False,
    )
    entry_types = [e.entry_type for e in entries]
    assert "STREET_IMMEDIATE" in entry_types
    assert "STREET_TRUE_UP" not in entry_types


def test_chargeback_voluntary(engine: CommissionEngine) -> None:
    """Voluntary chargeback should produce a single CHARGEBACK entry with negative amount."""
    entries = engine.process_enrollment(
        agent_id="CCNQPRMTSY",
        enrollment_id="ENR-003",
        plan_id="H4514-021",
        effective_date=datetime(2025, 11, 1),
        enrollment_date=datetime(2025, 10, 15),
        carrier_raw_status="CHGBCK-VOL",
    )
    assert len(entries) == 1
    assert entries[0].entry_type == "CHARGEBACK"
    assert entries[0].amount < 0
    assert entries[0].status == "CLAWBACK"


def test_cancellation_regex(engine: CommissionEngine) -> None:
    """CX-MM-YY status codes should be resolved via regex to CANCELLED_TERM."""
    entries = engine.process_enrollment(
        agent_id="CCNQPRMTSY",
        enrollment_id="ENR-004",
        plan_id="H3146-006",
        effective_date=datetime(2026, 1, 1),
        enrollment_date=datetime(2025, 12, 10),
        carrier_raw_status="CX-03-26",
    )
    assert len(entries) == 1
    assert entries[0].entry_type == "CHARGEBACK"
    assert entries[0].amount < 0


def test_compliance_hold(engine: CommissionEngine) -> None:
    """NO CERT status should create a COMPLIANCE_HOLD entry with $0 amount."""
    entries = engine.process_enrollment(
        agent_id="CCNQPRMTSY",
        enrollment_id="ENR-005",
        plan_id="H1036-291",
        effective_date=datetime(2026, 2, 1),
        enrollment_date=datetime(2026, 1, 20),
        carrier_raw_status="NO CERT",
    )
    assert len(entries) == 1
    assert entries[0].entry_type == "COMPLIANCE_HOLD"
    assert entries[0].amount == 0.0
    assert entries[0].status == "HOLD"
    assert len(engine.hold_report) == 1


def test_liquidity_netting_deficit_absorbed(engine: CommissionEngine) -> None:
    """If deficit exceeds earnings, the agent should receive $0 and the deficit
    should be reduced."""
    engine.set_agent_balance("CCNQPRMTSY", -10_000.0)
    entries = engine.process_enrollment(
        agent_id="CCNQPRMTSY",
        enrollment_id="ENR-006",
        plan_id="H5216-348",
        effective_date=datetime(2026, 2, 1),
        enrollment_date=datetime(2026, 1, 15),
        carrier_raw_status="ADVREN",
        carrier_balance=0.0,
        is_new_to_medicare=False,
    )
    immediate = next(e for e in entries if e.entry_type == "STREET_IMMEDIATE")
    assert immediate.amount == 0.0


def test_generate_pay_run_summary(engine: CommissionEngine) -> None:
    """generate_pay_run should return a non-empty DataFrame after processing enrollments."""
    entries = engine.process_enrollment(
        agent_id="CCNQPRMTSY",
        enrollment_id="ENR-007",
        plan_id="H4514-021",
        effective_date=datetime(2026, 2, 1),
        enrollment_date=datetime(2026, 1, 1),
        carrier_raw_status="ADVREN",
        carrier_balance=0.0,
        is_new_to_medicare=False,
    )
    engine.ledger.extend(entries)

    summary = engine.generate_pay_run(datetime(2026, 2, 13))
    assert not summary.empty
    assert "net_payable" in summary.columns
    row = summary[summary["agent_id"] == "CCNQPRMTSY"]
    assert not row.empty
    assert row["net_payable"].values[0] > 0
