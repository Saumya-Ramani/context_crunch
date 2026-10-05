"""A labelled evaluation set for ContextCrunch.

Each case is one history item on trial, with the goal, the tool that produced it
and the assistant findings that came after it. The ``label`` is what a careful
human would say should happen, and it is the ground truth the profile tuner
measures itself against.

The cases cover the junk patterns every agent produces, plus the material that
must survive:

- **consumed verbosity** - a big tool output whose insight was already extracted;
- **process boilerplate** - an artifact of doing rather than knowing;
- **superseded intermediates** - an earlier result made obsolete by a later one;
- **irreplaceable** - the only copy of a finding, a citable passage, a unique error.

Fifteen cases, five per archetype. That is far too few to tune thresholds on, and
this file exists to make the tuning *measurable* rather than to be the final
labelled set.
"""

from __future__ import annotations

from dataclasses import dataclass

KEEP = "keep"
TRUNCATE = "truncate"
DROP = "drop"


@dataclass(frozen=True)
class Case:
    """One labelled history item."""

    name: str
    goal: str
    tool: str
    content: str
    later_findings: tuple[str, ...]
    label: str
    pattern: str
    age: int = 6


def _code_goal() -> str:
    return "Fix the rounding bug in the billing service and make the tests pass."


def _support_goal() -> str:
    return "Resolve the double-charge dispute for customer 4471 and apply the discount."


def _research_goal() -> str:
    return "Assess whether Acme's gross-margin expansion is sustainable; cite sources."


CASES: tuple[Case, ...] = (
    # --- coding ----------------------------------------------------------------
    Case(
        name="read_file dump consumed into a finding",
        goal=_code_goal(),
        tool="read_file",
        content=(
            "src/billing/pricing.py\n"
            "def round_price(total):\n"
            "    return int(total) + 1\n"
            "\n"
            "class Invoice:\n"
            "    def apply_discount(self, rate):\n"
            "        self.total = round_price(self.total * (1 - rate))\n"
            "    # ... 180 more lines of pricing helpers ...\n"
        )
        * 12,
        later_findings=(
            "Found the bug: round_price uses int() so it truncates instead of rounding.",
            "Invoice.apply_discount is the only caller that matters here.",
        ),
        label=DROP,
        pattern="consumed verbosity",
    ),
    Case(
        name="pip install log",
        goal=_code_goal(),
        tool="run_in_terminal",
        content="Collecting pytest\nDownloading pluggy-1.5.0\nInstalling collected packages\n"
        "Successfully installed pytest-8.2.1 pluggy-1.5.0\n" * 40,
        later_findings=("Installed pytest 8.2.1; the test suite can now run.",),
        label=DROP,
        pattern="process boilerplate",
    ),
    Case(
        name="first pytest run, superseded by the second",
        goal=_code_goal(),
        tool="run_tests",
        content=(
            "platform linux -- Python 3.13.1, pytest-8.2.1, pluggy-1.5.0\n"
            "rootdir: /workspace/billing\n"
            "collected 147 items\n\n"
            "tests/test_pricing.py::test_round_half_up FAILED\n"
            "tests/test_invoices.py ..........                    [ 10%]\n"
            "E   AssertionError: expected Decimal('101.50') but got Decimal('101.00')\n\n"
            "=== 1 failed, 146 passed in 4.21s ===\n"
        )
        * 3,
        later_findings=(
            "The failing test was the rounding bug I had just patched.",
            "Re-ran the suite: 147 passed in 4.05s.",
        ),
        label=DROP,
        pattern="superseded intermediate",
    ),
    Case(
        name="the only copy of the root cause",
        goal=_code_goal(),
        tool="run_tests",
        content=(
            "E   AssertionError: expected Decimal('101.50') but got Decimal('101.00')\n"
            "    billing/pricing.py:42: in round_price\n"
            "        return int(total) + Decimal('1')\n"
            "E   where total = Decimal('100.50')\n"
        )
        * 4,
        later_findings=("The root cause is the int() cast in round_price.",),
        label=KEEP,
        pattern="irreplaceable",
    ),
    Case(
        name="final passing test run",
        goal=_code_goal(),
        tool="run_tests",
        content="147 passed in 4.05s\n",
        later_findings=("All 147 tests pass. The rounding bug is fixed.",),
        label=KEEP,
        pattern="irreplaceable",
        age=3,
    ),
    # --- support ---------------------------------------------------------------
    Case(
        name="KB search results already read",
        goal=_support_goal(),
        tool="search_kb",
        content="doc-118 Double charge policy ... doc-204 Refund SLA ... doc-331 Discount rules "
        "... doc-412 Escalation matrix ... doc-519 Chargeback guide ...\n" * 8,
        later_findings=(
            "Policy doc-331 allows a 10% goodwill discount for a verified double charge.",
            "Refund SLA from doc-204 is 5 business days.",
        ),
        label=DROP,
        pattern="consumed verbosity",
    ),
    Case(
        name="billing ledger before the credit was applied",
        goal=_support_goal(),
        tool="query_billing",
        content="2026-03 charge 45.00 PAID | 2026-04 charge 45.00 PAID | 2026-05 charge 45.00 PAID "
        "| 2026-06 charge 90.00 PAID | 2026-07 charge 45.00 PAID\n" * 6,
        later_findings=(
            "Charged twice in June: one 45.00 and one 90.00.",
            "Applied a 45.00 credit and the 10% goodwill discount.",
        ),
        label=DROP,
        pattern="superseded intermediate",
    ),
    Case(
        name="ticket status update confirmation",
        goal=_support_goal(),
        tool="update_ticket",
        content="Ticket 8871 updated: status=resolved, priority=low, assignee=team-billing\n"
        "Notification queued for customer 4471\n" * 3,
        later_findings=("Resolved the ticket and notified the customer.",),
        label=DROP,
        pattern="process boilerplate",
    ),
    Case(
        name="customer's own description of the problem",
        goal=_support_goal(),
        tool="send_email",
        content=(
            "I was charged 90 dollars on the 6th but my plan is 45 dollars. "
            "I have a screenshot from my bank showing two charges on the same day."
        ),
        later_findings=("Customer reports a duplicate charge with bank evidence.",),
        label=KEEP,
        pattern="irreplaceable",
    ),
    Case(
        name="current month invoice the agent is reasoning about",
        goal=_support_goal(),
        tool="query_billing",
        content="2026-07 charge 45.00 PAID | 2026-06 charge 90.00 PAID\n",
        later_findings=("The July charge is the disputed one.",),
        label=KEEP,
        pattern="irreplaceable",
        age=4,
    ),
    # --- research --------------------------------------------------------------
    Case(
        name="10-K with citable passages buried in boilerplate",
        goal=_research_goal(),
        tool="fetch_filing",
        content=(
            "## Risk Factors\n"
            "The company is subject to litigation and regulatory risk. p. 12\n\n"
            "## MD&A\n"
            "Gross margin improved 500bps to 47%, driven by renegotiated component pricing "
            "across the top ten suppliers. p. 22\n\n"
            "## Properties\n"
            "The company leases 14 facilities. p. 38\n\n"
            "## Executive Compensation\n"
            "Summary compensation table for named officers. p. 45\n\n"
            "## Supply Chain\n"
            "Two suppliers accounted for 61% of cost of goods sold. p. 87\n\n"
        )
        * 5,
        later_findings=(
            "Gross margin rose 500bps to 47% (p. 22), driven by component repricing.",
            "Two suppliers are 61% of COGS (p. 87), so concentration risk is real.",
        ),
        label=TRUNCATE,
        pattern="needs splitting",
    ),
    Case(
        name="earnings call transcript",
        goal=_research_goal(),
        tool="fetch_transcript",
        content=(
            "OPERATOR: Good morning, and welcome to the third quarter call.\n\n"
            "CFO: Revenue came in at 412 million, up 6 percent year over year.\n\n"
            "ANALYST: Can you speak to the component cost savings?\n\n"
            "CFO: Yes. We renegotiated with our top ten suppliers and that is the bulk of "
            "the margin improvement we discussed earlier this year.\n\n"
        )
        * 6,
        later_findings=("CFO confirms component repricing drove the margin gain.",),
        label=TRUNCATE,
        pattern="needs splitting",
    ),
    Case(
        name="citation verification report",
        goal=_research_goal(),
        tool="check_citations",
        content=(
            "checking ref [1] p.22 ok\nchecking ref [2] p.87 ok\n"
            "checking ref [3] page mismatch (said 22, found 23)\n"
            "checking ref [4] p.19 ok\nchecking ref [5] p.45 ok\n"
            "5 refs checked, 1 mismatch\n"
        )
        * 5,
        later_findings=("All memo citations verified except ref [3].",),
        label=DROP,
        pattern="process boilerplate",
    ),
    Case(
        name="the supplier concentration note, citable",
        goal=_research_goal(),
        tool="fetch_filing",
        content=(
            "## Supply Chain\n"
            "Two suppliers accounted for 61% of cost of goods sold for fiscal 2025, "
            "purchased under multi-year agreements without volume commitments. p. 87\n"
        )
        * 2,
        later_findings=("Concentration risk is the main threat to the margin story.",),
        label=KEEP,
        pattern="irreplaceable",
    ),
    Case(
        name="competitor gross margin figures",
        goal=_research_goal(),
        tool="fetch_filing",
        content="## MD&A\nGross margin was flat at 31% across fiscal 2024 and 2025. p. 19\n" * 2,
        later_findings=("Competitors are flat, so Acme's gap is procurement-driven.",),
        label=KEEP,
        pattern="irreplaceable",
    ),
)


def cases_for(archetype: str) -> tuple[Case, ...]:
    """Return the cases belonging to one archetype."""
    if archetype == "coding":
        return tuple(case for case in CASES if "billing service" in case.goal)
    if archetype == "support":
        return tuple(case for case in CASES if "double-charge" in case.goal)
    if archetype == "research":
        return tuple(case for case in CASES if "margin" in case.goal)
    return CASES