import json

import pytest
from pydantic import ValidationError

from app.models import (
    ClaimCheck,
    EvidenceItem,
    Finding,
    ResearchBrief,
    ResearchPackage,
    Source,
    VerificationReport,
)


def make_source(i=1, url=None):
    return Source(source_id=f"S{i}", title=f"Title {i}", url=url or f"https://example{i}.org/page",
                  snippet="text", provider="test")


def test_source_rejects_invalid_urls():
    for bad in ["not a url", "ftp://example.org/x", "https://", "javascript:alert(1)", "https://exa mple.org"]:
        with pytest.raises(ValidationError):
            make_source(url=bad)


def test_source_id_format_and_null_date():
    source = make_source()
    assert source.publication_date is None
    with pytest.raises(ValidationError):
        Source(source_id="X1", title="t", url="https://example.org", provider="p")


def test_package_rejects_unknown_and_duplicate_source_ids():
    finding = Finding(finding_id="F1", statement="s", source_ids=["S9"])
    with pytest.raises(ValidationError, match="unknown sources"):
        ResearchPackage(topic="t", research_queries=["q"], sources=[make_source()], findings=[finding],
                        search_provider="p", research_status="limited")
    with pytest.raises(ValidationError, match="duplicate"):
        ResearchPackage(topic="t", research_queries=["q"], sources=[make_source(1), make_source(1, "https://b.org")],
                        search_provider="p", research_status="limited")


def test_package_requires_nonempty_topic():
    with pytest.raises(ValidationError):
        ResearchPackage(topic="   ", research_queries=["q"], sources=[], search_provider="p", research_status="limited")


def test_retrieved_evidence_finding_needs_sources():
    with pytest.raises(ValidationError):
        Finding(finding_id="F1", statement="s", basis="retrieved_evidence", source_ids=[])
    assert Finding(finding_id="F1", statement="s", basis="model_interpretation").source_ids == []


def test_brief_missing_source_references_are_rejected_with_context():
    data = {
        "topic": "t", "executive_summary": "s", "key_points": ["k"], "recommended_blog_angle": "a",
        "supporting_evidence": [{"point": "p", "source_ids": ["S7"]}],
    }
    with pytest.raises(ValidationError, match="unknown source IDs"):
        ResearchBrief.model_validate(data, context={"valid_source_ids": {"S1", "S2"}})
    data["supporting_evidence"][0]["source_ids"] = []
    with pytest.raises(ValidationError):
        ResearchBrief.model_validate(data)


def test_evidence_item_accepts_known_ids():
    item = EvidenceItem.model_validate({"point": "p", "source_ids": ["S1"]}, context={"valid_source_ids": {"S1"}})
    assert item.source_ids == ["S1"]


def test_claim_status_validation():
    with pytest.raises(ValidationError):
        ClaimCheck(claim_id="C1", claim_text="c", status="probably", explanation="e")
    with pytest.raises(ValidationError, match="cites no sources"):
        ClaimCheck(claim_id="C1", claim_text="c", status="supported", explanation="e")
    ok = ClaimCheck(claim_id="C1", claim_text="c", status="unverifiable", explanation="e")
    assert ok.status == "unverifiable"


def claim(i, status, ids=()):
    return ClaimCheck(claim_id=f"C{i}", claim_text=f"claim {i}", status=status,
                      supporting_source_ids=list(ids), explanation="e", recommended_correction="fix")


def test_report_from_checks_groups_and_scores():
    checks = [claim(1, "supported", ["S1"]), claim(2, "partially_supported", ["S2"]),
              claim(3, "unsupported"), claim(4, "unverifiable")]
    report = VerificationReport.from_checks(checks, total_sources=4, limitations=["l"])
    assert report.overall_status == "needs_revision"
    assert report.claims_checked == 4
    assert [c.claim_id for c in report.material_issues] == ["C3"]
    assert report.source_coverage.claim_support_ratio == 0.5
    assert report.source_coverage.source_utilization_ratio == 0.5
    assert len(report.corrections) == 3  # supported claims need no correction


@pytest.mark.parametrize("statuses,expected", [
    (["supported"], "pass"),
    (["supported", "unverifiable"], "pass_with_warnings"),
    (["supported", "contradictory"], "needs_revision"),
    (["unverifiable", "unsupported"], "insufficient_evidence"),
    ([], "insufficient_evidence"),
])
def test_overall_status_rules(statuses, expected):
    checks = [claim(i, s, ["S1"] if s in ("supported", "partially_supported") else []) for i, s in enumerate(statuses, 1)]
    assert VerificationReport.from_checks(checks, 2, []).overall_status == expected


def test_report_count_must_match():
    report = VerificationReport.from_checks([claim(1, "supported", ["S1"])], 1, [])
    data = report.model_dump()
    data["claims_checked"] = 5
    with pytest.raises(ValidationError):
        VerificationReport.model_validate(data)


def test_models_serialise_to_valid_json_and_back():
    report = VerificationReport.from_checks([claim(1, "supported", ["S1"])], 1, ["limit"])
    text = report.model_dump_json()
    assert VerificationReport.model_validate(json.loads(text)) == report
