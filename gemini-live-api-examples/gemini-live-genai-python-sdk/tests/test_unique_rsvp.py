"""Unique RSVP counting: deduped by phone WITHIN a campaign, summed ACROSS campaigns."""

from datetime import datetime, timezone


def _mk_campaign(eo_db, admin, name):
    return eo_db.create_campaign(name, datetime.now(timezone.utc).isoformat(), admin, 4, 3, 1)


def test_same_number_in_two_campaigns_counts_twice(fresh_eo_db):
    eo_db = fresh_eo_db
    eo_db.init()
    admin = eo_db.create_user("a", "A", "h", "s", role="eo_admin")
    shivi = "+919000000001"
    c1 = _mk_campaign(eo_db, admin, "C1")
    c2 = _mk_campaign(eo_db, admin, "C2")
    eo_db.add_campaign_contacts(c1, [{"id": None, "phone": shivi, "name": "Shivi"}])
    eo_db.add_campaign_contacts(c2, [{"id": None, "phone": shivi, "name": "Shivi"}])
    eo_db.cc_set_outcome_by_phone(c1, shivi, "yes")
    eo_db.cc_set_outcome_by_phone(c2, shivi, "yes")

    assert eo_db.campaign_rsvp_yes_unique(c1) == 1
    assert eo_db.campaign_rsvp_yes_unique(c2) == 1
    assert eo_db.rsvp_yes_unique_totals()["yes"] == 2   # once per campaign


def test_duplicate_number_within_one_campaign_counts_once(fresh_eo_db):
    eo_db = fresh_eo_db
    eo_db.init()
    admin = eo_db.create_user("a", "A", "h", "s", role="eo_admin")
    shivi = "+919000000001"
    c1 = _mk_campaign(eo_db, admin, "C1")
    c2 = _mk_campaign(eo_db, admin, "C2")
    # Shivi appears TWICE in C1 (campaigns created before phone-dedupe could hold dups)
    eo_db.add_campaign_contacts(c1, [
        {"id": None, "phone": shivi, "name": "Shivi"},
        {"id": None, "phone": shivi, "name": "Shivi dup"},
    ])
    eo_db.add_campaign_contacts(c2, [{"id": None, "phone": shivi, "name": "Shivi"}])
    for row in eo_db.list_campaign_contacts(c1)["items"]:
        eo_db.cc_update(row["id"], rsvp_outcome="yes")
    eo_db.cc_set_outcome_by_phone(c2, shivi, "yes")

    assert eo_db.campaign_rsvp_yes_unique(c1) == 1      # deduped within the campaign
    assert eo_db.rsvp_yes_unique_totals()["yes"] == 2   # 1 (C1) + 1 (C2)


def test_totals_responded_and_agent_scoping(fresh_eo_db):
    eo_db = fresh_eo_db
    eo_db.init()
    admin = eo_db.create_user("a", "A", "h", "s", role="eo_admin")
    c1 = _mk_campaign(eo_db, admin, "C1")
    c2 = _mk_campaign(eo_db, admin, "C2")
    eo_db.add_campaign_contacts(c1, [
        {"id": None, "phone": "+919000000002", "name": "No-sayer"},
        {"id": None, "phone": "+919000000003", "name": "Yes-sayer"},
        {"id": None, "phone": "+919000000005", "name": "Never-reached"},
    ])
    eo_db.add_campaign_contacts(c2, [{"id": None, "phone": "+919000000004", "name": "Z"}])
    eo_db.cc_set_outcome_by_phone(c1, "+919000000002", "no")
    eo_db.cc_set_outcome_by_phone(c1, "+919000000003", "yes")
    eo_db.cc_set_outcome_by_phone(c2, "+919000000004", "yes")

    assert eo_db.rsvp_yes_unique_totals() == {"yes": 2, "responded": 3}
    assert eo_db.rsvp_yes_unique_totals([c1]) == {"yes": 1, "responded": 2}  # agent scope
    assert eo_db.rsvp_yes_unique_totals([]) == {"yes": 0, "responded": 0}    # agent with no campaigns
