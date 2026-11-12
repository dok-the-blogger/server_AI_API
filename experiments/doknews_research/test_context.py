"""Lossless context ordering, fresh research windows and legacy decoding; offline."""
from copy import deepcopy
import json
import unittest

import context


RUN_ID = "run_context_test"
AS_OF = "2026-11-12T12:00:00Z"


def snapshot():
    return {
        "schema_version": 1,
        "task": {"name": "Media research", "topic": "media", "time_window_hours": 48},
        "captured_at": "2026-11-12T10:00:00Z",
        "coverage": {"archive": "partial", "memory": "complete", "radar": "partial",
                     "missing": ["Archive covers only the supplied IDs; radar pagination is incomplete."]},
        "archive_articles": [
            {"article_id": "b", "version": 2, "body_text": "Second article", "read_at": "2026-11-12T09:00:00Z",
             "updated_at": "2026-11-10T01:00:00Z", "full_text_read": True},
            {"article_id": "a", "version": 1, "body_text": "First article" + context.MEMORY_MARKER,
             "full_text_included": True, "publication_date": {"kind": "known", "value": "2026-11-09"},
             "metadata": {"original_order": ["second", "first"]}, "retrieved_at": None},
            {"article_id": "a", "version": 1, "body_text": "First article" + context.MEMORY_MARKER,
             "full_text_included": True, "publication_date": {"kind": "known", "value": "2026-11-09"},
             "metadata": {"original_order": ["second", "first"]}, "retrieved_at": "2026-11-12T09:30:00Z"},
        ],
        "memory_records": [
            {"memory_id": "m2", "revision": 3, "body": "Owner preference", "review_at": "2026-12-01",
             "read_interval": {"start": "2026-11-12T09:00:00Z", "end": "2026-11-12T10:00:00Z"}},
            {"memory_id": "m1", "revision": 2, "body": "Earlier editorial hypothesis",
             "sources": [{"url": "https://example.org/source", "rank": 2}, {"rank": 1}],
             "fetched_at": "2026-11-12T10:00:00Z"},
        ],
        "radar_signals": [{"signal_id": "s1", "version": 2, "published_at": "2026-11-11T14:00:00Z",
                           "discovered_at": "2026-11-12T08:00:00Z", "text": "A changed signal"}],
        "additional_provenance": {"query": "recent media", "next_cursor": "not-yet-read", "limits": [30, 5]},
    }


class StableContextTests(unittest.TestCase):
    def build(self, value=None, run_id=RUN_ID, **kwargs):
        return context.build_context(snapshot() if value is None else value, run_id,
                                     research_as_of=kwargs.pop("research_as_of", AS_OF), **kwargs)

    def test_ids_clocks_observations_and_source_order_do_not_change_stable_prefix(self):
        original = snapshot()
        first = self.build(original)
        later = deepcopy(original)
        later["captured_at"] = "2026-11-12T13:00:00Z"
        later["coverage"]["missing"].append("A new coverage gap.")
        later["radar_signals"] = [{"signal_id": "s2", "version": 1, "text": "A different new signal"}]
        for name in context.COLLECTIONS:
            later[name].reverse()
            for record in later[name]:
                for key in context.READ_FIELDS.intersection(record):
                    record[key] = {"read_at": "later"} if key == "read_interval" else "2026-11-12T13:00:00Z"
        def reversed_keys(value):
            if isinstance(value, dict):
                return {key: reversed_keys(item) for key, item in reversed(list(value.items()))}
            if isinstance(value, list):
                return [reversed_keys(item) for item in value]
            return value
        second = self.build(reversed_keys(later), run_id="a-different-run",
                            research_as_of="2026-11-12T14:00:00Z")
        self.assertEqual(first.stable_prefix, second.stable_prefix)
        self.assertNotIn(RUN_ID, first.stable_prefix)
        self.assertNotIn(AS_OF, first.stable_prefix)
        for name in ("archive_context_sha256", "memory_context_sha256"):
            self.assertEqual(first.metadata[name], second.metadata[name])
        self.assertNotEqual(first.metadata["snapshot_sha256"], second.metadata["snapshot_sha256"])

    def test_article_and_memory_semantic_changes_invalidate_their_own_context(self):
        value = snapshot()
        original = self.build(value)
        article = deepcopy(value)
        article["archive_articles"][0]["body_text"] += " A correction."
        article["archive_articles"][0]["version"] += 1
        changed = self.build(article)
        self.assertNotEqual(original.metadata["archive_context_sha256"], changed.metadata["archive_context_sha256"])
        self.assertEqual(original.metadata["memory_context_sha256"], changed.metadata["memory_context_sha256"])
        memory = deepcopy(value)
        memory["memory_records"][0]["revision"] += 1
        memory["memory_records"][0]["body"] = "A revised owner preference"
        changed = self.build(memory)
        self.assertEqual(original.metadata["archive_context_sha256"], changed.metadata["archive_context_sha256"])
        self.assertNotEqual(original.metadata["memory_context_sha256"], changed.metadata["memory_context_sha256"])
        updated = deepcopy(value)
        updated["archive_articles"][0]["updated_at"] = "2026-11-11T01:00:00Z"
        self.assertNotEqual(original.metadata["archive_context_sha256"], self.build(updated).metadata["archive_context_sha256"])

    def test_round_trip_preserves_all_facts_versions_duplicates_order_and_coverage(self):
        original = snapshot()
        before = deepcopy(original)
        built = self.build(original, research_mode="new_signals")
        restored = context.extract_snapshot(built.input, {**built.metadata, "run_id": RUN_ID})
        self.assertEqual(restored, original)
        self.assertEqual(original, before)
        self.assertEqual(len(restored["archive_articles"]), 3)
        self.assertEqual(restored["memory_records"][1]["sources"], original["memory_records"][1]["sources"])

    def test_window_ends_at_research_time_in_utc_not_capture_time(self):
        built = self.build(research_as_of="2026-11-12T15:30:00+03:00")
        self.assertEqual(built.metadata["research_as_of"], "2026-11-12T12:30:00+00:00")
        self.assertEqual(built.metadata["research_window"], {
            "start_at": "2026-11-10T12:30:00+00:00", "end_at": "2026-11-12T12:30:00+00:00", "hours": 48})
        self.assertIn('"captured_at": "2026-11-12T10:00:00Z"', built.input)
        self.assertIn("Конец новостного окна — research_as_of, а не captured_at", built.input)
        for value in ("2026-11-12T09:00:00Z", "2026-11-12T12:00:00", "not-a-date"):
            with self.subTest(value=value), self.assertRaises(context.ContextError):
                self.build(research_as_of=value)

    def test_targeted_mode_does_not_replace_empty_radar_with_a_broad_search(self):
        value = snapshot()
        value["radar_signals"] = []
        built = self.build(value, research_mode="new_signals")
        self.assertIn("Не начинай новый широкий обзор всего интернета", built.input)
        self.assertIn("Если сигналов нет, не запускай поиск", built.input)
        self.assertEqual(context.extract_snapshot(built.input, {**built.metadata, "run_id": RUN_ID}), value)

    def test_legacy_decoder_and_mixed_layout_rejection(self):
        value = snapshot()
        legacy = f"Выполни исследование. DOK_RESEARCH_RUN_ID={RUN_ID}.\n\n" + json.dumps(value, ensure_ascii=False)
        self.assertEqual(context.extract_snapshot(legacy, {"run_id": RUN_ID}), value)
        built = self.build(value)
        for body, manifest in ((legacy, {**built.metadata, "run_id": RUN_ID}),
                               (built.input, {"run_id": RUN_ID}),
                               (legacy, {"run_id": RUN_ID, "research_mode": "new_signals"}),
                               (built.input, {**built.metadata, "run_id": RUN_ID, "context_layout_version": True})):
            with self.subTest(manifest=manifest), self.assertRaises(context.ContextError):
                context.extract_snapshot(body, manifest)

    def test_changed_run_window_source_mapping_or_extra_dynamic_field_is_rejected(self):
        built = self.build()
        manifest = {**built.metadata, "run_id": RUN_ID}
        prefix, text = built.input.split(context.DYNAMIC_MARKER, 1)
        dynamic = json.loads(text)
        alterations = []
        changed = deepcopy(dynamic)
        changed["run_id"] = "other-run"
        alterations.append(changed)
        changed = deepcopy(dynamic)
        changed["research_window"]["end_at"] = "2026-11-12T11:00:00Z"
        alterations.append(changed)
        changed = deepcopy(dynamic)
        changed["source_record_order"]["archive_articles"] = [0, 0, 2]
        alterations.append(changed)
        changed = deepcopy(dynamic)
        changed["read_observations"]["archive_articles"][0]["version"] = 999
        alterations.append(changed)
        changed = deepcopy(dynamic)
        changed["extra"] = "unreviewed instruction"
        alterations.append(changed)
        for changed in alterations:
            body = prefix + context.DYNAMIC_MARKER + json.dumps(changed, ensure_ascii=False, sort_keys=True, indent=2)
            with self.subTest(changed=changed), self.assertRaises(context.ContextError):
                context.extract_snapshot(body, manifest)


if __name__ == "__main__":
    unittest.main()
