import math
import unittest

from question_bank import QUESTION_DEFINITIONS, question_bank, validate_question, validate_question_definitions


STUDENTS = [
    "100000001",
    "123456789",
    "222222222",
    "333333333",
    "444444444",
]

SESSIONS = [
    "20260307-000001",
    "20260308-000001",
    "20260401-101010",
    "20260515-202020",
    "20261231-235959",
]

PERSONALIZED_BUILDERS = {
    "software_bug_variance",
    "factory_total_probability",
    "factory_bayes_machine3",
    "fabric_expected_value",
}

PROBABILITY_BOUNDED_BUILDERS = {
    "backup_power_union",
    "hotel_chain_total_probability",
    "medical_test_bayes",
    "tv_sets_p_ge_1",
    "factory_total_probability",
    "factory_bayes_machine3",
    "airbag_pmf_exact_two",
    "vacuum_pdf_prob",
    "circuit_reliability",
}

NON_PROBABILITY_BUILDERS = {
    "software_bug_variance",
    "fabric_expected_value",
}


class QuestionBankInvariantsTest(unittest.TestCase):
    def _generate_question(self, question_id: str, student_id: str, quiz_session: str) -> dict:
        return question_bank(
            student_id,
            quiz_session,
            slot_ids=[question_id, question_id, question_id, question_id, question_id],
        )[0]

    def test_question_invariants_hold_for_sampled_personalizations(self) -> None:
        for item in QUESTION_DEFINITIONS:
            question_id = item["id"]
            for student_id in STUDENTS:
                for quiz_session in SESSIONS:
                    with self.subTest(question_id=question_id, student_id=student_id, quiz_session=quiz_session):
                        question = self._generate_question(question_id, student_id, quiz_session)
                        self.assertTrue(math.isfinite(float(question["answer"])), msg=f"{question_id}: answer must be finite")
                        validate_question(question_id, question)

    def test_question_registry_requires_validators(self) -> None:
        validate_question_definitions()
        for item in QUESTION_DEFINITIONS:
            self.assertTrue(callable(item.get("validator")), msg=f"{item['id']}: validator must be callable")

    def test_same_student_same_session_is_stable(self) -> None:
        for item in QUESTION_DEFINITIONS:
            question_id = item["id"]
            question_a = self._generate_question(question_id, STUDENTS[0], SESSIONS[0])
            question_b = self._generate_question(question_id, STUDENTS[0], SESSIONS[0])
            self.assertEqual(question_a["text"], question_b["text"], msg=f"{question_id}: text must be stable")
            self.assertAlmostEqual(
                float(question_a["answer"]),
                float(question_b["answer"]),
                msg=f"{question_id}: answer must be stable",
            )

    def test_modified_builders_are_not_static(self) -> None:
        for question_id in PERSONALIZED_BUILDERS:
            signatures = set()
            for student_id in STUDENTS:
                for quiz_session in SESSIONS:
                    question = self._generate_question(question_id, student_id, quiz_session)
                    signatures.add((question["text"], round(float(question["answer"]), 12)))
            self.assertGreater(
                len(signatures),
                1,
                msg=f"{question_id}: builder should produce more than one personalized variant",
            )

    def test_probability_questions_declare_unit_interval_bounds(self) -> None:
        for question_id in PROBABILITY_BOUNDED_BUILDERS:
            for student_id in STUDENTS:
                for quiz_session in SESSIONS:
                    with self.subTest(question_id=question_id, student_id=student_id, quiz_session=quiz_session):
                        question = self._generate_question(question_id, student_id, quiz_session)
                        self.assertEqual(question.get("answer_min"), 0.0, msg=f"{question_id}: answer_min should be 0.0")
                        self.assertEqual(question.get("answer_max"), 1.0, msg=f"{question_id}: answer_max should be 1.0")
                        answer = float(question["answer"])
                        self.assertGreaterEqual(answer, 0.0, msg=f"{question_id}: answer should stay >= 0")
                        self.assertLessEqual(answer, 1.0, msg=f"{question_id}: answer should stay <= 1")

    def test_non_probability_questions_can_exceed_one(self) -> None:
        for question_id in NON_PROBABILITY_BUILDERS:
            answers = []
            for student_id in STUDENTS:
                for quiz_session in SESSIONS:
                    question = self._generate_question(question_id, student_id, quiz_session)
                    self.assertIsNone(question.get("answer_min"), msg=f"{question_id}: answer_min should be unset")
                    self.assertIsNone(question.get("answer_max"), msg=f"{question_id}: answer_max should be unset")
                    answers.append(float(question["answer"]))
            self.assertTrue(
                any(answer > 1.0 for answer in answers),
                msg=f"{question_id}: sampled variants should include an answer above 1.0",
            )


if __name__ == "__main__":
    unittest.main()
