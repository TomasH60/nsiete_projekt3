from router.metrics import evaluate_router, gqr_score


def test_gqr_score_is_harmonic_mean() -> None:
    assert gqr_score(0.8, 0.5) == 2 * 0.8 * 0.5 / (0.8 + 0.5)


def test_gqr_score_handles_zero_denominator() -> None:
    assert gqr_score(0.0, 0.0) == 0.0


def test_evaluate_router_splits_id_and_ood() -> None:
    scores = evaluate_router(
        y_true=[0, 1, 2, 3, 3],
        y_pred=[0, 3, 2, 3, 0],
    )

    assert scores.id_accuracy == 2 / 3
    assert scores.ood_accuracy == 1 / 2
    assert scores.gqr_score == gqr_score(2 / 3, 1 / 2)
