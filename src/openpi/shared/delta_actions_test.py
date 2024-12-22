from openpi.shared import delta_actions


def test_make_bool_mask():
    assert delta_actions.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert delta_actions.make_bool_mask(2, 0, 2) == (True, True, True, True)
