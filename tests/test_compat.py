from conftest import make_asset

from mksynth.compat import CompatConfig, build_table, can_stack, cavity_radius_at, nest_fit


def test_cup_nests_in_bowl_and_rests_on_floor(kitchen):
    fit = nest_fit(kitchen["cup"], kitchen["big_bowl"])
    assert fit is not None
    # The cup (r = 3.5 cm) clears the 6 cm floor radius, so it reaches the floor.
    assert fit["rest_offset"] == 0.0
    assert fit["inserted_fraction"] == 1.0


def test_tapered_cavity_holds_wide_object_higher_up():
    bowl = make_asset("bowl", height=0.08, radius=0.11, cavity_depth=0.07, cavity_radius=0.10, floor_radius=0.03)
    wide = make_asset("wide", height=0.04, radius=0.06)
    fit = nest_fit(wide, bowl)
    assert fit is not None
    # Radius 6 cm + 3 mm clearance is only free where the cavity radius reaches 6.3 cm.
    expected = (0.063 - 0.03) / (0.10 - 0.03) * 0.07
    assert abs(fit["rest_offset"] - expected) < 0.002
    assert cavity_radius_at(bowl, [fit["rest_offset"]])[0] >= 0.063 - 1e-9


def test_too_wide_does_not_nest(kitchen):
    assert nest_fit(kitchen["box"], kitchen["mug"]) is None
    assert nest_fit(kitchen["plate"], kitchen["big_bowl"]) is None


def test_lid_on_rim_is_not_nesting():
    # A saucer barely narrower than the opening of a shallow ramekin would sit with its centre of
    # mass at the rim: that is stacking, not nesting.
    ramekin = make_asset("ramekin", height=0.03, radius=0.05, cavity_depth=0.004, cavity_radius=0.045)
    saucer = make_asset("saucer", height=0.015, radius=0.04)
    assert nest_fit(saucer, ramekin) is None


def test_mostly_outside_is_not_nesting(kitchen):
    # A tall can in a shallow ramekin sticks out: inserted fraction below the minimum.
    cfg = CompatConfig(min_inserted_fraction=0.5)
    assert nest_fit(kitchen["can"], kitchen["ramekin"], cfg) is None


def test_volume_rule_for_nesting(kitchen):
    strict = CompatConfig(max_nest_volume_ratio=0.01)
    assert nest_fit(kitchen["cup"], kitchen["big_bowl"], strict) is None


def test_never_on_top(kitchen):
    for other in kitchen.values():
        assert not can_stack(kitchen["pitcher"], other)
        assert nest_fit(kitchen["pitcher"], other) is None


def test_stack_area_and_volume_rules(kitchen):
    assert can_stack(kitchen["small_box"], kitchen["box"])
    assert not can_stack(kitchen["box"], kitchen["small_box"])  # footprint larger than the top face
    assert not can_stack(kitchen["box"], kitchen["box"])         # never on itself
    assert not can_stack(kitchen["small_box"], kitchen["box"], CompatConfig(max_stack_volume_ratio=0.01))


def test_object_that_fits_inside_is_nested_not_stacked(kitchen):
    assert nest_fit(kitchen["cup"], kitchen["pot"]) is not None
    assert not can_stack(kitchen["cup"], kitchen["pot"])


def test_small_top_face_cannot_support(kitchen):
    assert not can_stack(kitchen["small_box"], kitchen["can"])  # can top face 4.4e-3 m^2 > min, but footprint too big
    tiny = make_asset("tiny", height=0.1, radius=0.02)
    assert not can_stack(make_asset("dot", height=0.01, radius=0.005), tiny)


def test_table_matches_pairwise_rules(kitchen):
    table = build_table(kitchen)
    for support, tops in table["stack"].items():
        for top in tops:
            assert can_stack(kitchen[top], kitchen[support])
    for container, inners in table["nest"].items():
        for inner, fit in inners.items():
            assert fit == nest_fit(kitchen[inner], kitchen[container])
    stack_pairs = sum(len(v) for v in table["stack"].values())
    brute = sum(can_stack(a, b) for a in kitchen.values() for b in kitchen.values())
    assert stack_pairs == brute


def test_stack_on_container_must_bridge_the_opening():
    from mksynth.compat import can_stack

    bowl = make_asset("bowl", height=0.07, radius=0.09, cavity_depth=0.06, cavity_radius=0.08)
    # Too wide to nest with clearance, too narrow to rest on the rim: it would wedge in.
    wedge = make_asset("wedge", height=0.05, radius=0.079)
    assert can_stack(wedge, bowl) is False
    lid = make_asset("lid", height=0.02, radius=0.09)  # wider than the opening: rests on the rim
    assert can_stack(lid, bowl) is True
