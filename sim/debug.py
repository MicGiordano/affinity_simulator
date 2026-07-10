from sim.affinity_catalog import get_card_spec, token_spec
from sim.models import Permanent
from sim.planner import PlanningState, apply_action, Action


def print_state(label, state):
    print(f"\n=== {label} ===")
    print("HAND:", [c.name for c in state.hand])
    print("BATTLEFIELD:", [p.spec.name for p in state.battlefield])
    print("GRAVEYARD:", [c.name for c in state.graveyard])


def make_state(hand_names, battlefield_names):
    hand = tuple(get_card_spec(n) for n in hand_names)
    battlefield = tuple(Permanent(get_card_spec(n), tapped=False, entered_this_turn=False)
                        for n in battlefield_names)
    return PlanningState(hand, battlefield, tuple(), ())


# =========================================================
# TEST 1: Blood Fountain → Blood Token generation
# =========================================================
def test_blood_fountain():
    state = make_state(
        hand_names=["Blood Fountain"],
        battlefield_names=["Swamp"]
    )

    action = Action("cast_spell", "Blood Fountain")

    new_state = apply_action(state, action)

    if new_state is None:
        print("🔥 ERROR: apply_action returned None")
        return

    print_state("Blood Fountain test", new_state)


# =========================================================
# TEST 2: Reckoner's Bargain sacrifice correctness
# =========================================================
def test_reckoner_bargain():
    state = make_state(
        hand_names=["Reckoner's Bargain"],
        battlefield_names=["Ichor Wellspring"]
    )

    action = Action("cast_spell", "Reckoner's Bargain", sacrifice_name="Ichor Wellspring")

    new_state = apply_action(state, action)

    print_state("Reckoner Bargain test", new_state)


# =========================================================
# TEST 3: Fanatical Offering → Map Token + sacrifice
# =========================================================
def test_fanatical_offering():
    state = make_state(
        hand_names=["Fanatical Offering"],
        battlefield_names=["Blood Token"]
    )

    action = Action("cast_spell", "Fanatical Offering", sacrifice_name="Blood Token")

    new_state = apply_action(state, action)

    print_state("Fanatical Offering test", new_state)


# =========================================================
# TEST 4: Land should NOT be sacrificed if alternatives exist
# =========================================================
def test_land_sacrifice():
    state = make_state(
        hand_names=["Reckoner's Bargain"],
        battlefield_names=["Swamp", "Blood Token"]
    )

    action = Action("cast_spell", "Reckoner's Bargain", sacrifice_name="Blood Token")

    new_state = apply_action(state, action)

    print_state("Land sacrifice avoidance test", new_state)


# =========================================================
# TEST 5: Toxin Analysis (requires creature + clue token)
# =========================================================
def test_toxin_analysis():
    state = make_state(
        hand_names=["Toxin Analysis"],
        battlefield_names=["Krark-Clan Shaman"]
    )

    action = Action("cast_spell", "Toxin Analysis")

    new_state = apply_action(state, action)

    print_state("Toxin Analysis test", new_state)


# =========================================================
# TEST 6: Permanent vs spell behavior
# =========================================================
def test_permanents_vs_spells():
    state = make_state(
        hand_names=["Thoughtcast", "Ichor Wellspring"],
        battlefield_names=["Swamp", "Swamp"]
    )

    # Cast permanent
    state1 = apply_action(state, Action("cast_spell", "Ichor Wellspring"))

    # Cast spell
    state2 = apply_action(state1, Action("cast_spell", "Thoughtcast"))

    print_state("Permanent vs spell test", state2)


# =========================================================
# RUN ALL TESTS
# =========================================================
if __name__ == "__main__":
    print("\n===== DEBUG RUN =====")
    test_blood_fountain()
    test_reckoner_bargain()
    test_fanatical_offering()
    test_land_sacrifice()
    test_toxin_analysis()
    test_permanents_vs_spells()