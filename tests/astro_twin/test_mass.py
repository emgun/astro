from astro_twin.mass import build_mass_budget_summary
from astro_twin.models import MassBudgetItemConfig, SpacecraftBusConfig


def test_build_mass_budget_summary_rolls_up_item_contingency_and_margin() -> None:
    summary = build_mass_budget_summary(
        SpacecraftBusConfig(
            name="ObserverSat",
            dry_mass_kg=120.0,
            payload_mass_kg=25.0,
            propellant_mass_kg=45.0,
            mass_margin_fraction_required=0.2,
            mass_budget_items=(
                MassBudgetItemConfig(
                    name="bus-structure",
                    category="bus",
                    mass_kg=90.0,
                    contingency_fraction=0.1,
                ),
                MassBudgetItemConfig(
                    name="avionics",
                    category="bus",
                    mass_kg=20.0,
                    contingency_fraction=0.2,
                ),
                MassBudgetItemConfig(
                    name="payload",
                    category="payload",
                    mass_kg=25.0,
                ),
            ),
        )
    )

    assert summary.itemized_base_mass_kg == 135.0
    assert summary.itemized_contingency_mass_kg == 13.0
    assert summary.itemized_total_mass_kg == 148.0
    assert summary.dry_payload_reference_mass_kg == 145.0
    assert summary.dry_payload_margin_kg == -3.0
    assert summary.category_totals_kg == {"bus": 123.0, "payload": 25.0}
