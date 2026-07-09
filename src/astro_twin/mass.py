from __future__ import annotations

from astro_twin.models import MassBudgetSummary, SpacecraftBusConfig


def build_mass_budget_summary(spacecraft: SpacecraftBusConfig) -> MassBudgetSummary:
    category_totals_kg: dict[str, float] = {}
    itemized_base_mass_kg = 0.0
    itemized_contingency_mass_kg = 0.0
    for item in spacecraft.mass_budget_items:
        contingency_mass_kg = item.mass_kg * item.contingency_fraction
        item_total_kg = item.mass_kg + contingency_mass_kg
        itemized_base_mass_kg += item.mass_kg
        itemized_contingency_mass_kg += contingency_mass_kg
        category_totals_kg[item.category] = (
            category_totals_kg.get(item.category, 0.0) + item_total_kg
        )

    dry_payload_reference_mass_kg = spacecraft.dry_mass_kg + spacecraft.payload_mass_kg
    itemized_total_mass_kg = itemized_base_mass_kg + itemized_contingency_mass_kg
    configured_wet_mass_kg = dry_payload_reference_mass_kg + spacecraft.propellant_mass_kg
    return MassBudgetSummary(
        dry_mass_kg=spacecraft.dry_mass_kg,
        payload_mass_kg=spacecraft.payload_mass_kg,
        propellant_mass_kg=spacecraft.propellant_mass_kg,
        configured_wet_mass_kg=configured_wet_mass_kg,
        itemized_base_mass_kg=itemized_base_mass_kg,
        itemized_contingency_mass_kg=itemized_contingency_mass_kg,
        itemized_total_mass_kg=itemized_total_mass_kg,
        dry_payload_reference_mass_kg=dry_payload_reference_mass_kg,
        dry_payload_margin_kg=dry_payload_reference_mass_kg - itemized_total_mass_kg,
        category_totals_kg=dict(sorted(category_totals_kg.items())),
    )
