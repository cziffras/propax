SOLID = "the rho-T grid reaches the solid beyond the melting line, not bounded yet (need to implement it in both make_fluid and HEOS module by copying coolprop's melting lines)"
DENSITY_MAXIMUM = "rho_L has a density maximum, the superancillary does not split in monotone segments yet (it is only necessary for `heavywater`)"

KNOWN_FAILURES = {
    "test_density_inverts_onto_its_branch[heavywater]": DENSITY_MAXIMUM,
    "test_the_flash_zeroes_its_residual[orthohydrogen-ST]": SOLID,
    **{
        f"test_the_fluid_is_thermodynamically_stable[{fluid}]": SOLID
        for fluid in ("helium", "hydrogen", "orthohydrogen", "parahydrogen", "nitrogen")
    },
    **{
        f"test_the_flash_zeroes_its_residual[{fluid}-{pair}]": SOLID
        for fluid in ("helium", "hydrogen", "orthohydrogen", "parahydrogen")
        for pair in ("DS", "PS")
    },
}
