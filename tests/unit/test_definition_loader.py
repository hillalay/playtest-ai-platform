from playtest.definition.loader import load_definition


def test_load_yaml_definition():

    definition = load_definition(
        "examples/arrow_grid/playtest.yaml"
    )

    assert (
        definition["game"]["id"]
        == "example.arrow_grid"
    )

    assert (
        definition["schema_version"]
        == "1.0"
    )