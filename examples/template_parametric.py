import BlendPATH

wangetal = BlendPATH.BlendPATH_scenario(
    casestudy_name="examples/wangetal2018",
)
design_options = ["a", "b", "nfc"]
blends = [0.2, 0.5, 1.0]
mod_types = ["dr", "ac", "pl"]

for do in design_options:
    wangetal.update_design_option(do)
    for h2b in blends:
        wangetal.blendH2(h2b)
        for mt in mod_types:
            wangetal.run_mod(mt)
