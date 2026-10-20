# Import BlendPATH functions into analysis script for use
import BlendPATH

# Define a case study scenario that models a network of interest with data from
# network_design.xlsx, default_inputs.csv, and override/ files
wangetal = BlendPATH.BlendPATH_scenario(
    casestudy_name="examples/wangetal2018",
)
# Define the relevant ASME B31.12 design option to set the maximum allowable
# operating pressure for existing pipeline segments that will transport hydrogen
# as a blend or in pure form
wangetal.update_design_option(design_option="nfc")

# Set the desired hydrogen content (in vol. %) in pipeline gas for analysis
wangetal.blendH2(blend=0.2)

# Run pipeline modification analysis with run_mod using the defined mod_type
# modification strategy. New pipeline infrastructure that is added to the
# existing pipeline network is rated to the defined design option with run_mod
wangetal.run_mod(mod_type="direct_replacement", design_option="b")

# Result xlsx files are outputted within a user-designated result_dir (defined in
# default_inputs.csv) under the casestudy_name folder.
