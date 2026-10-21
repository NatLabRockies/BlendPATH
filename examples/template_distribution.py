# Import BlendPATH functions into analysis script for use
import BlendPATH

# Define a case study scenario that models a distribution network of interest with
# data from network_design.xlsx, default_inputs.csv, and override/ files
distribution = BlendPATH.BlendPATH_scenario(
    casestudy_name="examples/distribution_example",
    scenario_type="distribution",
    # Define the design factor used to set the design pressure of existing
    # pipeline segments that will transport hydrogen as a blend or in pure form
)

# Set the desired hydrogen content (in vol. %) in pipeline gas for analysis
distribution.blendH2(blend=1.0)

# Set the maximum allowed gas velocity (m/s) in the distribution network
distribution.max_velocity_distribution = 20

# Run pipeline modification analysis with run_mod using the defined mod_type
# modification strategy. New pipeline infrastructure that is added to the
# existing pipeline network is rated to the defined design factor with run_mod
distribution.run_mod(mod_type="direct_replacement")

# Result xlsx files are outputted within a user-designated result_dir (defined in
# default_inputs.csv) under the casestudy_name folder.
