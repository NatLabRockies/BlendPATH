import logging

import BlendPATH.costing.costing as bp_cost
import BlendPATH.file_writing.file_writing as bp_file_w
import BlendPATH.file_writing.results_file_util as bp_res_file
import BlendPATH.modifications.mod_util as bp_mod_util
import BlendPATH.util.pipe_assessment as bp_pa
from BlendPATH.emissions.emissions import calc_emissions_intensity
from BlendPATH.emissions.emissions_params import EmissionsParams
from BlendPATH.network import BlendPATH_network
from BlendPATH.scenario_helper import (
    _SCENARIO_INPUTS,
    SCENARIO_VALUES,
    Design_params,
    Scenario_type,
    process_default_inputs,
)
from BlendPATH.util.disclaimer import disclaimer_message

logger = logging.getLogger(__name__)


class BlendPATH_scenario:
    """
    BlendPATH_scenario:
    ---------------
    Description: This is a BlendPATH scenario that does the modification and analysis on a BlendPATH network

    Required Parameters:
    ---------------
    casestudy_name: str = Path to directory of case study files
    """

    def __init__(self, casestudy_name: str, **kwargs) -> None:
        # Assign class variables
        self.casestudy_name = casestudy_name

        # Assign variables - check with user input and input file
        # First assign all defaults, and resolve with local overrides
        self.costing_params = bp_cost.Costing_params()
        self.design_params = Design_params()
        self.emissions_params = EmissionsParams()
        self.costing_params, self.design_params, network_params = (
            process_default_inputs(
                casestudy_name=self.casestudy_name,
                intialization_inputs=kwargs,
                scenario_instance=self,
            )
        )

        # Print disclaimer message
        logger.info(disclaimer_message())

        # Initialize BlendPATH_network
        self.network = BlendPATH_network.import_from_file(
            f"{self.casestudy_name}/network_design.xlsx",
            composition_tracking=network_params.composition_tracking,
            thermo_curvefit=network_params.thermo_curvefit,
            eos=network_params.eos,
            scenario_type=self.scenario_type,
            blend=network_params.blend,
        )

        # Set design option
        self.update_design_option(self.design_option, init=True)

        # Segmentation
        if self.scenario_type == Scenario_type.TRANSMISSION:
            self.network.pipe_segments = self.network.segment_pipe()

        # Blend in hydrogen
        self.blendH2(network_params.blend, self.costing_params.h2_price, True)

    def run_mod(
        self,
        mod_type: str,
        design_option: str = None,
        option: str = None,
        allow_compressor_bypass: bool = True,
    ) -> float:
        """
        Run a modification method on the network using the scenario parameters

        Parameters:
        ------------
        mod_type:str = direct_replacement, parallel_loop, additional_compressors, DR, PL, or AC
        design_option:str = None - Design option for new pipe
        """
        if design_option is None:
            design_option = SCENARIO_VALUES[self.scenario_type].default_design_option

        self.mod_type = bp_mod_util.check_mod_method(mod_type)
        # Update design option for new pipe
        self.design_option_new = bp_pa.check_design_option(design_option)
        logger.info(
            f"Running modification method: {self.mod_type} with design option: {self.design_option_new}"
        )

        # Get target network design file name
        new_filename_full = bp_mod_util.mod_network_file_name(
            casestudy_name=self.casestudy_name,
            results_dir=self.results_dir,
            mod_type=self.mod_type,
            blend=self.blend,
            design_option=self.design_option,
            filename_suffix=self.filename_suffix,
        )

        # Send to modification method handler
        mod_result = bp_mod_util.mod_handler(
            mod_type=self.mod_type,
            network=self.network,
            design_option=design_option,
            new_filename_full=new_filename_full,
            design_params=self.design_params,
            costing_params=self.costing_params,
            option=option,
            allow_compressor_bypass=allow_compressor_bypass,
        )

        # Import modified network and run simulation
        new_network = bp_mod_util.generate_modified_network(
            new_filename_full=new_filename_full,
            composition_tracking=self.composition_tracking,
            thermo_curvefit=self.thermo_curvefit,
            eos=self.eos,
            ff_type=self.network.ff_type,
            scenario_type=self.scenario_type,
        )
        self.new_network = new_network

        # Get the parameters that go into LCOT calculation
        mod_costing_params = bp_mod_util.get_mod_costing_params(
            network=new_network,
            design_params=self.design_params,
            costing_params=self.costing_params,
        )

        price_breakdown, pf_inputs = bp_cost.calc_lcot(
            mod_costing_params=mod_costing_params,
            new_pipe_capex=mod_result.cap_cost["total cost"],
            costing_params=self.costing_params,
        )

        emissions = calc_emissions_intensity(
            network=new_network,
            emissions_params=self.emissions_params,
        )

        # Print LCOT
        logger.info(
            f"LCOT: {price_breakdown['LCOT: Levelized cost of transport']} $/MMBTU"
        )

        # Make result files
        self.result_files(
            new_network=new_network,
            price_breakdown=price_breakdown,
            pf_inputs=pf_inputs,
            mod_result=mod_result,
            mod_costing_params=mod_costing_params,
            emissions=emissions,
        )

        return price_breakdown["LCOT: Levelized cost of transport"]

    def result_files(
        self,
        new_network: BlendPATH_network,
        price_breakdown: dict[str, float],
        pf_inputs: dict,
        mod_result: bp_mod_util.Mod_result,
        mod_costing_params: bp_mod_util.Mod_costing_params,
        emissions: dict | None = None,
    ) -> None:
        """
        Create result file with summary values
        """
        results_file_name = f"{self.casestudy_name}/{self.results_dir}/ResultFiles/{self.mod_type.upper()}_{self.blend}_{self.design_option}{self.filename_suffix}.xlsx"

        workbook = bp_file_w.file_setup(filename=results_file_name)

        bp_res_file.write_disclaimer_sheet(workbook=workbook)
        bp_res_file.write_inputs_sheet(workbook=workbook, scenario=self)
        bp_res_file.write_results_sheet(
            workbook=workbook,
            network=self.network,
            new_network=new_network,
            mod_result=mod_result,
            mod_costing_params=mod_costing_params,
            price_breakdown=price_breakdown,
            costing_params=self.costing_params,
            emissions=emissions,
        )
        bp_res_file.write_mod_network_sheet(
            workbook=workbook,
            network=self.network,
            new_network=new_network,
            mod_result=mod_result,
            mod_type=self.mod_type,
        )
        bp_res_file.write_comp_sheet(
            workbook=workbook,
            network=self.network,
            new_network=new_network,
            mod_result=mod_result,
            mod_type=self.mod_type,
        )
        bp_res_file.write_pressure_sheet(workbook=workbook, new_network=new_network)
        bp_res_file.write_demand_sheet(workbook=workbook, new_network=new_network)
        bp_res_file.write_profast_sheet(workbook=workbook, pf_inputs=pf_inputs)

        ### Closeout
        bp_file_w.file_closeout(workbook=workbook)

    def update_design_option(self, design_option: str, init: bool = False) -> None:
        """
        Update the design option of original pipeline

        Parameters:
        -----------
        design_option: str - Design option, must be 'a','b', or 'nfc'

        """
        self._design_option = bp_pa.check_design_option(design_option)

        # Reassess pipe with ASME B31.12
        self.network.pipe_assessment(
            self.design_params.asme,
            design_option=self.design_option,
        )

        # Reassign segment ASME pressure, only if called after scenario is initialized
        if not init:
            logger.info(
                f"Updating existing pipe design option to: {self._design_option}"
            )
            for ps in self.network.pipe_segments:
                ps.design_pressure_MPa = ps.pipes[0].design_pressure_MPa

    def blendH2(self, blend: float, h2_price: float = None, init: bool = False) -> None:
        """
        Blend amount of H2

        Parameters:
        -----------
        blend:float - Fraction of hydrogen
        h2_price: float = None - H2 price $/kg
        """
        # Check values
        try:
            blend = float(blend)
        except ValueError:
            raise ValueError(f"Blend must be numeric. The value entered was {blend}")
        if not (0 <= blend <= 1):
            raise ValueError(
                f"Blend must be between 0 and 1. The value entered was {blend}"
            )
        if not init:
            logger.info(f"Updating H2 blend to: {blend * 100:0.2f}%")
        self._blend = blend
        self.network.blendH2(self.blend)

        # Check if new H2 price was specified and calculate new compressor fuel cost
        if h2_price is not None:
            self.costing_params.h2_price = h2_price
        self.costing_params.cf_price = bp_cost.get_cs_fuel_cost(
            blend,
            self.costing_params.ng_price,
            self.costing_params.h2_price,
            self.network.composition,
            self.casestudy_name,
            self.costing_params.financial_overrides,
        )

    @property
    def location_class(self):
        return self.design_params.asme.location_class

    @location_class.setter
    def location_class(self, value):
        for formatting_fxn in _SCENARIO_INPUTS["location_class"].formatting:
            value = formatting_fxn(value)
        self.design_params.asme.location_class = int(value)

    @property
    def T_rating(self):
        return self.design_params.asme.T_rating

    @T_rating.setter
    def T_rating(self, value):
        for formatting_fxn in _SCENARIO_INPUTS["T_rating"].formatting:
            value = formatting_fxn(value)
        self.design_params.asme.T_rating = value

    @property
    def joint_factor(self):
        return self.design_params.asme.joint_factor

    @joint_factor.setter
    def joint_factor(self, value):
        for formatting_fxn in _SCENARIO_INPUTS["joint_factor"].formatting:
            value = formatting_fxn(value)
        self.design_params.asme.joint_factor = value

    @property
    def final_outlet_pressure_mpa_g(self):
        return self.design_params.final_outlet_pressure_mpa_g

    @final_outlet_pressure_mpa_g.setter
    def final_outlet_pressure_mpa_g(self, value):
        for formatting_fxn in _SCENARIO_INPUTS[
            "final_outlet_pressure_mpa_g"
        ].formatting:
            value = formatting_fxn(value)
        self.design_params.final_outlet_pressure_mpa_g = value

    @property
    def design_CR(self):
        return self.design_params.max_CR

    @design_CR.setter
    def design_CR(self, value):
        for formatting_fxn in _SCENARIO_INPUTS["design_CR"].formatting:
            value = formatting_fxn(value)
        self.design_params.max_CR = value

    @property
    def existing_compressors_to_electric(self):
        return self.design_params.existing_comp_elec

    @existing_compressors_to_electric.setter
    def existing_compressors_to_electric(self, value):
        for formatting_fxn in _SCENARIO_INPUTS[
            "existing_compressors_to_electric"
        ].formatting:
            value = formatting_fxn(value)
        self.design_params.existing_comp_elec = value

    @property
    def new_compressors_electric(self):
        return self.design_params.new_comp_elec

    @new_compressors_electric.setter
    def new_compressors_electric(self, value):
        for formatting_fxn in _SCENARIO_INPUTS["new_compressors_electric"].formatting:
            value = formatting_fxn(value)
        self.design_params.new_comp_elec = value

    @property
    def new_comp_eta_s(self):
        return self.design_params.new_comp_eta_s

    @new_comp_eta_s.setter
    def new_comp_eta_s(self, value):
        for formatting_fxn in _SCENARIO_INPUTS["new_comp_eta_s"].formatting:
            value = formatting_fxn(value)
        self.design_params.new_comp_eta_s = value

    @property
    def new_comp_eta_driver(self):
        return self.design_params.new_comp_eta_driver

    @new_comp_eta_driver.setter
    def new_comp_eta_driver(self, value):
        for formatting_fxn in _SCENARIO_INPUTS["new_comp_eta_driver"].formatting:
            value = formatting_fxn(value)
        self.design_params.new_comp_eta_driver = value

    @property
    def new_comp_eta_s_elec(self):
        return self.design_params.new_comp_eta_s_elec

    @new_comp_eta_s_elec.setter
    def new_comp_eta_s_elec(self, value):
        for formatting_fxn in _SCENARIO_INPUTS["new_comp_eta_s_elec"].formatting:
            value = formatting_fxn(value)
        self.design_params.new_comp_eta_s_elec = value

    @property
    def new_comp_eta_driver_elec(self):
        return self.design_params.new_comp_eta_driver_elec

    @new_comp_eta_driver_elec.setter
    def new_comp_eta_driver_elec(self, value):
        for formatting_fxn in _SCENARIO_INPUTS["new_comp_eta_driver_elec"].formatting:
            value = formatting_fxn(value)
        self.design_params.new_comp_eta_driver_elec = value

    @property
    def h2_price(self):
        return self.costing_params.h2_price

    @h2_price.setter
    def h2_price(self, value):
        self.costing_params.h2_price = value

    @property
    def ng_price(self):
        return self.costing_params.ng_price

    @ng_price.setter
    def ng_price(self, value):
        self.costing_params.ng_price = value

    @property
    def elec_price(self):
        return self.costing_params.elec_price

    @elec_price.setter
    def elec_price(self, value):
        self.costing_params.elec_price = value

    @property
    def region(self):
        return self.costing_params.region

    @region.setter
    def region(self, value):
        for formatting_fxn in _SCENARIO_INPUTS["region"].formatting:
            value = formatting_fxn(value)
        self.costing_params.region = value

    @property
    def ili_interval(self):
        return self.costing_params.ili_interval

    @ili_interval.setter
    def ili_interval(self, value):
        for formatting_fxn in _SCENARIO_INPUTS["ili_interval"].formatting:
            value = formatting_fxn(value)
        self.costing_params.ili_interval = value

    @property
    def original_pipeline_cost(self):
        return self.costing_params.original_pipeline_cost

    @original_pipeline_cost.setter
    def original_pipeline_cost(self, value):
        for formatting_fxn in _SCENARIO_INPUTS["original_pipeline_cost"].formatting:
            value = formatting_fxn(value)
        self.costing_params.original_pipeline_cost = value

    @property
    def pipe_markup(self):
        return self.costing_params.pipe_markup

    @pipe_markup.setter
    def pipe_markup(self, value):
        for formatting_fxn in _SCENARIO_INPUTS["pipe_markup"].formatting:
            value = formatting_fxn(value)
        self.costing_params.pipe_markup = value

    @property
    def compressor_markup(self):
        return self.costing_params.compressor_markup

    @compressor_markup.setter
    def compressor_markup(self, value):
        for formatting_fxn in _SCENARIO_INPUTS["compressor_markup"].formatting:
            value = formatting_fxn(value)
        self.costing_params.compressor_markup = value

    @property
    def financial_overrides(self):
        return self.costing_params.financial_overrides

    @financial_overrides.setter
    def financial_overrides(self, value):
        self.costing_params.financial_overrides = value

    @property
    def steel_cost(self):
        return self.costing_params.steel_cost

    @steel_cost.setter
    def steel_cost(self, value):
        self.costing_params.steel_cost = value

    @property
    def scenario_type(self):
        return self._scenario_type

    @scenario_type.setter
    def scenario_type(self, value):
        for formatting_fxn in _SCENARIO_INPUTS["scenario_type"].formatting:
            value = formatting_fxn(value)
        self._scenario_type = value

    @property
    def eos(self):
        return self.network.eos

    @eos.setter
    def eos(self, value):
        self.network.eos = value

    @property
    def thermo_curvefit(self):
        return self.network.thermo_curvefit

    @thermo_curvefit.setter
    def thermo_curvefit(self, value):
        for formatting_fxn in _SCENARIO_INPUTS["thermo_curvefit"].formatting:
            value = formatting_fxn(value)
        self.network.set_thermo_curvefit(value)

    @property
    def composition_tracking(self):
        return self.network.composition_tracking

    @composition_tracking.setter
    def composition_tracking(self, value):
        for formatting_fxn in _SCENARIO_INPUTS["composition_tracking"].formatting:
            value = formatting_fxn(value)
        self.network.set_composition_tracking(value)

    @property
    def blend(self):
        return self._blend

    @blend.setter
    def blend(self, value):
        for formatting_fxn in _SCENARIO_INPUTS["blend"].formatting:
            value = formatting_fxn(value)

        self.blendH2(blend=value)

    @property
    def design_option(self):
        return self._design_option

    @design_option.setter
    def design_option(self, value):
        for formatting_fxn in _SCENARIO_INPUTS["design_option"].formatting:
            value = formatting_fxn(value)

        self._design_option = value
        self.update_design_option(value)
