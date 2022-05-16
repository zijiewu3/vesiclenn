from setuptools import setup

setup(
        name="cga_vesiclenn",
        install_requires="crease_ga",
        entry_points={"crease_ga.plugins":["vesiclenn=cga_vesiclenn.scatterer_generator:scatterer_generator",
                                           "vesiclenn_log_normal = cga_vesiclenn.scatterer_generator:scatterer_generator_log_normal",
                                           "vesiclenn_log_log = cga_vesiclenn.scatterer_generator:scatterer_generator_log_log",
                                           "vesiclenn_disp_nonvar = cga_vesiclenn_disp_nonvar.scatterer_generator:scatterer_generator",
                                           "vesiclenn_disp_fixta = cga_vesiclenn_disp_fixta.scatterer_generator:scatterer_generator",
                                           ]},
        py_modules=["cga_vesiclenn"],
                        )
