from setuptools import setup

setup(
        name="cga_vesiclenn",
        install_requires="crease_ga",
        entry_points={"crease_ga.plugins":["vesiclenn=cga_vesiclenn.scatterer_generator:scatterer_generator",
                                           "vesiclenn_2parts = cga_vesiclenn.scatter_generator:scatterer_generator_2parts"]},
        py_modules=["cga_vesiclenn"],
                        )
