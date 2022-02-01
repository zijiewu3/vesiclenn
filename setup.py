from setuptools import setup

setup(
        name="cga_vesiclenn",
        install_requires="crease_ga",
        entry_points={"crease_ga.plugins":["vesiclenn=cga_vesiclenn.scatterer_generator:scatterer_generator"]},
        py_modules=["cga_vesiclenn"],
                        )
