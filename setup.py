from setuptools import setup, find_packages

setup(
        name='diogenes',
        version='0.0.1',
        url='https://github.com/dssg/diogenes',
        author='Center for Data Science and Public Policy',
        description='A grid search library for machine learning',
        packages=find_packages(),
        install_requires=('numpy>=1.10.1',
                          'scipy',
                          'pandas',
                          'scikit-learn', 
                          'matplotlib', 
                          'SQLAlchemy',
                          'joblib',
                          'pdfkit'),
        zip_safe=False)
