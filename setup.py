from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
# from builtins import *

from codecs import open
from os.path import realpath, dirname, join
from setuptools import setup, find_packages
import sys
import re

DISTNAME = 'pymc-learn'
DESCRIPTION = "Practical Probabilistic Machine Learning in Python"
AUTHOR = 'Pymc-Learn Team'
AUTHOR_EMAIL = 'daniel.emaasit@gmail.com'
URL = "https://github.com/pymc-learn/pymc-learn"


classifiers = [
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Operating System :: OS Independent'
]

PROJECT_ROOT = dirname(realpath(__file__))

with open(join(PROJECT_ROOT, 'README.rst'), encoding='utf-8') as r:
    readme = r.read()

REQUIREMENTS_FILE = join(PROJECT_ROOT, 'requirements.txt')

with open(REQUIREMENTS_FILE) as f:
    install_reqs = f.read().splitlines()

if sys.version_info < (3, 4):
    install_reqs.append('enum34')


def get_version():
    VERSIONFILE = join('pmlearn', '__init__.py')
    lines = open(VERSIONFILE, 'rt').readlines()
    version_regex = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in lines:
        mo = re.search(version_regex, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError('Unable to find version in %s.' % (VERSIONFILE,))


with open('AUTHORS.txt') as a:
    # reSt-ify the authors list
    authors = ''
    for author in a.read().split('\n'):
        authors += '| '+author+'\n'

with open('LICENSE') as l:
    license = l.read()


if __name__ == "__main__":
    setup(
        name=DISTNAME,
        version=get_version(),
        description=DESCRIPTION,
        long_description=readme,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        license=license,
        packages=find_packages(),
        package_data={'docs': ['*']},
        include_package_data=True,
        zip_safe=False,
        install_requires=install_reqs,
        classifiers=classifiers
        )