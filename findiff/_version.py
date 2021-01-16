"""Determine package version from git repository tag

Each time this file is imported it checks whether the package version
can be determined using `git describe`. If this fails (because either
this file is not located at the 1st level down the repository root or
it is not under version control), the version is read from the script
"_version_save.py" which is not versioned by git, but always included
in the final distribution archive (e.g. via PyPI). If the git version
does not match the saved version, then "_version_save.py" is updated.


Usage
-----
1. Put this file in your main module directory:

    REPO_ROOT/package_name/_version.py

2. Add this line to REPO_ROOT/package_name/__init__.py

    from ._version import version as __version__  # noqa: F401

3. (Optional) Add this line to REPO_ROOT/.gitignore

    _version_save.py

Features
--------
- supports Python 2 and 3
- supports frozen applications (e.g. PyInstaller)
- supports installing into a virtual environment that is located in
  a git repository
- saved version is located in a python file and therefore no other
  files (e.g. MANIFEST.in) need be edited
- fallback version is the creation date
- excluded from code coverage via "pragma: no cover"

Changelog
---------
2019-11-06.2
 - use os.path.split instead of counting os.path.sep (Windows)
2019-11-06
 - remove deprecated imp dependency (replace with parser)
 - check whether this file is versioned and its location is correct
 - code cleanup and docs update
"""

# Put the entire script into a `True` statement and add the hint
# `pragma: no cover` to ignore code coverage here.
if True:  # pragma: no cover
    import os
    from os.path import abspath, basename, dirname, join, split
    import subprocess
    import sys
    import time
    import traceback
    import warnings

    def git_describe():
        """
        Return a string describing the version returned by the
        command `git describe --tags HEAD`.
        If it is not possible to determine the correct version,
        then an empty string is returned.
        """
        # Make sure we are in a directory that belongs to the correct
        # repository.
        ourdir = dirname(abspath(__file__))

        def _minimal_ext_cmd(cmd):
            # Construct minimal environment
            env = {}
            for k in ['SYSTEMROOT', 'PATH']:
                v = os.environ.get(k)
                if v is not None:
                    env[k] = v
            # LANGUAGE is used on win32
            env['LANGUAGE'] = 'C'
            env['LANG'] = 'C'
            env['LC_ALL'] = 'C'
            pop = subprocess.Popen(cmd,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   env=env)
            out = pop.communicate()[0]
            return out.strip().decode('ascii', errors="ignore")

        # change directory
        olddir = abspath(os.curdir)
        os.chdir(ourdir)

        # Make sure that we are getting "git describe" from our own
        # repository (and not from a repository where we just happen
        # to be in the directory tree).
        git_revision = ""
        try:
            # If this file is not under version control, "loc" will
            # be empty.
            loc = _minimal_ext_cmd(['git', 'ls-files', '--full-name',
                                    __file__])
            # If it is under version control, it should be located
            # one hierarchy down from the repository root (either
            # __file__ is "docs/conf.py" or "package_name/_version.py".
            if len(split(loc)) == 2:
                try:
                    git_revision = _minimal_ext_cmd(['git', 'describe',
                                                     '--tags', 'HEAD'])
                except OSError:
                    pass
        except OSError:
            pass
        # Go back to original directory
        os.chdir(olddir)

        return git_revision

    def load_version(versionfile):
        """load version from version_save.py"""
        longversion = ""
        try:
            with open(versionfile, "r") as fd:
                data = fd.readlines()
                for line in data:
                    if line.startswith("longversion"):
                        longversion = line.split("=")[1].strip().strip("'")
        except BaseException:
            try:
                from ._version_save import longversion
            except BaseException:
                try:
                    from _version_save import longversion
                except BaseException:
                    pass

        return longversion

    def write_version(version, versionfile):
        """save version to version_save.py"""
        data = "#!/usr/bin/env python\n" \
            + "# This file was created automatically\n" \
            + "longversion = '{VERSION}'\n"
        try:
            with open(versionfile, "w") as fd:
                fd.write(data.format(VERSION=version))
        except BaseException:
            if not os.path.exists(versionfile):
                # Only issue a warning if the file does not exist.
                msg = "Could not write package version to {}.".format(
                    versionfile)
                warnings.warn(msg)

    hdir = dirname(abspath(__file__))
    if basename(__file__) == "conf.py" and "name" in locals():
        # This script is executed in conf.py from the docs directory
        versionfile = join(join(join(hdir, ".."),
                                name),  # noqa: F821
                           "_version_save.py")
    else:
        # This script is imported as a module
        versionfile = join(hdir, "_version_save.py")

    # Determine the accurate version
    longversion = ""

    # 1. git describe
    try:
        # Get the version using `git describe`
        longversion = git_describe()
    except BaseException:
        pass

    # 2. previously created version file
    if longversion == "":
        # Either this is not a git repository or we are in the
        # wrong git repository.
        # Get the version from the previously generated `_version_save.py`
        longversion = load_version(versionfile)

    # 3. last resort: date
    if longversion == "":
        print("Could not determine version. Reason:")
        print(traceback.format_exc())
        ctime = os.stat(__file__)[8]
        longversion = time.strftime("%Y.%m.%d-%H-%M-%S", time.gmtime(ctime))
        print("Using creation time as version: {}".format(longversion))

    if not hasattr(sys, 'frozen'):
        # Save the version to `_version_save.py` to allow distribution using
        # `python setup.py sdist`.
        # This is only done if the program is not frozen (with e.g.
        # pyinstaller),
        if longversion != load_version(versionfile):
            write_version(longversion, versionfile)

    # PEP 440-conform development version:
    version = ".post".join(longversion.split("-")[:2])
