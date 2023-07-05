# Deep Learning ToolBox Documentation

We use Sphinx to generate the documentation from the comments in the
code.


## Cross-references

Cross-reference in the documentation can make use of roles. The
following roles are used in the Deep Learning ToolBox:

* `:py:attr:`
  Reference a data attribute of an object.
* `:py:class:`
  Reference a class
* `:py:func:`
  Reference a Python function
* `:py:meth:`
  Reference a method of an object.
* `:py:mod:`
  Reference a module or package.

For details see:
* Sphinx documentation: [Cross-referencing Python objects](https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#cross-referencing-python-objects)

## Code and code blocks

Inline code examples should be included in double backticks.

Codeblock can be written indented, following a `.. code-block:: pyton`
line.  The block ends at the end of the indentation.

### Dotted names


### Checks


Check the roles used for cross-referencing:
```sh
find . -name '*.py' -exec grep ':py:' '{}' ';' | sed 's/^.*\(:py:[^:]*:\).*/\1/m' | sort | uniq -c
```


# Generating documentation from source code

See
* README.md
* TODO.ulf
* doc/Makefile
* dltb-doc.sh

## Required tools

* sphinx-autodoc-typehints
* sphinx-rtd-theme


### Debian/Ubuntu

```sh
sudo apt install python3-sphinx-rtd-theme python3-sphinx-autodoc-typehints 
```

### Conda

```sh
conda install -c conda-forge sphinx-autodoc-typehints sphinx_rtd_theme
```

## Documentation for the whole project

```sh
make -C doc html
```

## Documentation for individual files

```sh
./dltb-doc.sh dltb/base/implementation.py
```

Manual invocation:

```sh
mkdir -p tmp/cache tmp/rest tmp/html
sphinx-apidoc --force --separate --module-first --no-toc \
              --output-dir tmp/rest dltb ${EXCLUDES}
sphinx-build -b html -c doc/source -d tmp/cache tmp/rest tmp/html tmp/rest/*.rst
```
