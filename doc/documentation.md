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

### Dotted names


### Checks


Check the roles used for cross-referencing:
```sh
find . -name '*.py' -exec grep ':py:' '{}' ';' | sed 's/^.*\(:py:[^:]*:\).*/\1/m' | sort | uniq -c
```
