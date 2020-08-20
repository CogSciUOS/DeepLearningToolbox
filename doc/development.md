


# Coding standards




## Coding standards for the Qt graphical user interface (`qtqui`)

In the Qt graphical user interface we deviate in some points from the
general coding standards of the toolbox to conform with the Qt
standards:
* use camel case instead of snake case for methods and attribute names

Linting the GUI
```sh
pylint --extension-pkg-whitelist=PyQt5 --method-naming-style=camelCase --attr-naming-style=camelCase --attr-naming-style=camelCase [FILE]...
```
