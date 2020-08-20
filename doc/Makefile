# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXAPIDOC  = sphinx-apidoc
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = PyQtVis
SOURCEDIR     = source
BUILDDIR      = build
BASEDIR       = ..

PACKAGES = base model controller toolbox util datasource visualizations \
    tools tools.lucid tools.train tools.am tools.activation \
    network network.layers \
    qtgui qtgui.panels qtgui.widgets \
    gtkgui

RST_FILES = $(addprefix $(SOURCEDIR)/,$(addsuffix .rst,$(PACKAGES)))
# network.tests 


# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile packages clean-packages

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@echo "make: catch-all [$@]"
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

$(SOURCEDIR)/%.rst:
	@$(SPHINXAPIDOC) -o $(SOURCEDIR) $(BASEDIR)/$(subst .,/,$*)

packages: $(RST_FILES)

clean-packages:
	rm -f $(addprefix $(SOURCEDIR)/,$(addsuffix .rst,$(PACKAGES)))
