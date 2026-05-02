# Minimal Sphinx Makefile for ArchePy.
#
# Usage:
#   make html        # build HTML docs to _build/html
#   make clean       # remove _build/
#   make linkcheck   # verify external links

SPHINXOPTS    ?= -W --keep-going
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help clean html linkcheck Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf $(BUILDDIR)/ _autosummary/

# Catch-all target: route all unknown targets to Sphinx.
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
