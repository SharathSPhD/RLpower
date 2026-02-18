## Makefile for sCO2RL paper compilation (ArXiv-compatible)
##
## Usage:
##   make          — compile PDF
##   make clean    — remove auxiliary files
##   make arxiv    — create sco2rl_arxiv.zip for arXiv upload

PDF     = main.pdf
SOURCES = main.tex sec_abstract.tex sec_intro.tex sec_related.tex \
          sec_architecture.tex sec_method.tex sec_results.tex \
          sec_bugs.tex sec_conclusion.tex sco2rl.bib

LATEXMK = latexmk
FLAGS   = -pdf -bibtex -interaction=nonstopmode

.PHONY: all clean arxiv

all: $(PDF)

$(PDF): $(SOURCES) figures/*.png
	$(LATEXMK) $(FLAGS) main.tex

clean:
	$(LATEXMK) -C
	rm -f *.aux *.log *.bbl *.blg *.out *.toc *.fls *.fdb_latexmk

arxiv: clean $(PDF)
	zip -r sco2rl_arxiv.zip \
	    main.tex sec_*.tex sco2rl.bib \
	    figures/ Makefile \
	    --exclude "figures/*.xcf" \
	    --exclude "*.aux" --exclude "*.log"
	@echo "ArXiv package: sco2rl_arxiv.zip"
