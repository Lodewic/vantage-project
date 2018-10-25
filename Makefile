all: requirements finaldataset report
.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = lvt-vantage-ai-case
PROFILE = default
PROJECT_NAME = vantage-project
PYTHON_INTERPRETER = python
R_INTERPRETER = Rscript

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: environment.yml
	conda env update --prune -f environment.yml

## Make Dataset
data: requirements finaldataset

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 vantage

# Download Data from S3 DrivenData public url
sync_data_from_s3:
	# Replaced this with hard-coded public links in get_data.py
	$(PYTHON_INTERPRETER) vantage/data/get_data.py

	# I really like the idea of syncing data with S3 though!
	# ifeq (default,$(PROFILE))
	# 	aws s3 sync s3://$(BUCKET)/data/ data/
	# else
	# 	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
	# endif

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
		conda env create -f environment.yml
else
	@echo ">>> please install conda."
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT GOALS                                                                 #
#################################################################################
report: data/interim/data_encoded.csv 
	Rscript -e "library(rmarkdown); render_site('./reports/.')"

# The final dataset is the encoded dataset as I have removed any steps of
# the classification itself. This code was not finished and thus removed.
finaldataset: data/interim/data_encoded.csv

cleandatatset: data/interim/data_cleaned.csv

encodedataset: data/interim/data_encoded.csv

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################
## This rule didn't work in Windows(?) It would not actually create environment.log
## Changed the requirements rule instead 
# environment.log: environment.yml
# 	conda env update --prune -f environment.yml
# 	touch environment.log
	
data/processed/dataset.csv:	data/interim/data_encoded.csv
	$(PYTHON_INTERPRETER) vantage/data/clean_data

data/interim/data_encoded.csv: data/interim/data_cleaned.csv
	$(PYTHON_INTERPRETER) vantage/features/encode_data.py
	
data/interim/data_cleaned.csv:
	$(PYTHON_INTERPRETER) vantage/data/clean_data.py

# One more rule to 'get' the data if it is not there?
# Or is it cleaner to just run make get_data once?

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
