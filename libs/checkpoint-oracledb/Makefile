.PHONY: test test_watch lint type format start-oracle stop-oracle

######################
# TESTING AND COVERAGE
######################

start-oracle:
	docker compose -f tests/compose-oracle.yml up -V --force-recreate --wait || ( \
		echo "Failed to start Oracle, printing logs..."; \
		docker compose -f tests/compose-oracle.yml logs; \
		exit 1 \
	)

stop-oracle:
	docker compose -f tests/compose-oracle.yml down

UV_RUN = ./.venv/bin/python -m

TEST ?= .
test: start-oracle
	@$(UV_RUN) pytest $(TEST)
	@EXIT_CODE=$$?; \
	make stop-oracle; \
	echo "Finished testing Oracle; Exit code: $$EXIT_CODE"; \
	exit $$EXIT_CODE

test_watch:
	make start-oracle; \
	$(UV_RUN) ptw $(TEST); \
	EXIT_CODE=$$?; \
	make stop-oracle; \
	exit $$EXIT_CODE

conformance: start-oracle
	@$(UV_RUN) tests.conformance
	@EXIT_CODE=$$?; \
	make stop-oracle; \
	echo "Finished langgraph conformance test of Oracle; Exit code: $$EXIT_CODE"; \
	exit $$EXIT_CODE


######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --name-only --relative --diff-filter=d main . | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langgraph
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	$(UV_RUN) ruff check .
	[ "$(PYTHON_FILES)" = "" ] || $(UV_RUN) ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || $(UV_RUN) ruff check --select I $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE)
	[ "$(PYTHON_FILES)" = "" ] || $(UV_RUN) mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

type:
	mkdir -p $(MYPY_CACHE) && $(UV_RUN) mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	$(UV_RUN) ruff format $(PYTHON_FILES)
	$(UV_RUN) ruff check --select I --fix $(PYTHON_FILES)
