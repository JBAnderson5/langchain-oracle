.PHONY: test test_watch lint type format start-oracle stop-oracle enable-oracle-vectors setup-oracle-db

UV_RUN = ./.venv/bin/python -m

######################
# TESTING AND COVERAGE
######################

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


######################
# Local Docker DB
######################

start-oracle:
	docker compose -f tests/compose-oracle.yml up -V --force-recreate --wait || ( \
		echo "Failed to start Oracle, printing logs..."; \
		docker compose -f tests/compose-oracle.yml logs; \
		exit 1 \
	)
	$(MAKE) setup-oracle-db

stop-oracle:
	docker compose -f tests/compose-oracle.yml down
# stops DB and deletes data
remove-oracle:
	docker compose -f tests/compose-oracle.yml down -v

enable-oracle-vectors:
	@printf "%s\n" \
		"alter system set vector_memory_size=1G scope=spfile sid='*';" \
		"shutdown immediate;" \
		"startup;" \
		"exit" \
	| docker exec -i oracle-free bash -lc "export ORACLE_SID=FREE; sqlplus -s / as sysdba"
	@docker compose -f tests/compose-oracle.yml up -d --wait

setup-oracle-db:
	@printf "%s\n" \
		"ALTER USER SYSTEM DEFAULT TABLESPACE USERS;" \
		"ALTER USER SYSTEM QUOTA UNLIMITED ON USERS;" \
		"ALTER USER SYSTEM TEMPORARY TABLESPACE TEMP;" \
		"exit" \
	| docker exec -i oracle-free bash -lc "export ORACLE_SID=FREE; sqlplus -s / as sysdba"