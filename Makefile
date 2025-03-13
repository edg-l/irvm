

.PHONY: check
check:
	cargo fmt --all -- --check
	cargo clippy --workspace -- -D warnings

.PHONY: test
test:
	cargo test --workspace
