fmt:
	poetry run black ./
	poetry run isort ./

lint:
	poetry run black --check ./
	poetry run isort --check ./
	poetry run mypy ./
	poetry run flake8 ./
	@echo "\n🎉 All checks passed successfully! ✨🍰✨"

test:
	poetry run pytest