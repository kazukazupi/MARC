fmt:
	poetry run black ./
	poetry run isort ./

lint:
	poetry run mypy ./
	poetry run flake8 ./

test:
	poetry run pytest