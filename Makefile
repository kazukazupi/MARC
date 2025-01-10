fmt:
	poetry run black ./
	poetry run isort ./

lint:
	mypy ./