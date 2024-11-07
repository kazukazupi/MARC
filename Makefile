fmt:
	black ./
	isort ./

lint:
	mypy ./