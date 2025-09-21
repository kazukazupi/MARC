fmt:
	poetry run black ./*.py ./alg/ ./analysis ./envs
	poetry run isort ./*.py ./alg/ ./analysis ./envs

lint:
	poetry run black --check ./*.py ./alg/ ./analysis ./envs
	poetry run isort --check ./*.py ./alg/ ./analysis ./envs
	poetry run mypy ./*.py ./alg/ ./analysis ./envs
	poetry run pflake8 ./*.py ./alg/ ./analysis ./envs
	@echo "🎉 All checks passed successfully! ✨🍰✨"

test:
	poetry run pytest