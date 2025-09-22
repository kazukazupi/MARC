fmt:
	poetry run black ./*.py ./alg/ ./analysis ./envs ./tests ./utils
	poetry run isort ./*.py ./alg/ ./analysis ./envs ./tests ./utils

lint:
	poetry run black --check ./*.py ./alg/ ./analysis ./envs ./tests ./utils
	poetry run isort --check ./*.py ./alg/ ./analysis ./envs ./tests ./utils
	poetry run mypy ./*.py ./alg/ ./analysis ./envs ./tests ./utils
	poetry run pflake8 ./*.py ./alg/ ./analysis ./envs ./tests ./utils
	@echo "🎉 All checks passed successfully! ✨🍰✨"

test:
	poetry run pytest